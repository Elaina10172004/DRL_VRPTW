



import argparse  
import copy  
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

import json  

import math  

import os  

from pathlib import Path  



import torch  

import torch.nn.functional as F  

from torch.distributions import Categorical  

from tqdm import tqdm



from cost_utils import compute_cost  

from config import solve_defaults, train_defaults  

from evaluate import evaluate  

from local_search import time_budget_search  

from neural_policy import (

    AttentionVRPTW,  

    BatchVRPTWEnv,  

    _apply_feasibility_mask,  

    _normalize_probs,  
    bf16_autocast,

    build_dyn_features,  

    get_cand_phi_feature_dim,

    get_dyn_feature_dim,

    lexicographic_key,  

    neural_construct,  
    normalize_checkpoint_state_dict,

    pad_instances,  
    resolve_bf16_mode,

)

from vrptw_data import read_solomon, route_cost_and_feasible  





def _extract_state_dict(ckpt):

    if isinstance(ckpt, dict) and "model" in ckpt and isinstance(ckpt["model"], dict):

        return normalize_checkpoint_state_dict(ckpt["model"])
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt and isinstance(ckpt["model_state_dict"], dict):

        return normalize_checkpoint_state_dict(ckpt["model_state_dict"])

    return normalize_checkpoint_state_dict(ckpt)





def _resolve_solve_use_latent_multi_default() -> bool:
    return bool(getattr(solve_defaults, "solve_use_latent_multi", True))


def _resolve_solve_latent_multi_k_default() -> int:
    return max(1, int(getattr(solve_defaults, "latent_multi_k", 1)))


def _infer_dims_from_state(state: dict, default_embed: int | None = None):

    if default_embed is None:
        default_embed = int(getattr(train_defaults, "model_embed_dim", 128))
    embed_dim = int(default_embed)
    node_dim = 10
    dyn_dim = int(get_dyn_feature_dim())
    ctx_dim = None

    if isinstance(state, dict):
        if "node_enc.weight" in state:
            embed_dim = int(state["node_enc.weight"].shape[0])
            node_dim = int(state["node_enc.weight"].shape[1])
        elif "depot_enc.weight" in state:
            node_dim = int(state["depot_enc.weight"].shape[1])
        if "dyn_proj.weight" in state:
            dyn_dim = int(state["dyn_proj.weight"].shape[1])

        if "W_q.weight" in state:
            ctx_dim = int(state["W_q.weight"].shape[1])
        elif "end_scorer.weight" in state:
            ctx_dim = int(state["end_scorer.weight"].shape[1])

    latent_dim = 0
    if isinstance(state, dict):
        if "latent_context_proj.weight" in state:
            latent_dim = int(state["latent_context_proj.weight"].shape[1])
        elif "latent_film.0.weight" in state:
            latent_dim = int(state["latent_film.0.weight"].shape[1])
        elif ctx_dim is not None:
            latent_dim = max(0, ctx_dim - embed_dim * 3)
    return embed_dim, node_dim, dyn_dim, latent_dim


def _infer_encoder_arch_from_state(state: dict):
    """
    Infer (n_heads, n_layers, ff_dim) from checkpoint state when possible.
    Falls back to config defaults.
    """
    n_heads = int(getattr(train_defaults, "model_n_heads", 4))
    n_layers = int(getattr(train_defaults, "model_n_layers", 2))
    ff_dim_cfg = int(getattr(train_defaults, "model_ff_dim", 0))
    ff_dim = None if ff_dim_cfg <= 0 else ff_dim_cfg

    if not isinstance(state, dict):
        return n_heads, n_layers, ff_dim

    # n_layers from encoder layer keys
    layer_ids = set()
    for k in state.keys():
        if not k.startswith("encoder.layers."):
            continue
        parts = k.split(".")
        if len(parts) > 2 and parts[2].isdigit():
            layer_ids.add(int(parts[2]))
    if layer_ids:
        n_layers = max(layer_ids) + 1

    # n_heads from gated attention projection
    gate_key = "encoder.layers.0.self_attn.gate_proj.weight"
    if gate_key in state:
        n_heads = int(state[gate_key].shape[0])

    ff_key = "encoder.layers.0.ffn.lin1.weight"
    if ff_key in state:
        ff_dim = int(state[ff_key].shape[0])

    return n_heads, n_layers, ff_dim





def _infer_cand_phi_from_state(state: dict):

    use_aug = bool(getattr(solve_defaults, "use_dynamic_key_aug", False))

    cfg_dim = int(getattr(solve_defaults, "cand_phi_dim", get_cand_phi_feature_dim())) if use_aug else 0

    cfg_hidden = int(getattr(solve_defaults, "cand_phi_hidden_dim", 0))

    if not use_aug:

        return 0, 0

    if not isinstance(state, dict):

        return cfg_dim, cfg_hidden

    if "phi_proj.weight" in state:

        return int(state["phi_proj.weight"].shape[1]), 0

    if "phi_proj.0.weight" in state:

        return int(state["phi_proj.0.weight"].shape[1]), int(state["phi_proj.0.weight"].shape[0])

    return cfg_dim, cfg_hidden





def _state_has_phi_proj(state: dict) -> bool:

    return isinstance(state, dict) and any(k.startswith("phi_proj.") for k in state.keys())





def _state_has_raw_alpha(state: dict) -> bool:

    if not isinstance(state, dict):

        return False

    return any(k.startswith("raw_alpha_") for k in state.keys()) or ("raw_bias_scale" in state)


def _state_uses_legacy_concat_latent(state: dict, *, embed_dim: int, latent_dim: int) -> bool:
    if not isinstance(state, dict):
        return False
    if int(latent_dim) <= 0:
        return False
    if ("latent_context_proj.weight" in state) or any(k.startswith("latent_film.") for k in state.keys()):
        return False
    expected_ctx_dim = int(embed_dim) * 3 + int(latent_dim)
    for key in ("W_q.weight", "end_scorer.weight", "gate_ctx.weight"):
        if key in state and int(state[key].shape[1]) == expected_ctx_dim:
            return True
    return False


def _prepare_model_from_state(state, device: str, latent_dim: int | None, source: str = ""):
    state = normalize_checkpoint_state_dict(state)

    # infer embed/latent from state for robust loading

    embed_dim, node_dim, dyn_dim, inferred_latent = _infer_dims_from_state(state)
    n_heads, n_layers, ff_dim = _infer_encoder_arch_from_state(state)

    use_latent = inferred_latent if latent_dim is None else int(latent_dim)

    if source:

        if latent_dim is None and inferred_latent:

            print(
                f"[Info] inferred dims from {source}: "
                f"node_dim={node_dim}, dyn_dim={dyn_dim}, latent_dim={inferred_latent}, embed_dim={embed_dim}"
            )

        elif latent_dim is not None and inferred_latent != latent_dim:

            print(f"[Warn] checkpoint latent_dim={inferred_latent}, but requested latent_dim={latent_dim}")

    cand_phi_dim, cand_phi_hidden_dim = _infer_cand_phi_from_state(state)

    use_raw_cfg = bool(getattr(solve_defaults, "use_raw_feature_bias", False))

    state_has_raw = _state_has_raw_alpha(state)

    use_raw_feature_bias = use_raw_cfg and state_has_raw

    if source and use_raw_cfg and (not state_has_raw):

        print(f"[Info] checkpoint has no raw-feature bias head, disable raw bias for {source}")

    latent_injection_mode = "legacy_concat" if _state_uses_legacy_concat_latent(
        state,
        embed_dim=embed_dim,
        latent_dim=use_latent,
    ) else "film"
    if source and latent_injection_mode == "legacy_concat":
        print(f"[Info] using legacy concat-latent loader for {source}")

    model = AttentionVRPTW(

        node_dim=node_dim,
        dyn_dim=dyn_dim,

        embed_dim=embed_dim,
        n_heads=n_heads,
        n_layers=n_layers,
        ff_dim=ff_dim,

        latent_dim=use_latent,

        cand_phi_dim=cand_phi_dim,

        cand_phi_hidden_dim=cand_phi_hidden_dim,

        use_raw_feature_bias=use_raw_feature_bias,
        latent_injection_mode=latent_injection_mode,
        use_residual_gate=bool(getattr(train_defaults, "use_residual_gate", True)),
        residual_gate_init_bias=float(getattr(train_defaults, "residual_gate_init_bias", 2.0)),

    )

    state_has_phi = _state_has_phi_proj(state)

    expect_phi = cand_phi_dim > 0

    expect_raw = bool(use_raw_feature_bias)

    if (state_has_phi == expect_phi) and (state_has_raw == expect_raw):

        model.load_state_dict(state)

    else:

        # allow toggling dynamic-key augmentation from config without failing strict load

        model.load_state_dict(state, strict=False)

        if source:

            print(

                f"[Info] loaded with strict=False for head mismatch "

                f"(checkpoint_has_phi={state_has_phi}, expected_phi={expect_phi}, "
                f"checkpoint_has_raw={state_has_raw}, expected_raw={expect_raw})"

            )

    model.to(device)

    return model, node_dim, dyn_dim, embed_dim, inferred_latent, use_latent, cand_phi_dim, cand_phi_hidden_dim, use_raw_feature_bias



def _load_model_from_path(pt_path: str, device: str, latent_dim: int | None):

    pt_file = Path(pt_path)

    if not pt_file.exists():

        print(f"[璺宠繃] 鏈壘鍒版ā鍨嬫枃浠? {pt_path}")  

        return None

    raw_state = torch.load(pt_file, map_location=device, weights_only=False)  

    state = _extract_state_dict(raw_state)  

    model, node_dim, dyn_dim, embed_dim, inferred_latent, use_latent, cand_phi_dim, cand_phi_hidden_dim, use_raw_feature_bias = _prepare_model_from_state(

        state, device, latent_dim, source=pt_path

    )

    return {

        "model": model,  

        "state": state,  

        "node_dim": node_dim,

        "dyn_dim": dyn_dim,

        "embed_dim": embed_dim,

        "inferred_latent": inferred_latent,

        "latent_dim": use_latent,

        "cand_phi_dim": cand_phi_dim,

        "cand_phi_hidden_dim": cand_phi_hidden_dim,

        "use_raw_feature_bias": use_raw_feature_bias,

    }





def _match_model_node_feat_dim(
    node_feats: torch.Tensor,
    depot_feat: torch.Tensor,
    target_node_dim: int,
):

    tgt = int(target_node_dim)
    if tgt <= 0:
        return node_feats, depot_feat

    cur = int(node_feats.size(-1))
    if cur == tgt:
        return node_feats, depot_feat
    if cur > tgt:
        return node_feats[..., :tgt], depot_feat[..., :tgt]

    pad = tgt - cur
    node_pad = node_feats.new_zeros(*node_feats.shape[:-1], pad)
    depot_pad = depot_feat.new_zeros(*depot_feat.shape[:-1], pad)
    return torch.cat([node_feats, node_pad], dim=-1), torch.cat([depot_feat, depot_pad], dim=-1)


def _match_model_dyn_feat_dim(dyn_feats: torch.Tensor, target_dyn_dim: int):
    tgt = int(target_dyn_dim)
    if tgt <= 0:
        return dyn_feats
    cur = int(dyn_feats.size(-1))
    if cur == tgt:
        return dyn_feats
    if cur > tgt:
        return dyn_feats[..., :tgt]
    pad = tgt - cur
    pad_shape = list(dyn_feats.shape[:-1]) + [pad]
    pad_tensor = dyn_feats.new_zeros(*pad_shape)
    return torch.cat([dyn_feats, pad_tensor], dim=-1)


def _match_model_cand_phi_dim(cand_phi: torch.Tensor, target_cand_dim: int):
    tgt = int(target_cand_dim)
    if tgt <= 0:
        return cand_phi
    cur = int(cand_phi.size(-1))
    if cur == tgt:
        return cand_phi
    if cur > tgt:
        return cand_phi[..., :tgt]
    pad = tgt - cur
    pad_shape = list(cand_phi.shape[:-1]) + [pad]
    pad_tensor = cand_phi.new_zeros(*pad_shape)
    return torch.cat([cand_phi, pad_tensor], dim=-1)


def _repeat_for_batch(t: torch.Tensor, repeat_n: int):

    reps = [repeat_n] + [1] * (t.dim() - 1)

    return t.repeat(*reps)





def _count_unserved(inst, routes) -> int:

    served = set()

    for r in routes:

        served.update(r)

    return max(0, len(inst.customers) - len(served))





def _candidate_lex_key(inst, feasible: bool, routes, total_dist: float):

    

    unserved = _count_unserved(inst, routes)

    return lexicographic_key(not bool(feasible), unserved, len(routes), float(total_dist))





def _replay_single_env(env_data_1, pad_mask_1: torch.Tensor, action_prefix: list[int], device: str):

    """

    Replay one action prefix on a single-instance env and return:

    env, prev_action, done_flag.

    """

    env = BatchVRPTWEnv(env_data_1, device=device, track_routes=True)

    prev_action = torch.full((1,), env.N, dtype=torch.long, device=device)

    with torch.no_grad():

        for a in action_prefix:

            act = torch.tensor([int(a)], dtype=torch.long, device=device)

            env.step(act)

            prev_action = act

            real_counts = pad_mask_1.sum(dim=1)

            visited_counts = env.visited.sum(dim=1)

            is_done = visited_counts >= real_counts

            at_depot = env.loc == 0

            fully_done = is_done & at_depot

            if fully_done.any():

                env.finished[fully_done] = True

    done = bool(env.finished[0].item())

    return env, prev_action, done





def _extract_single_routes(env: BatchVRPTWEnv):

    routes = list(env.routes[0])

    curr = env.current_route[0]

    if curr:

        routes.append(curr)

    return [r for r in routes if len(r) > 0]


def _clone_env_state(env: BatchVRPTWEnv) -> BatchVRPTWEnv:
    """Shallow-copy static tensors and deep-copy mutable rollout state."""
    cloned = copy.copy(env)
    cloned.visited = env.visited.clone()
    cloned.finished = env.finished.clone()
    cloned.loc = env.loc.clone()
    cloned.load = env.load.clone()
    cloned.curr_dist = env.curr_dist.clone()
    cloned.total_dist = env.total_dist.clone()
    cloned.route_len = env.route_len.clone()
    cloned.veh_count = env.veh_count.clone()
    cloned.infeasible = env.infeasible.clone()
    cloned.time = env.time.clone()
    if bool(getattr(env, "track_routes", False)):
        cloned.routes = [
            [list(route) for route in env.routes[b]] for b in range(int(env.B))
        ]
        cloned.current_route = [list(env.current_route[b]) for b in range(int(env.B))]
    else:
        cloned.routes = []
        cloned.current_route = []
    return cloned


def _mark_finished_if_terminal(env: BatchVRPTWEnv, pad_mask: torch.Tensor) -> None:
    terminal = env.compute_terminal_mask(pad_mask)
    if terminal.any():
        env.finished[terminal] = True


def _prepare_single_decode_context(model, inst, *, device: str, use_bf16: bool):
    node_feats, depot_feat, pad_mask, env_data = pad_instances([inst])
    target_node_dim = int(getattr(getattr(model, "node_enc", None), "in_features", node_feats.size(-1)))
    target_dyn_dim = int(getattr(getattr(model, "dyn_proj", None), "in_features", get_dyn_feature_dim()))
    target_cand_dim = int(getattr(model, "cand_phi_dim", get_cand_phi_feature_dim()))
    node_feats, depot_feat = _match_model_node_feat_dim(node_feats, depot_feat, target_node_dim)
    node_feats = node_feats.to(device)
    depot_feat = depot_feat.to(device)
    pad_mask = pad_mask.to(device)
    env_data = {k: v.to(device) for k, v in env_data.items()}
    use_bf16 = resolve_bf16_mode(device, use_bf16, scope="solve_decode", verbose=False)

    with torch.no_grad():
        with bf16_autocast(use_bf16):
            nodes_enc, depot_emb, graph_emb, _enc_aux, _ = model.encode(node_feats, depot_feat, pad_mask)
            k_pre = model.W_k(nodes_enc)

    return {
        "node_feats": node_feats,
        "depot_feat": depot_feat,
        "pad_mask": pad_mask,
        "env_data": env_data,
        "env_data_1": {k: v[:1] for k, v in env_data.items()},
        "nodes_enc": nodes_enc,
        "depot_emb": depot_emb,
        "graph_emb": graph_emb,
        "k_pre": k_pre,
        "target_dyn_dim": target_dyn_dim,
        "target_cand_dim": target_cand_dim,
        "use_bf16": bool(use_bf16),
    }


def _prepare_single_latent(latent: torch.Tensor | None, *, device: str, dtype: torch.dtype) -> torch.Tensor | None:
    if latent is None:
        return None
    latent = latent.to(device=device, dtype=dtype)
    if latent.dim() == 1:
        latent = latent.unsqueeze(0)
    if latent.size(0) != 1:
        latent = latent[:1]
    return latent.contiguous()


def _repeat_decode_context(decode_ctx: dict, repeat_n: int) -> dict:
    repeat_n = max(1, int(repeat_n))
    if repeat_n == 1:
        return decode_ctx
    out = dict(decode_ctx)
    for key in ("node_feats", "depot_feat", "pad_mask", "nodes_enc", "depot_emb", "graph_emb", "k_pre"):
        out[key] = _repeat_for_batch(decode_ctx[key], repeat_n)
    out["env_data"] = {k: _repeat_for_batch(v, repeat_n) for k, v in decode_ctx["env_data_1"].items()}
    out["env_data_1"] = decode_ctx["env_data_1"]
    return out


def _repeat_single_env_state(env: BatchVRPTWEnv, env_data_1: dict, repeat_n: int, *, device: str) -> BatchVRPTWEnv:
    repeat_n = max(1, int(repeat_n))
    repeated_env_data = {k: _repeat_for_batch(v, repeat_n) for k, v in env_data_1.items()}
    cloned = BatchVRPTWEnv(repeated_env_data, device=device, track_routes=bool(getattr(env, "track_routes", False)))
    cloned.visited = env.visited[:1].repeat(repeat_n, 1).clone()
    cloned.finished = env.finished[:1].repeat(repeat_n).clone()
    cloned.loc = env.loc[:1].repeat(repeat_n).clone()
    cloned.load = env.load[:1].repeat(repeat_n).clone()
    cloned.curr_dist = env.curr_dist[:1].repeat(repeat_n).clone()
    cloned.total_dist = env.total_dist[:1].repeat(repeat_n).clone()
    cloned.route_len = env.route_len[:1].repeat(repeat_n).clone()
    cloned.veh_count = env.veh_count[:1].repeat(repeat_n).clone()
    cloned.infeasible = env.infeasible[:1].repeat(repeat_n).clone()
    cloned.time = env.time[:1].repeat(repeat_n).clone()
    if bool(getattr(env, "track_routes", False)):
        base_routes = [list(route) for route in env.routes[0]]
        base_current_route = list(env.current_route[0])
        cloned.routes = [[list(route) for route in base_routes] for _ in range(repeat_n)]
        cloned.current_route = [list(base_current_route) for _ in range(repeat_n)]
    else:
        cloned.routes = []
        cloned.current_route = []
    return cloned


def _compute_step_policy(
    model,
    env: BatchVRPTWEnv,
    prev_action: torch.Tensor,
    decode_ctx: dict,
    *,
    latent: torch.Tensor | None,
):
    pad_mask = decode_ctx["pad_mask"]
    mask = env.get_mask(pad_mask)
    active = (~env.finished) & (mask.sum(dim=1) > 0.5)
    if not bool(active.any().item()):
        return None

    safe_mask = mask.clone()
    safe_mask[~active] = 0
    safe_mask[~active, env.N] = 1.0
    row_sum = safe_mask.sum(dim=1, keepdim=True)
    zero_rows = row_sum.squeeze(1) <= 0
    if zero_rows.any():
        safe_mask[zero_rows] = 0
        safe_mask[zero_rows, env.N] = 1.0

    dyn_global = build_dyn_features(env)
    dyn_global = _match_model_dyn_feat_dim(dyn_global, int(decode_ctx["target_dyn_dim"]))
    cand_phi = env.get_candidate_features(pad_mask)
    cand_phi = _match_model_cand_phi_dim(cand_phi, int(decode_ctx["target_cand_dim"]))
    with bf16_autocast(bool(decode_ctx["use_bf16"])):
        logits = model.decode_step(
            decode_ctx["nodes_enc"],
            decode_ctx["node_feats"],
            decode_ctx["depot_emb"],
            decode_ctx["graph_emb"],
            dyn_global,
            prev_action,
            safe_mask,
            latent=latent,
            k_precomputed=decode_ctx["k_pre"],
            cand_phi=cand_phi,
        )
    raw_probs = F.softmax(logits, dim=1)
    probs = _normalize_probs(raw_probs, safe_mask)
    return {"safe_mask": safe_mask, "probs": probs, "logits": logits, "active": active}


def _extract_routes_from_env_row(env: BatchVRPTWEnv, row: int):
    routes = list(env.routes[row])
    curr = env.current_route[row]
    if curr:
        routes.append(curr)
    return [r for r in routes if len(r) > 0]


def _greedy_complete_from_state(
    model,
    inst,
    env: BatchVRPTWEnv,
    prev_action: torch.Tensor,
    decode_ctx: dict,
    *,
    latent: torch.Tensor | None,
    max_steps: int,
):
    with torch.no_grad():
        for _ in range(max(1, int(max_steps))):
            step_out = _compute_step_policy(
                model,
                env,
                prev_action,
                decode_ctx,
                latent=latent,
            )
            if step_out is None:
                break
            safe_mask = step_out["safe_mask"]
            active = step_out["active"]
            logits = step_out["logits"].masked_fill(safe_mask < 0.5, -1e9)
            actions = torch.argmax(logits, dim=1)
            safe_actions = actions.clone()
            safe_actions[~active] = env.N
            env.step(safe_actions)
            prev_action = safe_actions
            _mark_finished_if_terminal(env, decode_ctx["pad_mask"])
    return _normalize_solution_dict(inst, _extract_routes_from_env_row(env, 0))


def _greedy_complete_batch_from_state(
    model,
    inst,
    env: BatchVRPTWEnv,
    prev_action: torch.Tensor,
    decode_ctx: dict,
    *,
    latent: torch.Tensor | None,
    max_steps: int,
):
    with torch.no_grad():
        for _ in range(max(1, int(max_steps))):
            step_out = _compute_step_policy(
                model,
                env,
                prev_action,
                decode_ctx,
                latent=latent,
            )
            if step_out is None:
                break
            safe_mask = step_out["safe_mask"]
            active = step_out["active"]
            logits = step_out["logits"].masked_fill(safe_mask < 0.5, -1e9)
            actions = torch.argmax(logits, dim=1)
            safe_actions = actions.clone()
            safe_actions[~active] = env.N
            env.step(safe_actions)
            prev_action = safe_actions
            _mark_finished_if_terminal(env, decode_ctx["pad_mask"])

    return [
        _normalize_solution_dict(inst, _extract_routes_from_env_row(env, row))
        for row in range(int(env.B))
    ]






def refine_routes(inst, routes, ls_budget: float, alpha: float = 250.0):

    

    improved, _, dist_i, _, _, _ = time_budget_search(

        inst,

        candidates=[routes],  

        budget_sec=ls_budget,  

        alpha=alpha,  

        base_iters=max(10, len(inst.customers)),  

    )

    return improved, dist_i  







def refine_routes_batch(inst, routes_list, ls_budget: float, alpha: float = 250.0, parallel_workers: int = 1):

    improved, _, dist_i, _, _, _ = time_budget_search(

        inst,

        candidates=routes_list,

        budget_sec=ls_budget,

        alpha=alpha,

        base_iters=max(10, len(inst.customers)),

        parallel_workers=parallel_workers,

    )

    return improved, dist_i







def _prepare_finetune_batch(
    inst,
    repeat_n: int,
    device: str,
    target_node_dim: int | None = None,
):

    node_feats, depot_feat, pad_mask, env_data = pad_instances([inst])  
    if target_node_dim is not None:
        node_feats, depot_feat = _match_model_node_feat_dim(node_feats, depot_feat, int(target_node_dim))

    b_nf = _repeat_for_batch(node_feats, repeat_n).to(device)  

    b_df = _repeat_for_batch(depot_feat, repeat_n).to(device)  

    b_pm = _repeat_for_batch(pad_mask, repeat_n).to(device)  

    b_env_data = {k: _repeat_for_batch(v, repeat_n).to(device) for k, v in env_data.items()}  

    return b_nf, b_df, b_pm, b_env_data







def _evaluate_finetune_batch(
    env,
    inst,
    alpha: float,
    best_cost: float,
    latent_batch: torch.Tensor | None = None,
):

    batch_costs = []

    best_sample = None

    for i in range(len(env.routes)):

        curr_r = env.current_route[i]

        final_routes = env.routes[i]

        if curr_r:

            final_routes.append(curr_r)



        final_routes = [r for r in final_routes if len(r) > 0]

        dist_val = 0.0

        feas_all = True

        served_set = set()

        for r in final_routes:

            d, feas, _ = route_cost_and_feasible(inst, r)

            if not feas:

                feas_all = False

                dist_val = float("inf")

                served_set.update(r)

                break

            dist_val += d

            served_set.update(r)



        unserved = max(0, len(inst.customers) - len(served_set))

        cost = compute_cost(

            dist_val,

            final_routes,

            alpha=alpha,

            unserved=unserved,

            unserved_penalty=500.0,

            infeasible=not feas_all,

            infeasible_penalty=5000.0,

        )

        batch_costs.append(cost if math.isfinite(cost) else 1e8)

        if unserved == 0 and feas_all:

            vehicles = len(final_routes)

            c_feas = compute_cost(dist_val, final_routes, alpha=alpha, vehicles_override=vehicles)

            if c_feas < best_cost:

                best_cost = c_feas

                best_sample = {
                    "feasible": True,
                    "routes": final_routes,
                    "vehicles": vehicles,
                    "total_distance": dist_val,
                    "cost": c_feas,
                }
                if latent_batch is not None and i < int(latent_batch.size(0)):
                    best_sample["latent"] = [
                        float(x) for x in latent_batch[i].detach().cpu().tolist()
                    ]

    return batch_costs, best_cost, best_sample





class EASLayer(torch.nn.Module):
    """Per-instance lightweight adaptation layer (EAS-style)."""

    def __init__(self, embed_dim: int):
        super().__init__()
        self.proj1 = torch.nn.Linear(embed_dim, embed_dim, bias=True)
        self.proj2 = torch.nn.Linear(embed_dim, embed_dim, bias=True)
        # Start from identity behavior: adapted = base + 0
        torch.nn.init.zeros_(self.proj1.weight)
        torch.nn.init.zeros_(self.proj1.bias)
        torch.nn.init.zeros_(self.proj2.weight)
        torch.nn.init.zeros_(self.proj2.bias)

    def forward(self, node_emb: torch.Tensor) -> torch.Tensor:
        delta = self.proj2(F.relu(self.proj1(node_emb)))
        return node_emb + delta


class EASWrappedModel(torch.nn.Module):
    """Model wrapper that applies a trained EAS layer at encode-time."""

    def __init__(
        self,
        base_model: AttentionVRPTW,
        eas_layer: EASLayer,
        tuned_latent: torch.Tensor | None = None,
    ):
        super().__init__()
        self.base_model = base_model
        self.eas_layer = eas_layer
        self.latent_dim = getattr(base_model, "latent_dim", 0)
        self.tuned_latent = tuned_latent

    def encode(
        self,
        node_feats: torch.Tensor,
        depot_feat: torch.Tensor,
        pad_mask: torch.Tensor | None = None,
    ):
        node_emb, depot_emb, _graph_emb, aux_loss, enc_stats = self.base_model.encode(
            node_feats,
            depot_feat,
            pad_mask,
        )
        node_emb = self.eas_layer(node_emb)
        if pad_mask is not None:
            valid = (pad_mask > 0.5).to(node_emb.dtype).unsqueeze(-1)
            node_emb = node_emb * valid
            graph_pool = (node_emb * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
            graph_emb = self.base_model.graph_proj(graph_pool)
        else:
            graph_emb = self.base_model.graph_proj(node_emb.mean(dim=1))
        return node_emb, depot_emb, graph_emb, aux_loss, enc_stats

    def decode_step(self, *args, **kwargs):
        return self.base_model.decode_step(*args, **kwargs)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.base_model, name)




def eas_adapt_on_instance(

    base_state: dict,

    inst,

    *,

    device: str,

    steps: int,
    rollout_batch: int,

    lr: float,

    entropy_coef: float,

    node_dim: int,
    dyn_dim: int,

    embed_dim: int,

    latent_dim: int,

    cand_phi_dim: int | None = None,

    cand_phi_hidden_dim: int | None = None,

    use_raw_feature_bias: bool | None = None,
    use_bf16: bool | None = None,

):

    if cand_phi_dim is None:

        use_aug = bool(getattr(solve_defaults, "use_dynamic_key_aug", False))

        cand_phi_dim = int(getattr(solve_defaults, "cand_phi_dim", get_cand_phi_feature_dim())) if use_aug else 0

    if cand_phi_hidden_dim is None:

        cand_phi_hidden_dim = int(getattr(solve_defaults, "cand_phi_hidden_dim", 0))

    if use_raw_feature_bias is None:

        use_raw_feature_bias = bool(getattr(solve_defaults, "use_raw_feature_bias", False))
    if use_bf16 is None:
        use_bf16 = bool(getattr(solve_defaults, "use_bf16", True))
    use_bf16 = resolve_bf16_mode(device, use_bf16, scope="eas", verbose=False)

    n_heads, n_layers, ff_dim = _infer_encoder_arch_from_state(base_state)
    model = AttentionVRPTW(
        node_dim=node_dim,
        dyn_dim=dyn_dim,
        embed_dim=embed_dim,
        n_heads=n_heads,
        n_layers=n_layers,
        ff_dim=ff_dim,
        latent_dim=latent_dim,
        cand_phi_dim=cand_phi_dim,
        cand_phi_hidden_dim=cand_phi_hidden_dim,
        use_raw_feature_bias=bool(use_raw_feature_bias),
        use_residual_gate=bool(getattr(train_defaults, "use_residual_gate", True)),
        residual_gate_init_bias=float(getattr(train_defaults, "residual_gate_init_bias", 2.0)),
    )

    state_has_phi = _state_has_phi_proj(base_state)
    expect_phi = cand_phi_dim > 0
    state_has_raw = _state_has_raw_alpha(base_state)
    expect_raw = bool(use_raw_feature_bias)
    if (state_has_phi == expect_phi) and (state_has_raw == expect_raw):
        model.load_state_dict(base_state)
    else:
        model.load_state_dict(base_state, strict=False)
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    eas_layer = EASLayer(embed_dim=embed_dim).to(device)
    wrapped_model = EASWrappedModel(model, eas_layer).to(device)

    BK = max(1, int(rollout_batch))
    learnable_latent = None
    optimize_params = list(eas_layer.parameters())
    model_latent_dim = int(getattr(model, "latent_dim", 0))
    if model_latent_dim > 0:
        # Joint optimization: adapt EAS layer and instance latent vectors together.
        learnable_latent = torch.nn.Parameter(
            0.1 * torch.randn(BK, model_latent_dim, device=device)
        )
        optimize_params.append(learnable_latent)

    if steps <= 0:
        if learnable_latent is not None:
            wrapped_model.tuned_latent = learnable_latent.detach().mean(dim=0).clone()
        wrapped_model.eval()
        return wrapped_model, None

    b_nf, b_df, b_pm, b_env_data = _prepare_finetune_batch(inst, BK, device, target_node_dim=node_dim)
    optimizer = torch.optim.AdamW(optimize_params, lr=lr, weight_decay=1e-4)
    tqdm.write(
        f"[eas] rollout_batch={BK}, joint_latent={bool(learnable_latent is not None)}"
    )

    eas_layer.train()
    alpha = 250.0
    best_sample = None
    best_cost = float('inf')
    best_latent_vec = None

    for step in range(steps):
        env = BatchVRPTWEnv(b_env_data, device=device)
        with torch.no_grad():
            with bf16_autocast(use_bf16):
                nodes_enc_base, depot_emb, _graph_emb_base, _enc_aux, _ = model.encode(b_nf, b_df, b_pm)
        nodes_enc = eas_layer(nodes_enc_base)
        valid = (b_pm > 0.5).to(nodes_enc.dtype).unsqueeze(-1)
        nodes_enc = nodes_enc * valid
        graph_pool = (nodes_enc * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
        graph_emb = model.graph_proj(graph_pool)
        k_pre = model.W_k(nodes_enc)

        log_probs = []
        entropies = []
        prev_action = torch.full((BK,), env.N, dtype=torch.long, device=device)
        latent = learnable_latent
        batch_indices = torch.zeros(BK, dtype=torch.long, device=device)

        for _ in range(env.N * 2 + 100):
            mask = env.get_mask(b_pm)
            active = (~env.finished) & (mask.sum(dim=1) > 0.5)
            if not active.any():
                break

            safe_mask = mask.clone()
            safe_mask[~active] = 0
            safe_mask[~active, env.N] = 1.0
            safe_mask = _apply_feasibility_mask(env, safe_mask, [inst], batch_indices)

            dyn_global = build_dyn_features(env)
            target_dyn_dim = int(getattr(getattr(model, "dyn_proj", None), "in_features", dyn_global.size(-1)))
            dyn_global = _match_model_dyn_feat_dim(dyn_global, target_dyn_dim)
            with bf16_autocast(use_bf16):
                logits = model.decode_step(
                    nodes_enc,
                    b_nf,
                    depot_emb,
                    graph_emb,
                    dyn_global,
                    prev_action,
                    safe_mask,
                    latent=latent,
                    k_precomputed=k_pre,
                )
                raw_probs = F.softmax(logits, dim=1)
                probs = _normalize_probs(raw_probs, safe_mask)

            dist = Categorical(probs.float())
            actions = dist.sample()
            log_p = dist.log_prob(actions)
            entropies.append(dist.entropy())

            log_probs.append(log_p)
            safe_actions = actions.clone()
            safe_actions[~active] = env.N
            env.step(safe_actions)
            prev_action = safe_actions

            real_counts = b_pm.sum(dim=1)
            visited_counts = env.visited.sum(dim=1)
            is_done = visited_counts >= real_counts
            at_depot = env.loc == 0
            fully_done = is_done & at_depot
            if fully_done.any():
                env.finished[fully_done] = True

        if not log_probs:
            continue

        batch_costs, best_cost, step_best = _evaluate_finetune_batch(
            env,
            inst,
            alpha,
            best_cost,
            latent_batch=latent.detach() if latent is not None else None,
        )
        if step_best:
            best_sample = step_best
            if "latent" in step_best:
                best_latent_vec = torch.tensor(
                    step_best["latent"],
                    device=device,
                    dtype=b_nf.dtype,
                )

        reward = -torch.tensor(batch_costs, device=device, dtype=torch.float32)
        adv = (reward - reward.mean()) / (reward.std(unbiased=False) + 1e-6)
        sum_log_probs = torch.stack(log_probs, dim=0).sum(dim=0)
        loss = -(adv * sum_log_probs).mean()

        if entropy_coef > 0.0 and entropies:
            entropy_term = torch.stack(entropies, dim=0).mean()
            loss = loss - entropy_coef * entropy_term

        optimizer.zero_grad()
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(optimize_params, max_norm=21.0))
        optimizer.step()

        if (step + 1) % 10 == 0 or step == steps - 1:
            latent_norm = float(learnable_latent.detach().norm(dim=1).mean().item()) if learnable_latent is not None else 0.0
            tqdm.write(
                f'[eas] step={step+1}/{steps} loss={loss.item():.4f} mean_cost={-reward.mean().item():.2f} '
                f'grad_norm={grad_norm:.4f} lr={lr:.2e} latent_norm={latent_norm:.4f}'
            )

    if learnable_latent is not None:
        with torch.no_grad():
            if best_latent_vec is not None and best_latent_vec.numel() == model_latent_dim:
                wrapped_model.tuned_latent = best_latent_vec.detach().clone()
            else:
                wrapped_model.tuned_latent = learnable_latent.detach().mean(dim=0).clone()

    eas_layer.eval()
    wrapped_model.eval()
    return wrapped_model, best_sample


def finetune_on_instance(*args, **kwargs):
    """Alias for EAS single-instance adaptation."""
    return eas_adapt_on_instance(*args, **kwargs)





def _normalize_solution_dict(inst, routes):
    routes2 = [r for r in routes if len(r) > 0]
    ev = evaluate(inst, routes2)
    ev["routes"] = routes2
    ev["vehicles"] = len(routes2)
    ev["total_distance"] = float(ev.get("total_distance", 0.0))
    ev["unserved"] = _count_unserved(inst, routes2)
    ev["infeasible"] = (not bool(ev.get("feasible", False)))
    return ev


def _select_best_from_route_candidates(
    inst,
    route_candidates,
    *,
    use_local_search: bool,
    ls_parallel: int,
    ls_budget: float,
    ls_workers: int | None,
):
    if not route_candidates:
        return None

    cleaned = []
    for r in route_candidates:
        if not r:
            continue
        rr = [route for route in r if len(route) > 0]
        if rr:
            cleaned.append(rr)
    if not cleaned:
        return None

    best = None
    best_key = None
    workers = ls_workers if (ls_workers and ls_workers > 1) else (os.cpu_count() or 1)
    if bool(use_local_search) and ls_parallel and len(cleaned) > 1 and workers > 1:
        best_r, _best_d = refine_routes_batch(inst, cleaned, ls_budget, parallel_workers=workers)
        ev = _normalize_solution_dict(inst, best_r)
        key = _candidate_lex_key(inst, bool(ev.get("feasible", False)), ev["routes"], ev["total_distance"])
        return ev if key is not None else None

    for r0 in cleaned:
        if bool(use_local_search):
            r_refined, _d2 = refine_routes(inst, r0, ls_budget)
            r2 = [route for route in r_refined if len(route) > 0]
        else:
            r2 = r0
        ev = _normalize_solution_dict(inst, r2)
        key_ev = _candidate_lex_key(inst, bool(ev.get("feasible", False)), ev["routes"], ev["total_distance"])
        if (best_key is None) or (key_ev < best_key):
            best_key = key_ev
            best = ev
    return best


def lookahead_search_once(
    model,
    inst,
    *,
    device: str,
    confident_prob: float,
    top_k: int,
    max_steps: int | None = None,
    latent: torch.Tensor | None = None,
    use_bf16: bool = False,
):
    confident_prob = float(max(0.0, min(1.0, confident_prob)))
    top_k = max(1, int(top_k))

    decode_ctx = _prepare_single_decode_context(model, inst, device=device, use_bf16=use_bf16)
    latent = _prepare_single_latent(latent, device=device, dtype=decode_ctx["node_feats"].dtype)
    env = BatchVRPTWEnv(decode_ctx["env_data_1"], device=device, track_routes=True)
    prev_action = torch.full((1,), env.N, dtype=torch.long, device=device)
    max_steps = int(max_steps) if max_steps is not None else int(len(inst.customers) * 3 + 50)
    lookahead_stats = {
        "greedy_shortcuts": 0,
        "lookahead_steps": 0,
        "rollout_candidates": 0,
    }

    with torch.no_grad():
        for _ in range(max_steps):
            step_out = _compute_step_policy(
                model,
                env,
                prev_action,
                decode_ctx,
                latent=latent,
            )
            if step_out is None:
                break

            safe_mask = step_out["safe_mask"]
            probs = step_out["probs"]
            top_prob, top_action = torch.max(probs[0], dim=0)
            choose_action = int(top_action.item())

            if float(top_prob.item()) > confident_prob:
                lookahead_stats["greedy_shortcuts"] += 1
            else:
                valid_count = int((safe_mask[0] > 0.5).sum().item())
                if valid_count <= 1:
                    choose_action = int(top_action.item())
                else:
                    safe_probs = probs[0].masked_fill(safe_mask[0] < 0.5, -1.0)
                    branch_n = min(top_k, valid_count)
                    top_branch_probs, top_branch_actions = torch.topk(safe_probs, k=branch_n, dim=0)

                    branch_env = _repeat_single_env_state(
                        env,
                        decode_ctx["env_data_1"],
                        branch_n,
                        device=device,
                    )
                    branch_decode_ctx = _repeat_decode_context(decode_ctx, branch_n)
                    branch_actions = top_branch_actions.to(device=device, dtype=torch.long)
                    branch_env.step(branch_actions)
                    _mark_finished_if_terminal(branch_env, branch_decode_ctx["pad_mask"])
                    branch_latent = None
                    if latent is not None:
                        branch_latent = latent.expand(branch_n, -1).contiguous()
                    rollout_results = _greedy_complete_batch_from_state(
                        model,
                        inst,
                        branch_env,
                        branch_actions,
                        branch_decode_ctx,
                        latent=branch_latent,
                        max_steps=max_steps,
                    )

                    best_branch_key = None
                    for branch_idx, rollout_res in enumerate(rollout_results):
                        action_idx = int(top_branch_actions[branch_idx].item())
                        prob_i = float(top_branch_probs[branch_idx].item())
                        branch_cost = compute_cost(
                            float(rollout_res.get("total_distance", 0.0)),
                            rollout_res.get("routes", []),
                            unserved=int(rollout_res.get("unserved", 0)),
                            infeasible=bool(rollout_res.get("infeasible", False)),
                            vehicles_override=int(rollout_res.get("vehicles", len(rollout_res.get("routes", [])))),
                        )
                        branch_key = (
                            float(branch_cost),
                            *_candidate_lex_key(
                                inst,
                                bool(rollout_res.get("feasible", False)),
                                rollout_res.get("routes", []),
                                float(rollout_res.get("total_distance", 0.0)),
                            ),
                            -prob_i,
                        )
                        if (best_branch_key is None) or (branch_key < best_branch_key):
                            best_branch_key = branch_key
                            choose_action = int(action_idx)

                    lookahead_stats["lookahead_steps"] += 1
                    lookahead_stats["rollout_candidates"] += int(branch_n)

            action_t = torch.tensor([choose_action], dtype=torch.long, device=device)
            env.step(action_t)
            prev_action = action_t
            _mark_finished_if_terminal(env, decode_ctx["pad_mask"])

    best = _normalize_solution_dict(inst, _extract_single_routes(env))
    best["lookahead"] = lookahead_stats
    return best


def _get_model_tuned_latent(
    model,
    *,
    device: str,
    latent_dim: int,
) -> torch.Tensor | None:
    tuned = getattr(model, "tuned_latent", None)
    if tuned is None or latent_dim <= 0:
        return None
    if not torch.is_tensor(tuned):
        try:
            tuned = torch.tensor(tuned, device=device, dtype=torch.float32)
        except Exception:
            return None
    else:
        tuned = tuned.to(device=device, dtype=torch.float32)
    tuned = tuned.detach()
    if tuned.dim() == 2:
        tuned = tuned[0]
    elif tuned.dim() != 1:
        tuned = tuned.flatten()
    if tuned.numel() != latent_dim:
        return None
    return tuned.contiguous()


def _build_latent_candidates(
    model,
    *,
    device: str,
    latent_dim: int,
    random_samples: int,
):
    if latent_dim <= 0:
        return [None], {"zero_latent_injected": False, "uses_eas_latent": False}

    latents = []
    zero_latent = torch.zeros(latent_dim, device=device)
    latents.append(zero_latent)

    tuned_latent = _get_model_tuned_latent(model, device=device, latent_dim=latent_dim)
    uses_eas_latent = tuned_latent is not None
    if tuned_latent is not None and not torch.allclose(tuned_latent, zero_latent):
        latents.append(tuned_latent)

    extra_random = max(0, int(random_samples) - 1)
    for _ in range(extra_random):
        latents.append(torch.randn(latent_dim, device=device))

    return latents, {"zero_latent_injected": True, "uses_eas_latent": bool(uses_eas_latent)}


def _solve_once_core(
    model,
    inst,
    *,
    device: str,
    use_bf16: bool,
    ls_budget: float,
    use_latent_multi: bool,
    decode_mode: str,
    use_local_search: bool,
    latent_multi_k: int,
    ls_parallel: int,
    ls_workers: int | None,
    log_greedy: bool,
    decode_max_steps: int | None,
    latent_search_samples: int,
    lookahead_confident_prob: float,
    lookahead_top_k: int,
):
    decode_mode = str(decode_mode).strip().lower()
    if decode_mode not in {"greedy", "lookahead"}:
        raise ValueError(f"Unsupported decode mode: {decode_mode}")

    route_candidates = []
    model_latent_dim = int(getattr(model, "latent_dim", 0))
    latent_sample_count = max(1, int(latent_search_samples))
    latent_candidates, latent_meta = _build_latent_candidates(
        model,
        device=device,
        latent_dim=model_latent_dim,
        random_samples=latent_sample_count,
    )

    if decode_mode == "lookahead":
        for latent_vec in latent_candidates:
            sampled_latent = latent_vec.unsqueeze(0) if latent_vec is not None else None
            lookahead_res = lookahead_search_once(
                model,
                inst,
                device=device,
                confident_prob=float(lookahead_confident_prob),
                top_k=int(lookahead_top_k),
                max_steps=decode_max_steps,
                latent=sampled_latent,
                use_bf16=bool(use_bf16),
            )
            if lookahead_res is not None:
                route_candidates.append(lookahead_res.get("routes", []))
    else:
        if bool(use_latent_multi):
            rollout_n = max(1, int(latent_multi_k))
            for latent_vec in latent_candidates:
                batch_instances = [inst] * rollout_n
                latent_override = None
                if latent_vec is not None:
                    latent_override = latent_vec.unsqueeze(0).expand(rollout_n, -1).contiguous()
                sample_routes_list, _ = neural_construct(
                    batch_instances,
                    greedy=False,
                    device=device,
                    model=model,
                    latent_dim=getattr(model, "latent_dim", 0),
                    latent_override=latent_override,
                    use_bf16=bool(use_bf16),
                )
                route_candidates.extend(sample_routes_list)
        else:
            for latent_vec in latent_candidates:
                latent_override = None
                if latent_vec is not None:
                    latent_override = latent_vec.unsqueeze(0)
                routes_greedy_batch, _ = neural_construct(
                    [inst],
                    greedy=True,
                    device=device,
                    model=model,
                    latent_dim=getattr(model, "latent_dim", 0),
                    latent_override=latent_override,
                    use_bf16=bool(use_bf16),
                )
                if routes_greedy_batch:
                    route_candidates.append(routes_greedy_batch[0])

    best = _select_best_from_route_candidates(
        inst,
        route_candidates,
        use_local_search=bool(use_local_search),
        ls_parallel=int(ls_parallel),
        ls_budget=ls_budget,
        ls_workers=ls_workers,
    )
    if best is None:
        return None

    if model_latent_dim > 0:
        best = dict(best)
        best["latent_search"] = {
            "method": "random",
            "samples": int(len(latent_candidates)),
            "zero_latent_injected": bool(latent_meta.get("zero_latent_injected", True)),
            "uses_eas_latent": bool(latent_meta.get("uses_eas_latent", False)),
        }

    if log_greedy and best is not None and decode_mode == "greedy" and (not bool(use_latent_multi)):
        inst_name = getattr(inst, "name", "")
        prefix = f"[greedy {inst_name}]" if inst_name else "[greedy]"
        veh = int(best.get("vehicles", len(best.get("routes", []))))
        dist = float(best.get("total_distance", 0.0))
        c_g = compute_cost(dist, best.get("routes", []), vehicles_override=veh)
        print(f"{prefix} vehicles={veh}, dist={dist:.3f}, cost={c_g:.3f}")

    return best


def _transform_xy_by_aug8(x: float, y: float, cx: float, cy: float, aug_id: int):
    dx = float(x) - float(cx)
    dy = float(y) - float(cy)
    aug_id = int(aug_id) % 8
    if aug_id == 0:
        tx, ty = dx, dy
    elif aug_id == 1:
        tx, ty = -dx, dy
    elif aug_id == 2:
        tx, ty = dx, -dy
    elif aug_id == 3:
        tx, ty = -dx, -dy
    elif aug_id == 4:
        tx, ty = dy, dx
    elif aug_id == 5:
        tx, ty = -dy, dx
    elif aug_id == 6:
        tx, ty = dy, -dx
    else:
        tx, ty = -dy, -dx
    return float(cx + tx), float(cy + ty)


def _build_aug8_instances(inst):
    all_pts = [(float(inst.depot.x), float(inst.depot.y))] + [(float(c.x), float(c.y)) for c in inst.customers]
    cx = sum(p[0] for p in all_pts) / float(len(all_pts))
    cy = sum(p[1] for p in all_pts) / float(len(all_pts))
    out = []
    for aug_id in range(8):
        aug = copy.deepcopy(inst)
        aug.name = f"{inst.name}_aug{aug_id}"
        dx, dy = _transform_xy_by_aug8(inst.depot.x, inst.depot.y, cx, cy, aug_id)
        aug.depot.x = dx
        aug.depot.y = dy
        for i, c in enumerate(inst.customers):
            x2, y2 = _transform_xy_by_aug8(c.x, c.y, cx, cy, aug_id)
            aug.customers[i].x = x2
            aug.customers[i].y = y2
        out.append(aug)
    return out


def _solve_aug8_latent_multi_greedy_vectorized(
    model,
    inst,
    *,
    device: str,
    use_bf16: bool,
    ls_budget: float,
    use_local_search: bool,
    latent_multi_k: int,
    ls_parallel: int,
    ls_workers: int | None,
    latent_search_samples: int,
):
    # Single-batch vectorized greedy solve: aug8_count * latent_multi_k candidates per latent sample.
    model_latent_dim = int(getattr(model, "latent_dim", 0))
    latent_candidates, latent_meta = _build_latent_candidates(
        model,
        device=device,
        latent_dim=model_latent_dim,
        random_samples=max(1, int(latent_search_samples)),
    )
    aug_instances = _build_aug8_instances(inst)
    rollout_n = max(1, int(latent_multi_k))
    batch_per_latent = int(len(aug_instances) * rollout_n)
    print(
        f"[aug8-batch] vectorized greedy: aug={len(aug_instances)}, "
        f"latent_multi_k={rollout_n}, batch_per_latent={batch_per_latent}"
    )

    route_candidates = []
    for latent_vec in latent_candidates:
        batch_instances = [aug_inst for aug_inst in aug_instances for _ in range(rollout_n)]
        latent_override = None
        if latent_vec is not None:
            latent_override = latent_vec.unsqueeze(0).expand(batch_per_latent, -1).contiguous()
        sample_routes_list, _ = neural_construct(
            batch_instances,
            greedy=False,
            device=device,
            model=model,
            latent_dim=getattr(model, "latent_dim", 0),
            latent_override=latent_override,
            use_bf16=bool(use_bf16),
        )
        route_candidates.extend(sample_routes_list)

    best = _select_best_from_route_candidates(
        inst,
        route_candidates,
        use_local_search=bool(use_local_search),
        ls_parallel=int(ls_parallel),
        ls_budget=ls_budget,
        ls_workers=ls_workers,
    )
    if best is None:
        return None

    if model_latent_dim > 0:
        best = dict(best)
        best["latent_search"] = {
            "method": "random",
            "samples": int(len(latent_candidates)),
            "zero_latent_injected": bool(latent_meta.get("zero_latent_injected", True)),
            "uses_eas_latent": bool(latent_meta.get("uses_eas_latent", False)),
        }
    return best


def solve_once(
    model,
    inst,
    device: str,
    use_bf16: bool,
    ls_budget: float,
    latent_multi_k: int = 1,
    ls_parallel: int = 0,
    ls_workers: int | None = None,
    log_greedy: bool = False,
    use_latent_multi: bool = True,
    decode_mode: str = "greedy",
    use_aug8: bool = False,
    use_local_search: bool = True,
    decode_max_steps: int | None = None,
    latent_search_samples: int = 1,
    lookahead_confident_prob: float = 0.95,
    lookahead_top_k: int = 3,
    parallel_aug8_workers: int = 1,
):
    decode_mode = str(decode_mode).strip().lower()
    if decode_mode not in {"greedy", "lookahead"}:
        raise ValueError(f"Unsupported decode mode: {decode_mode}")

    if bool(use_aug8) and decode_mode == "greedy" and bool(use_latent_multi):
        return _solve_aug8_latent_multi_greedy_vectorized(
            model,
            inst,
            device=device,
            use_bf16=bool(use_bf16),
            ls_budget=ls_budget,
            use_local_search=bool(use_local_search),
            latent_multi_k=max(1, int(latent_multi_k)),
            ls_parallel=int(ls_parallel),
            ls_workers=ls_workers,
            latent_search_samples=max(1, int(latent_search_samples)),
        )

    if not bool(use_aug8):
        return _solve_once_core(
            model,
            inst,
            device=device,
            use_bf16=bool(use_bf16),
            ls_budget=ls_budget,
            use_latent_multi=bool(use_latent_multi),
            decode_mode=str(decode_mode),
            use_local_search=bool(use_local_search),
            latent_multi_k=max(1, int(latent_multi_k)),
            ls_parallel=int(ls_parallel),
            ls_workers=ls_workers,
            log_greedy=bool(log_greedy),
            decode_max_steps=decode_max_steps,
            latent_search_samples=max(1, int(latent_search_samples)),
            lookahead_confident_prob=float(lookahead_confident_prob),
            lookahead_top_k=int(lookahead_top_k),
        )

    best = None
    best_key = None
    aug_instances = _build_aug8_instances(inst)
    aug_workers = max(1, int(parallel_aug8_workers))

    def _run_one_aug(aug_inst):
        return _solve_once_core(
            model,
            aug_inst,
            device=device,
            use_bf16=bool(use_bf16),
            ls_budget=ls_budget,
            use_latent_multi=bool(use_latent_multi),
            decode_mode=str(decode_mode),
            use_local_search=bool(use_local_search),
            latent_multi_k=max(1, int(latent_multi_k)),
            ls_parallel=int(ls_parallel),
            ls_workers=ls_workers,
            log_greedy=False,
            decode_max_steps=decode_max_steps,
            latent_search_samples=max(1, int(latent_search_samples)),
            lookahead_confident_prob=float(lookahead_confident_prob),
            lookahead_top_k=int(lookahead_top_k),
        )

    if aug_workers > 1 and len(aug_instances) > 1:
        workers = min(aug_workers, len(aug_instances))
        with ThreadPoolExecutor(max_workers=workers) as ex:
            aug_results = list(ex.map(_run_one_aug, aug_instances))
    else:
        aug_results = [_run_one_aug(aug_inst) for aug_inst in aug_instances]

    for aug_res in aug_results:
        if aug_res is None:
            continue
        mapped = _normalize_solution_dict(inst, aug_res.get("routes", []))
        if isinstance(aug_res, dict) and ("latent_search" in aug_res):
            mapped["latent_search"] = copy.deepcopy(aug_res.get("latent_search"))
        key = _candidate_lex_key(inst, bool(mapped.get("feasible", False)), mapped["routes"], mapped["total_distance"])
        if (best_key is None) or (key < best_key):
            best_key = key
            best = mapped
    return best


def extract_routes_from_solution(sol: dict, sol_name: str | None = None):
    if "routes" in sol:
        return sol["routes"], sol.get("total_distance")
    if not sol:
        raise ValueError("Empty solution file")
    if sol_name is None and sol:
        sol_name = next(iter(sol.keys()))
    lookup = {k.lower(): k for k in sol.keys()}
    key = lookup.get(sol_name.lower()) if sol_name else None
    if key is None or "final" not in sol[key] or "routes" not in sol[key]["final"]:
        raise ValueError("solution JSON missing final/routes or name mismatch")
    final = sol[key]["final"]
    return final["routes"], final.get("total_distance")


def solve_with_model(
    pt_path: str,
    inst,
    device: str,
    ls_budget: float,
    latent_multi_k: int,
    use_bf16: bool | None = None,
    latent_dim: int | None = None,
):
    if use_bf16 is None:
        use_bf16 = bool(getattr(solve_defaults, "use_bf16", True))
    use_bf16 = resolve_bf16_mode(device, use_bf16, scope="solve_with_model", verbose=False)
    info = _load_model_from_path(pt_path, device=device, latent_dim=latent_dim)
    if not info:
        return None
    info["model"].eval()
    best = solve_once(
        info["model"],
        inst,
        device=device,
        use_bf16=bool(use_bf16),
        ls_budget=ls_budget,
        latent_multi_k=latent_multi_k,
        use_latent_multi=_resolve_solve_use_latent_multi_default(),
        decode_mode=str(getattr(solve_defaults, "solve_decode_mode", "greedy")),
        use_aug8=bool(getattr(solve_defaults, "solve_use_aug8", False)),
        use_local_search=bool(getattr(solve_defaults, "solve_use_local_search", True)),
        ls_parallel=int(getattr(solve_defaults, "ls_parallel", True)),
        ls_workers=getattr(solve_defaults, "ls_parallel_workers", None),
        decode_max_steps=getattr(solve_defaults, "decode_max_steps", None),
        latent_search_samples=int(getattr(solve_defaults, "latent_search_samples", 1)),
        lookahead_confident_prob=float(getattr(solve_defaults, "lookahead_confident_prob", 0.95)),
        lookahead_top_k=int(getattr(solve_defaults, "lookahead_top_k", 3)),
        parallel_aug8_workers=int(getattr(solve_defaults, "solve_parallel_aug8_workers", 1)),
    )
    return best, info


def _pick_best(inst, cands):
    best_key = None
    best_label = None
    best_res = None
    for label, res in cands:
        if res is None:
            continue
        routes = res.get("routes", [])
        dist = float(res.get("total_distance", 0.0))
        feasible = bool(res.get("feasible", False))
        key = _candidate_lex_key(inst, feasible, routes, dist)
        vehicles = int(len(routes))
        c = compute_cost(dist, routes, vehicles_override=vehicles)
        print(f"[{label}] vehicles={vehicles}, dist={dist:.3f}, cost={c:.3f}")
        if (best_key is None) or (key < best_key):
            best_key = key
            best_label = label
            best_res = res
    return best_label, best_res, best_key


def _collect_instances(args):
    inst_paths = []
    if args.instances_dir:
        dir_path = Path(args.instances_dir)
        if not dir_path.exists():
            raise ValueError(f"instances_dir not found: {dir_path}")
        inst_paths = sorted(str(p) for p in dir_path.glob("*.txt"))
    elif args.instance:
        inst_paths = [args.instance]
    if not inst_paths:
        raise ValueError("Please provide --instance or --instances_dir")
    return inst_paths


def _write_summary(out_all: str, summary_out: dict):
    out_all_path = Path(out_all)
    if out_all_path.exists():
        with open(out_all_path, "r", encoding="utf-8") as f:
            try:
                existing = json.load(f)
            except json.JSONDecodeError:
                existing = {}
    else:
        existing = {}
    existing.update(summary_out)
    with open(out_all_path, "w", encoding="utf-8") as f:
        json.dump(existing, f, ensure_ascii=False, indent=2)
    print(f"[info] wrote summary: {out_all_path}")


def _solve_single_instance(inst, inst_name: str, args, base_info, model_path: str, idx: int, total: int):
    if args.n_customers is not None:
        inst.customers = inst.customers[: args.n_customers]

    candidates = []
    process_best = {}
    solve_res = solve_once(
        base_info["model"],
        inst,
        device=args.device,
        use_bf16=bool(args.use_bf16),
        ls_budget=args.ls_budget_sec,
        latent_multi_k=max(1, args.latent_multi_k),
        ls_parallel=int(args.ls_parallel),
        ls_workers=args.ls_parallel_workers,
        log_greedy=True,
        use_latent_multi=bool(args.solve_use_latent_multi),
        decode_mode=str(args.solve_decode_mode),
        use_aug8=bool(args.solve_use_aug8),
        use_local_search=bool(args.solve_use_local_search),
        decode_max_steps=args.decode_max_steps,
        latent_search_samples=max(1, int(args.latent_search_samples)),
        lookahead_confident_prob=float(args.lookahead_confident_prob),
        lookahead_top_k=int(args.lookahead_top_k),
        parallel_aug8_workers=max(1, int(args.parallel_aug8_workers)),
    )
    if solve_res:
        candidates.append((model_path, solve_res))
        process_best["base_solve"] = copy.deepcopy(solve_res)

    _best_label, best_overall, _best_key = _pick_best(inst, candidates) if candidates else (None, None, None)

    if bool(args.solve_use_eas) and args.eas_steps > 0:
        eas_model, eas_best = eas_adapt_on_instance(
            base_state=base_info["state"],
            inst=inst,
            device=args.device,
            steps=args.eas_steps,
            rollout_batch=max(1, int(args.eas_rollout_batch)),
            lr=args.eas_lr,
            entropy_coef=args.eas_entropy_coef,
            node_dim=base_info.get("node_dim", 6),
            dyn_dim=base_info.get("dyn_dim", int(get_dyn_feature_dim())),
            embed_dim=base_info["embed_dim"],
            latent_dim=base_info["latent_dim"],
            cand_phi_dim=base_info.get("cand_phi_dim", get_cand_phi_feature_dim()),
            cand_phi_hidden_dim=base_info.get("cand_phi_hidden_dim", 0),
            use_raw_feature_bias=base_info.get("use_raw_feature_bias", bool(getattr(solve_defaults, "use_raw_feature_bias", False))),
            use_bf16=bool(args.use_bf16),
        )
        eas_res = solve_once(
            eas_model,
            inst,
            device=args.device,
            use_bf16=bool(args.use_bf16),
            ls_budget=args.ls_budget_sec,
            latent_multi_k=max(1, args.latent_multi_k),
            ls_parallel=int(args.ls_parallel),
            ls_workers=args.ls_parallel_workers,
            log_greedy=False,
            use_latent_multi=bool(args.solve_use_latent_multi),
            decode_mode=str(args.solve_decode_mode),
            use_aug8=bool(args.solve_use_aug8),
            use_local_search=bool(args.solve_use_local_search),
            decode_max_steps=args.decode_max_steps,
            latent_search_samples=max(1, int(args.latent_search_samples)),
            lookahead_confident_prob=float(args.lookahead_confident_prob),
            lookahead_top_k=int(args.lookahead_top_k),
            parallel_aug8_workers=max(1, int(args.parallel_aug8_workers)),
        )
        if eas_res:
            candidates.append((f"{model_path} (eas_solve)", eas_res))
            process_best["eas_solve"] = copy.deepcopy(eas_res)
        if eas_best:
            candidates.append((f"{model_path} (eas_sample)", eas_best))
            process_best["eas_inner_best"] = copy.deepcopy(eas_best)
        if eas_res or eas_best:
            _best_label, best_overall, _best_key = _pick_best(inst, candidates)

    if best_overall:
        best_overall = copy.deepcopy(best_overall)
        if process_best:
            best_overall["process_best"] = process_best
        best_cost = compute_cost(
            float(best_overall.get("total_distance", 0.0)),
            best_overall.get("routes", []),
            vehicles_override=int(best_overall.get("vehicles", len(best_overall.get("routes", [])))),
        )
        print(
            f"[{idx}/{total}] choose {inst_name}: vehicles={best_overall['vehicles']}, dist={best_overall['total_distance']:.3f}, cost={best_cost:.3f}"
        )
    else:
        print(f"[{idx}/{total}] {inst_name}: no feasible solution")
    return best_overall


def main():
    ap = argparse.ArgumentParser()
    default_use_eas = int(getattr(solve_defaults, "solve_use_eas", False))
    default_eas_steps = int(getattr(solve_defaults, "eas_steps", 10))
    default_eas_rollout_batch = int(getattr(solve_defaults, "eas_rollout_batch", 64))
    default_eas_lr = float(getattr(solve_defaults, "eas_lr", 5e-5))
    default_eas_entropy = float(getattr(solve_defaults, "eas_entropy_coef", 0.001))
    ap.add_argument("--instance", default=solve_defaults.instance, help="single instance path")
    ap.add_argument("--instances_dir", default=solve_defaults.instances_dir, help="batch solve dir (overrides --instance)")
    ap.add_argument("--pt", default=solve_defaults.pt, help="model path")
    ap.add_argument("--out_all", default=solve_defaults.out_all, help="output summary JSON")
    ap.add_argument("--n_customers", type=int, default=solve_defaults.n_customers, help="optional customer truncation")

    ap.add_argument(
        "--solve_use_latent_multi",
        type=int,
        default=int(_resolve_solve_use_latent_multi_default()),
        choices=[0, 1],
        help="switch1: 1=latent 多样化并行解码, 0=single",
    )
    ap.add_argument("--solve_decode_mode", type=str, default=str(getattr(solve_defaults, "solve_decode_mode", "greedy")), choices=["greedy", "lookahead"], help="switch2: greedy / lookahead")
    ap.add_argument("--solve_use_aug8", type=int, default=int(getattr(solve_defaults, "solve_use_aug8", False)), choices=[0, 1], help="switch3: aug8 on/off")
    ap.add_argument("--solve_use_local_search", type=int, default=int(getattr(solve_defaults, "solve_use_local_search", True)), choices=[0, 1], help="switch4: local search on/off")
    ap.add_argument("--solve_use_eas", dest="solve_use_eas", type=int, default=default_use_eas, choices=[0, 1], help="switch5: single-instance EAS on/off")

    ap.add_argument("--ls_budget_sec", type=float, default=solve_defaults.ls_budget_sec, help="local search budget (sec)")
    ap.add_argument("--ls_parallel", type=int, default=int(getattr(solve_defaults, "ls_parallel", True)), help="local search parallel (0/1)")
    ap.add_argument("--ls_parallel_workers", type=int, default=getattr(solve_defaults, "ls_parallel_workers", None), help="local search worker count")
    ap.add_argument(
        "--latent_multi_k",
        type=int,
        default=_resolve_solve_latent_multi_k_default(),
        help="latent 多样化并行 rollout 数",
    )
    ap.add_argument("--decode_max_steps", type=int, default=getattr(solve_defaults, "decode_max_steps", None), help="greedy/lookahead max decode steps")
    ap.add_argument("--lookahead_confident_prob", type=float, default=float(getattr(solve_defaults, "lookahead_confident_prob", 0.95)), help="lookahead: direct greedy threshold")
    ap.add_argument("--lookahead_top_k", type=int, default=int(getattr(solve_defaults, "lookahead_top_k", 3)), help="lookahead: branch only the top-k actions when not confident")
    ap.add_argument("--latent_search_samples", type=int, default=int(getattr(solve_defaults, "latent_search_samples", 1)), help="latent samples per decode run (1 disables extra latent search)")
    ap.add_argument(
        "--parallel_aug8_workers",
        type=int,
        default=int(getattr(solve_defaults, "solve_parallel_aug8_workers", 1)),
        help="parallel workers for aug8 runs (ignored in vectorized greedy+latent_multi+aug8 mode)",
    )
    ap.add_argument(
        "--parallel_instance_workers",
        type=int,
        default=int(getattr(solve_defaults, "solve_parallel_instance_workers", 1)),
        help="parallel workers across instances (each worker keeps its own model on GPU)",
    )

    ap.add_argument("--device", default=solve_defaults.device)
    ap.add_argument(
        "--use_bf16",
        type=int,
        default=int(getattr(solve_defaults, "use_bf16", True)),
        choices=[0, 1],
        help="Enable CUDA BF16 autocast (auto fallback to fp32 when unsupported).",
    )
    ap.add_argument("--latent_dim", type=int, default=solve_defaults.latent_dim, help="latent dim override")
    ap.add_argument("--eas_steps", dest="eas_steps", type=int, default=default_eas_steps, help="single-instance EAS steps")
    ap.add_argument("--eas_rollout_batch", dest="eas_rollout_batch", type=int, default=default_eas_rollout_batch, help="parallel rollouts per EAS step")
    ap.add_argument("--eas_lr", dest="eas_lr", type=float, default=default_eas_lr, help="single-instance EAS lr")
    ap.add_argument("--eas_entropy_coef", dest="eas_entropy_coef", type=float, default=default_eas_entropy, help="single-instance EAS entropy")

    args = ap.parse_args()
    args.solve_decode_mode = str(args.solve_decode_mode).strip().lower()
    args.latent_search_samples = max(1, int(args.latent_search_samples))
    args.parallel_aug8_workers = max(1, int(args.parallel_aug8_workers))
    args.parallel_instance_workers = max(1, int(args.parallel_instance_workers))
    args.eas_rollout_batch = max(1, int(args.eas_rollout_batch))
    args.latent_multi_k = max(1, int(args.latent_multi_k))
    args.lookahead_confident_prob = float(max(0.0, min(1.0, float(args.lookahead_confident_prob))))
    args.lookahead_top_k = max(1, int(args.lookahead_top_k))
    args.use_bf16 = int(resolve_bf16_mode(args.device, bool(args.use_bf16), scope="solve", verbose=True))
    solve_defaults.solve_use_latent_multi = bool(args.solve_use_latent_multi)
    solve_defaults.latent_multi_k = int(args.latent_multi_k)
    solve_defaults.use_bf16 = bool(args.use_bf16)
    solve_defaults.solve_parallel_instance_workers = int(args.parallel_instance_workers)
    solve_defaults.lookahead_confident_prob = float(args.lookahead_confident_prob)
    solve_defaults.lookahead_top_k = int(args.lookahead_top_k)
    solve_defaults.decode_max_steps = args.decode_max_steps

    inst_paths = _collect_instances(args)

    model_path = args.pt
    if not model_path:
        raise ValueError("Please provide --pt")

    summary_out = {}
    total = len(inst_paths)
    if total <= 1 or int(args.parallel_instance_workers) <= 1:
        base_info = _load_model_from_path(model_path, device=args.device, latent_dim=args.latent_dim)
        if base_info is None:
            raise ValueError(f"model path invalid or load failed: {model_path}")
        base_info["model"].eval()
        for idx, inst_path in enumerate(tqdm(inst_paths, desc="Solving"), start=1):
            inst = read_solomon(inst_path)
            inst_name = Path(inst_path).stem
            best_overall = _solve_single_instance(inst, inst_name, args, base_info, model_path, idx, total)
            if best_overall:
                summary_out[inst_name] = best_overall
    else:
        workers = min(int(args.parallel_instance_workers), total)
        print(
            f"[parallel-instance] workers={workers}, total_instances={total}, "
            f"latent_multi_k={int(args.latent_multi_k)}, aug8={int(args.solve_use_aug8)}"
        )
        thread_state = threading.local()

        def _solve_task(task: tuple[int, str]):
            idx, inst_path = task
            inst = read_solomon(inst_path)
            inst_name = Path(inst_path).stem
            base_info_local = getattr(thread_state, "base_info", None)
            if base_info_local is None:
                base_info_local = _load_model_from_path(model_path, device=args.device, latent_dim=args.latent_dim)
                if base_info_local is None:
                    raise ValueError(f"model path invalid or load failed: {model_path}")
                base_info_local["model"].eval()
                thread_state.base_info = base_info_local
            best = _solve_single_instance(inst, inst_name, args, base_info_local, model_path, idx, total)
            return inst_name, best

        tasks = [(idx, inst_path) for idx, inst_path in enumerate(inst_paths, start=1)]
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(_solve_task, task) for task in tasks]
            for fut in tqdm(as_completed(futures), total=len(futures), desc="Solving"):
                inst_name, best_overall = fut.result()
                if best_overall:
                    summary_out[inst_name] = best_overall

    if args.out_all and summary_out:
        _write_summary(args.out_all, summary_out)


if __name__ == "__main__":
    main()



