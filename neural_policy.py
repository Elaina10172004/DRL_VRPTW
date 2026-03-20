


import math
import os
import copy
import csv
import random
from contextlib import nullcontext
from typing import List, Tuple, Optional, Dict, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, MultiStepLR
from tqdm import tqdm

from cost_utils import compute_cost
from vrptw_data import Instance, route_cost_and_feasible
from config import train_defaults


def _device_is_cuda(device: str | torch.device) -> bool:
    if isinstance(device, torch.device):
        return device.type == "cuda"
    dev = str(device).strip().lower()
    return dev.startswith("cuda")


def resolve_bf16_mode(
    device: str | torch.device,
    requested: bool,
    *,
    scope: str = "runtime",
    verbose: bool = True,
) -> bool:
    if not bool(requested):
        return False
    if not _device_is_cuda(device):
        if verbose:
            print(f"[bf16] {scope}: device={device} is not CUDA, fallback to fp32.")
        return False
    if not torch.cuda.is_available():
        if verbose:
            print(f"[bf16] {scope}: CUDA unavailable, fallback to fp32.")
        return False
    is_supported = True
    if hasattr(torch.cuda, "is_bf16_supported"):
        try:
            is_supported = bool(torch.cuda.is_bf16_supported())
        except Exception:
            is_supported = True
    if not is_supported:
        if verbose:
            print(f"[bf16] {scope}: current GPU does not support bf16, fallback to fp32.")
        return False
    if verbose:
        print(f"[bf16] {scope}: enabled on {device}.")
    return True


def bf16_autocast(enabled: bool):
    if not bool(enabled):
        return nullcontext()
    if hasattr(torch, "autocast"):
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return torch.cuda.amp.autocast(dtype=torch.bfloat16)


def set_seed(seed: int = 42):
    
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)



set_seed(42)



class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        
        return x + self.pe[:, : x.size(1)]


class FeedForwardBlock(nn.Module):
    def __init__(self, embed_dim: int, ff_dim: int):
        super().__init__()
        self.lin1 = nn.Linear(embed_dim, ff_dim)
        self.lin2 = nn.Linear(ff_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(F.relu(self.lin1(x)))


class POMOInstanceNorm(nn.Module):
    """
    POMO-style instance norm on sequence embeddings.
    Input/Output shape: [B, N, E] (batch_first=True) or [N, B, E] (batch_first=False).
    """

    def __init__(self, d_model: int, eps: float = 1e-5, batch_first: bool = True):
        super().__init__()
        self.eps = float(eps)
        self.batch_first = bool(batch_first)
        self.weight = nn.Parameter(torch.ones(int(d_model)))
        self.bias = nn.Parameter(torch.zeros(int(d_model)))

    def forward(self, x: torch.Tensor, padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"POMOInstanceNorm expects a 3D tensor, got shape={tuple(x.shape)}")

        x_bne = x if self.batch_first else x.transpose(0, 1)

        if padding_mask is None:
            # Strict POMO path: transpose to [B, E, N], instance-normalize, transpose back.
            weight = self.weight.to(dtype=x_bne.dtype, device=x_bne.device)
            bias = self.bias.to(dtype=x_bne.dtype, device=x_bne.device)
            out = F.instance_norm(
                x_bne.transpose(1, 2),
                running_mean=None,
                running_var=None,
                weight=weight,
                bias=bias,
                use_input_stats=True,
                momentum=0.1,
                eps=self.eps,
            ).transpose(1, 2)
        else:
            if padding_mask.dim() != 2:
                raise ValueError(f"padding_mask must be [B,N], got shape={tuple(padding_mask.shape)}")
            if padding_mask.size(0) != x_bne.size(0) or padding_mask.size(1) != x_bne.size(1):
                raise ValueError(
                    f"padding_mask shape mismatch: mask={tuple(padding_mask.shape)}, input={tuple(x_bne.shape)}"
                )

            valid = (~padding_mask.to(device=x_bne.device, dtype=torch.bool)).to(dtype=x_bne.dtype).unsqueeze(-1)
            valid_count = valid.sum(dim=1, keepdim=True).clamp_min(1.0)

            mean = (x_bne * valid).sum(dim=1, keepdim=True) / valid_count
            centered = (x_bne - mean) * valid
            var = (centered * centered).sum(dim=1, keepdim=True) / valid_count
            inv_std = torch.rsqrt(var + self.eps)

            out = centered * inv_std
            weight = self.weight.to(dtype=x_bne.dtype, device=x_bne.device).view(1, 1, -1)
            bias = self.bias.to(dtype=x_bne.dtype, device=x_bne.device).view(1, 1, -1)
            out = out * weight + bias
            # Keep padded tokens neutral so they do not leak into following layers.
            out = out * valid

        if not self.batch_first:
            out = out.transpose(0, 1)
        return out


class HeadGatedMultiheadAttention(nn.MultiheadAttention):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        add_bias_kv: bool = False,
        add_zero_attn: bool = False,
        kdim: Optional[int] = None,
        vdim: Optional[int] = None,
        batch_first: bool = False,
        use_gated_attn: bool = True,
        gated_attn_init_bias: float = 2.0,
        alpha_attn_gate: float = 1.0,
    ):
        super().__init__(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            bias=bias,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
            kdim=kdim,
            vdim=vdim,
            batch_first=batch_first,
        )
        self.use_gated_attn = bool(use_gated_attn)
        self.alpha_attn_gate = float(alpha_attn_gate)
        self.gate_proj = nn.Linear(self.embed_dim, self.num_heads)
        nn.init.zeros_(self.gate_proj.weight)
        nn.init.constant_(self.gate_proj.bias, float(gated_attn_init_bias))
        self.last_gated_attn_mean: Optional[torch.Tensor] = None
        self.last_gated_attn_sparsity: Optional[torch.Tensor] = None

    def _expand_to_sdpa_mask(
        self,
        attn_mask: Optional[torch.Tensor],
        key_padding_mask: Optional[torch.Tensor],
        bsz: int,
        tgt_len: int,
        src_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        allowed_mask: Optional[torch.Tensor] = None
        additive_mask: Optional[torch.Tensor] = None

        if key_padding_mask is not None:
            kpm = key_padding_mask.to(device=device)
            if kpm.dtype == torch.bool:
                # MHA bool key_padding_mask: True means ignored -> convert to SDPA allowed-mask.
                kpm_allowed = (~kpm).view(bsz, 1, 1, src_len)
                allowed_mask = kpm_allowed if allowed_mask is None else (allowed_mask & kpm_allowed)
            else:
                # Float key_padding_mask keeps additive semantics.
                kpm_add = kpm.to(dtype=dtype).view(bsz, 1, 1, src_len)
                additive_mask = kpm_add if additive_mask is None else (additive_mask + kpm_add)

        if attn_mask is not None:
            am = attn_mask.to(device=device)
            if am.dtype == torch.bool:
                # MHA bool attn_mask: True means disallow -> invert for SDPA (True means allowed).
                am_allowed = ~am
                if am_allowed.dim() == 2:
                    am_allowed = am_allowed.view(1, 1, tgt_len, src_len)
                elif am_allowed.dim() == 3:
                    if am_allowed.size(0) == bsz * self.num_heads:
                        am_allowed = am_allowed.view(bsz, self.num_heads, tgt_len, src_len)
                    elif am_allowed.size(0) == bsz:
                        am_allowed = am_allowed.view(bsz, 1, tgt_len, src_len)
                    elif am_allowed.size(0) == 1:
                        am_allowed = am_allowed.view(1, 1, tgt_len, src_len)
                    else:
                        raise ValueError(f"Unsupported bool attn_mask shape: {tuple(am_allowed.shape)}")
                elif am_allowed.dim() != 4:
                    raise ValueError(f"Unsupported bool attn_mask shape: {tuple(am_allowed.shape)}")
                allowed_mask = am_allowed if allowed_mask is None else (allowed_mask & am_allowed)
            else:
                # Float attn_mask keeps additive semantics.
                am_add = am.to(dtype=dtype)
                if am_add.dim() == 2:
                    am_add = am_add.view(1, 1, tgt_len, src_len)
                elif am_add.dim() == 3:
                    if am_add.size(0) == bsz * self.num_heads:
                        am_add = am_add.view(bsz, self.num_heads, tgt_len, src_len)
                    elif am_add.size(0) == bsz:
                        am_add = am_add.view(bsz, 1, tgt_len, src_len)
                    elif am_add.size(0) == 1:
                        am_add = am_add.view(1, 1, tgt_len, src_len)
                    else:
                        raise ValueError(f"Unsupported float attn_mask shape: {tuple(am_add.shape)}")
                elif am_add.dim() != 4:
                    raise ValueError(f"Unsupported float attn_mask shape: {tuple(am_add.shape)}")
                additive_mask = am_add if additive_mask is None else (additive_mask + am_add)

        if allowed_mask is not None and additive_mask is not None:
            return additive_mask.masked_fill(~allowed_mask, float("-inf"))
        if allowed_mask is not None:
            return allowed_mask
        return additive_mask

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = True,
        attn_mask: Optional[torch.Tensor] = None,
        average_attn_weights: bool = True,
        is_causal: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        del average_attn_weights
        if query.dim() != 3 or key.dim() != 3 or value.dim() != 3:
            raise ValueError("HeadGatedMultiheadAttention expects batched 3D inputs.")
        if self._qkv_same_embed_dim is False:
            raise ValueError("HeadGatedMultiheadAttention currently expects q/k/v same embed dim.")
        if self.bias_k is not None or self.bias_v is not None or self.add_zero_attn:
            raise ValueError("HeadGatedMultiheadAttention does not support add_bias_kv/add_zero_attn.")

        if self.batch_first:
            q_in = query
            k_in = key
            v_in = value
        else:
            q_in = query.transpose(0, 1)
            k_in = key.transpose(0, 1)
            v_in = value.transpose(0, 1)

        bsz, tgt_len, embed_dim = q_in.shape
        src_len = k_in.size(1)
        if embed_dim != self.embed_dim:
            raise ValueError(f"Expected embed_dim={self.embed_dim}, got {embed_dim}")

        if self.in_proj_bias is None:
            b_q = b_k = b_v = None
        else:
            b_q, b_k, b_v = self.in_proj_bias.chunk(3, dim=0)
        w_q, w_k, w_v = self.in_proj_weight.chunk(3, dim=0)

        q = F.linear(q_in, w_q, b_q)
        k = F.linear(k_in, w_k, b_k)
        v = F.linear(v_in, w_v, b_v)

        q = q.view(bsz, tgt_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3).contiguous()
        k = k.view(bsz, src_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3).contiguous()
        v = v.view(bsz, src_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3).contiguous()

        sdpa_mask = self._expand_to_sdpa_mask(
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            bsz=bsz,
            tgt_len=tgt_len,
            src_len=src_len,
            device=q_in.device,
            dtype=q.dtype,
        )
        dropout_p = float(self.dropout) if self.training else 0.0
        attn_per_head = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=sdpa_mask,
            dropout_p=dropout_p,
            is_causal=bool(is_causal),
        )

        if self.use_gated_attn:
            gate_logits = self.gate_proj(q_in)
            gate = torch.sigmoid(gate_logits).permute(0, 2, 1).unsqueeze(-1)
            gated_attn = attn_per_head * gate.to(dtype=attn_per_head.dtype)
            alpha_attn_gate = float(getattr(self, "alpha_attn_gate", 1.0))
            if alpha_attn_gate <= 0.0:
                out_per_head = attn_per_head
            elif alpha_attn_gate >= 1.0:
                out_per_head = gated_attn
            else:
                out_per_head = (1.0 - alpha_attn_gate) * attn_per_head + alpha_attn_gate * gated_attn
            with torch.no_grad():
                gate_f = gate.detach().float()
                self.last_gated_attn_mean = gate_f.mean()
                self.last_gated_attn_sparsity = (gate_f < 0.1).float().mean()
        else:
            out_per_head = attn_per_head
            self.last_gated_attn_mean = attn_per_head.new_zeros(())
            self.last_gated_attn_sparsity = attn_per_head.new_zeros(())

        out = out_per_head.permute(0, 2, 1, 3).contiguous().view(bsz, tgt_len, self.embed_dim)
        out = self.out_proj(out)
        if not self.batch_first:
            out = out.transpose(0, 1)
        if need_weights:
            return out, None
        return out, None


class GatedTransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float = 0.1,
        activation: str = "relu",
        layer_norm_eps: float = 1e-5,
        batch_first: bool = True,
        norm_first: bool = False,
        use_gated_attn: bool = True,
        gated_attn_init_bias: float = 2.0,
    ):
        super().__init__()
        if activation != "relu":
            raise ValueError("GatedTransformerEncoderLayer currently supports activation='relu' only")

        self.self_attn = HeadGatedMultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=batch_first,
            use_gated_attn=use_gated_attn,
            gated_attn_init_bias=gated_attn_init_bias,
            alpha_attn_gate=1.0,
        )
        self.ffn = FeedForwardBlock(embed_dim=d_model, ff_dim=dim_feedforward)
        self.norm_first = bool(norm_first)
        self.norm1 = POMOInstanceNorm(d_model, eps=layer_norm_eps, batch_first=batch_first)
        self.norm2 = POMOInstanceNorm(d_model, eps=layer_norm_eps, batch_first=batch_first)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.use_gated_attn = bool(use_gated_attn)
        self.alpha_attn_gate = 1.0

    def _sa_block(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor],
        key_padding_mask: Optional[torch.Tensor],
        is_causal: bool = False,
    ) -> torch.Tensor:
        setattr(self.self_attn, "use_gated_attn", bool(getattr(self, "use_gated_attn", True)))
        setattr(self.self_attn, "alpha_attn_gate", float(getattr(self, "alpha_attn_gate", 1.0)))
        try:
            x = self.self_attn(
                x,
                x,
                x,
                attn_mask=attn_mask,
                key_padding_mask=key_padding_mask,
                need_weights=False,
                is_causal=is_causal,
            )[0]
        except TypeError:
            x = self.self_attn(
                x,
                x,
                x,
                attn_mask=attn_mask,
                key_padding_mask=key_padding_mask,
                need_weights=False,
            )[0]
        return self.dropout1(x)

    def forward(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        if self.norm_first:
            src = src + self._sa_block(
                self.norm1(src, src_key_padding_mask),
                src_mask,
                src_key_padding_mask,
                is_causal=is_causal,
            )
            src = src + self.dropout2(self.ffn(self.norm2(src, src_key_padding_mask)))
        else:
            src = self.norm1(
                src + self._sa_block(src, src_mask, src_key_padding_mask, is_causal=is_causal),
                src_key_padding_mask,
            )
            src = self.norm2(src + self.dropout2(self.ffn(src)), src_key_padding_mask)
        stats = {
            "gated_attn_mean": (
                self.self_attn.last_gated_attn_mean
                if (self.use_gated_attn and self.self_attn.last_gated_attn_mean is not None)
                else src.new_zeros(())
            ),
            "gated_attn_sparsity": (
                self.self_attn.last_gated_attn_sparsity
                if (self.use_gated_attn and self.self_attn.last_gated_attn_sparsity is not None)
                else src.new_zeros(())
            ),
        }
        return src, src.new_zeros(()), stats


class DepthAttnResidual(nn.Module):
    """
    Depth-wise residual routing over historical layer states.
    History states are normalized with POMO instance stats, scored by a learned query,
    then mixed with softmax over depth.
    """

    def __init__(
        self,
        d_model: int,
        layer_norm_eps: float = 1e-5,
        batch_first: bool = True,
    ):
        super().__init__()
        self.batch_first = bool(batch_first)
        self.norm = POMOInstanceNorm(d_model=d_model, eps=layer_norm_eps, batch_first=batch_first)
        self.depth_query = nn.Parameter(torch.zeros(int(d_model)))
        self.depth_scale = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.last_depth_weight_mean: Optional[torch.Tensor] = None

    def _to_bne(self, x: torch.Tensor) -> torch.Tensor:
        return x if self.batch_first else x.transpose(0, 1)

    def _from_bne(self, x: torch.Tensor) -> torch.Tensor:
        return x if self.batch_first else x.transpose(0, 1)

    def forward(
        self,
        states: List[torch.Tensor],
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if len(states) <= 0:
            raise ValueError("DepthAttnResidual requires at least one history state.")

        ref = states[-1]
        ref_bne = self._to_bne(ref)
        if ref_bne.dim() != 3:
            raise ValueError(f"DepthAttnResidual expects 3D states, got shape={tuple(ref.shape)}")
        bsz, n_token, d_model = ref_bne.shape

        pm_bool: Optional[torch.Tensor] = None
        if padding_mask is not None:
            if padding_mask.dim() != 2:
                raise ValueError(f"padding_mask must be [B,N], got shape={tuple(padding_mask.shape)}")
            if padding_mask.size(0) != bsz or padding_mask.size(1) != n_token:
                raise ValueError(
                    f"padding_mask shape mismatch: mask={tuple(padding_mask.shape)}, expected={(bsz, n_token)}"
                )
            pm_bool = padding_mask.to(device=ref_bne.device, dtype=torch.bool)

        norm_states_bne: List[torch.Tensor] = []
        value_states_bne: List[torch.Tensor] = []
        for idx, state in enumerate(states):
            if state.shape != ref.shape:
                raise ValueError(
                    f"All history states must share shape. states[{idx}]={tuple(state.shape)} vs ref={tuple(ref.shape)}"
                )
            norm_state = self.norm(state, padding_mask=pm_bool)
            norm_state_bne = self._to_bne(norm_state)
            value_state_bne = self._to_bne(state)
            norm_states_bne.append(norm_state_bne)
            value_states_bne.append(value_state_bne)

        q = self.depth_query.to(device=ref_bne.device, dtype=ref_bne.dtype).view(1, 1, d_model)
        scale = self.depth_scale.to(device=ref_bne.device, dtype=ref_bne.dtype)
        logits = torch.stack([(s * q).sum(dim=-1) * scale for s in norm_states_bne], dim=1)  # [B,D,N]
        weights = F.softmax(logits, dim=1)

        if pm_bool is not None:
            valid = (~pm_bool).to(dtype=weights.dtype, device=weights.device)
            weights = weights * valid.unsqueeze(1)
        values = torch.stack(value_states_bne, dim=1)  # [B,D,N,E]
        out_bne = (weights.unsqueeze(-1) * values).sum(dim=1)

        if pm_bool is not None:
            valid = (~pm_bool).to(dtype=out_bne.dtype, device=out_bne.device).unsqueeze(-1)
            out_bne = out_bne * valid

        with torch.no_grad():
            self.last_depth_weight_mean = weights.detach().float().mean()
        return self._from_bne(out_bne)


class EncoderStack(nn.Module):
    def __init__(self, encoder_layer: GatedTransformerEncoderLayer, num_layers: int):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(max(1, int(num_layers)))])
        self.num_layers = len(self.layers)
        d_model = int(encoder_layer.self_attn.embed_dim)
        batch_first = bool(getattr(encoder_layer.self_attn, "batch_first", True))
        layer_norm_eps = float(getattr(encoder_layer.norm1, "eps", 1e-5))
        self.depth_residuals = nn.ModuleList(
            [
                DepthAttnResidual(
                    d_model=d_model,
                    layer_norm_eps=layer_norm_eps,
                    batch_first=batch_first,
                )
                for _ in range(max(0, self.num_layers - 1))
            ]
        )
        self.last_gated_attn_mean: Optional[torch.Tensor] = None
        self.last_gated_attn_sparsity: Optional[torch.Tensor] = None
        self.last_depth_attn_mean: Optional[torch.Tensor] = None

    def forward(
        self,
        src: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        out = src
        valid_mask: Optional[torch.Tensor] = None
        if src_key_padding_mask is not None:
            valid_mask = (~src_key_padding_mask.to(device=out.device, dtype=torch.bool)).to(dtype=out.dtype).unsqueeze(-1)
            out = out * valid_mask
        history: List[torch.Tensor] = [out]
        total_aux = src.new_zeros(())
        gated_mean_list: List[torch.Tensor] = []
        gated_sparsity_list: List[torch.Tensor] = []
        depth_mean_list: List[torch.Tensor] = []
        for idx, layer in enumerate(self.layers):
            # The first encoder layer sees the stem directly; depth routing starts from layer 2
            # to avoid a one-state AttnRes module with dead parameters.
            if idx == 0 or len(self.depth_residuals) == 0:
                x_in = out
            else:
                x_in = self.depth_residuals[idx - 1](history, padding_mask=src_key_padding_mask)
                depth_mean = getattr(self.depth_residuals[idx - 1], "last_depth_weight_mean", None)
                if depth_mean is not None:
                    depth_mean_list.append(depth_mean)
            out, aux_loss, layer_stats = layer(
                x_in,
                src_mask=mask,
                src_key_padding_mask=src_key_padding_mask,
                is_causal=is_causal,
            )
            if valid_mask is not None:
                out = out * valid_mask
            total_aux = total_aux + aux_loss
            gated_mean_list.append(layer_stats["gated_attn_mean"])
            gated_sparsity_list.append(layer_stats["gated_attn_sparsity"])
            history.append(out)
        self.last_gated_attn_mean = torch.stack(gated_mean_list, dim=0) if gated_mean_list else None
        self.last_gated_attn_sparsity = torch.stack(gated_sparsity_list, dim=0) if gated_sparsity_list else None
        self.last_depth_attn_mean = torch.stack(depth_mean_list, dim=0) if depth_mean_list else None
        stats = {
            "gated_attn_mean": (
                self.last_gated_attn_mean if self.last_gated_attn_mean is not None else out.new_zeros(())
            ),
            "gated_attn_sparsity": (
                self.last_gated_attn_sparsity if self.last_gated_attn_sparsity is not None else out.new_zeros(())
            ),
            "depth_attn_mean": (
                self.last_depth_attn_mean if self.last_depth_attn_mean is not None else out.new_zeros(())
            ),
        }
        return out, total_aux / max(1, self.num_layers), stats


def normalize_checkpoint_state_dict(state_obj):
    if not isinstance(state_obj, dict):
        return state_obj

    normalized = {}
    legacy_ffn_map = {
        ".moe_ffn.experts.0.0.": ".ffn.lin1.",
        ".moe_ffn.experts.0.2.": ".ffn.lin2.",
    }

    for key, value in state_obj.items():
        if key == "value_baseline_mix_logit" or key.startswith("value_head."):
            continue
        mapped_key = key
        for old_key, new_key in legacy_ffn_map.items():
            if old_key in key:
                mapped_key = key.replace(old_key, new_key)
                break
        if ".moe_ffn." in mapped_key:
            continue
        normalized[mapped_key] = value

    return normalized




def _use_angle_node_features() -> bool:
    # Angle features are fixed on in the current baseline.
    return True


def get_planned_node_feature_names() -> List[str]:
    names = [
        "x_norm",
        "y_norm",
        "demand_norm",
        "ready_norm",
        "due_norm",
        "service_norm",
        "tw_width_norm",
        "dist_to_depot_norm",
    ]
    if _use_angle_node_features():
        names.extend(["angle_sin", "angle_cos"])
    return names


def get_planned_node_feature_dim() -> int:
    return len(get_planned_node_feature_names())


def get_dyn_feature_dim() -> int:
    return len(get_dyn_feature_names())


def get_dyn_feature_names() -> List[str]:
    return [
        "curr_x_norm",
        "curr_y_norm",
        "curr_time_ratio",
        "remaining_cap_ratio",
    ]


def get_cand_phi_feature_names() -> List[str]:
    return [
        "travel_dist_norm",
        "wait_norm",
        "tw_slack_ratio",
        "arrival_time_norm",
        "departure_time_norm",
        "depot_angle_diff_norm",
        "knn_nearest_dist_norm",
        "knn_mean_dist_norm",
        "knn_min_dist_norm",
    ]


def get_cand_phi_feature_dim() -> int:
    return len(get_cand_phi_feature_names())




def _resolve_model_dyn_dim() -> int:
    cfg_dyn = getattr(train_defaults, "dyn_dim", None)
    if cfg_dyn is None:
        return int(get_dyn_feature_dim())
    cfg_val = int(cfg_dyn)
    if cfg_val <= 0:
        return int(get_dyn_feature_dim())
    return cfg_val


def _resolve_model_arch_defaults() -> Tuple[int, int, int, Optional[int]]:
    embed_dim = int(getattr(train_defaults, "model_embed_dim", 128))
    n_heads = int(getattr(train_defaults, "model_n_heads", 4))
    n_layers = int(getattr(train_defaults, "model_n_layers", 2))
    ff_dim = int(getattr(train_defaults, "model_ff_dim", 0))
    ff_dim = None if ff_dim <= 0 else ff_dim
    return embed_dim, n_heads, n_layers, ff_dim


def _depot_due_sentinel_threshold() -> float:
    return float(getattr(train_defaults, "depot_due_sentinel_threshold", 1e5))


def _resolve_raw_feat_dim(node_dim: int) -> int:
    cfg_raw = getattr(train_defaults, "raw_feat_dim", None)
    if cfg_raw is None:
        return int(node_dim)
    raw_feat_dim = int(cfg_raw)
    if raw_feat_dim <= 0:
        return int(node_dim)
    if raw_feat_dim > int(node_dim):
        raise ValueError(
            f"raw_feat_dim={raw_feat_dim} cannot exceed node_dim={int(node_dim)} "
            "(raw bias uses node feature prefix)."
        )
    return raw_feat_dim


def _resolve_raw_feature_names(raw_feat_dim: int) -> List[str]:
    base = get_planned_node_feature_names()
    if raw_feat_dim <= len(base):
        return base[:raw_feat_dim]
    tail = [f"raw_f{i}" for i in range(len(base), raw_feat_dim)]
    return base + tail


def _compute_coord_norm_params(
    coords: torch.Tensor,
    valid_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute per-instance normalization parameters for x/y.
    Returns (min_x, min_y, max_xy) with shape [B].
    """
    if valid_mask is not None:
        if valid_mask.dim() != 2 or valid_mask.shape[:1] != coords.shape[:1]:
            raise ValueError(f"valid_mask must be [B,N], got {tuple(valid_mask.shape)}")
        valid_mask = valid_mask.to(device=coords.device, dtype=torch.bool)
        inf = torch.tensor(float("inf"), device=coords.device, dtype=coords.dtype)
        ninf = torch.tensor(float("-inf"), device=coords.device, dtype=coords.dtype)
        x = coords[..., 0]
        y = coords[..., 1]
        min_x = x.masked_fill(~valid_mask, inf).min(dim=1).values
        max_x = x.masked_fill(~valid_mask, ninf).max(dim=1).values
        min_y = y.masked_fill(~valid_mask, inf).min(dim=1).values
        max_y = y.masked_fill(~valid_mask, ninf).max(dim=1).values
    else:
        min_x = coords[..., 0].min(dim=1).values
        max_x = coords[..., 0].max(dim=1).values
        min_y = coords[..., 1].min(dim=1).values
        max_y = coords[..., 1].max(dim=1).values
    span_x = (max_x - min_x).clamp_min(1.0)
    span_y = (max_y - min_y).clamp_min(1.0)
    max_xy = torch.maximum(span_x, span_y)
    return min_x, min_y, max_xy


def _compute_knn_stats(
    dist_cust: torch.Tensor,
    valid_mask: torch.Tensor,
    k: int = 5,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute per-customer KNN distance stats.
    dist_cust: [B, N, N] customer-customer distances
    valid_mask: [B, N] valid customer mask
    Returns: (nearest, mean, min) each [B, N]
    """
    if dist_cust.dim() != 3:
        raise ValueError(f"dist_cust must be [B,N,N], got {tuple(dist_cust.shape)}")
    if valid_mask.dim() != 2:
        raise ValueError(f"valid_mask must be [B,N], got {tuple(valid_mask.shape)}")
    B, N, _ = dist_cust.shape
    if N <= 1:
        zeros = dist_cust.new_zeros((B, N))
        return zeros, zeros, zeros

    k_eff = max(1, min(int(k), N - 1))
    valid_mask = valid_mask.to(dtype=torch.bool, device=dist_cust.device)
    pair_mask = valid_mask.unsqueeze(1) & valid_mask.unsqueeze(2)
    dist_masked = dist_cust.masked_fill(~pair_mask, float("inf"))
    eye = torch.eye(N, device=dist_cust.device, dtype=torch.bool).unsqueeze(0)
    dist_masked = dist_masked.masked_fill(eye, float("inf"))
    knn_vals, _ = torch.topk(dist_masked, k_eff, dim=-1, largest=False)
    knn_vals = torch.nan_to_num(knn_vals, nan=0.0, posinf=0.0, neginf=0.0)
    knn_nearest = knn_vals[..., 0]
    knn_mean = knn_vals.mean(dim=-1)
    knn_min = knn_vals.min(dim=-1).values
    return knn_nearest, knn_mean, knn_min


def _match_model_node_feat_dim(
    node_feats: torch.Tensor,
    depot_feat: torch.Tensor,
    target_node_dim: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
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


def _match_dyn_feat_dim(dyn_feats: torch.Tensor, target_dyn_dim: int) -> torch.Tensor:
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


def _match_cand_phi_dim(cand_phi: torch.Tensor, target_cand_dim: int) -> torch.Tensor:
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


def pad_instances(instances: List[Instance]):
    max_n = max(len(inst.customers) for inst in instances)
    B = len(instances)
    use_angle = _use_angle_node_features()
    feat_dim = get_planned_node_feature_dim()

    batch_nf: List[List[List[float]]] = []
    batch_df: List[List[float]] = []
    batch_pad: List[List[float]] = []

    coords = torch.zeros(B, max_n + 1, 2)
    demands = torch.zeros(B, max_n + 1)
    readys = torch.zeros(B, max_n + 1)
    dues = torch.zeros(B, max_n + 1)
    services = torch.zeros(B, max_n + 1)
    capacities = torch.zeros(B)

    for b, inst in enumerate(instances):
        n = len(inst.customers)
        depot = inst.depot
        customers = inst.customers

        xs = [c.x for c in customers] + [depot.x]
        ys = [c.y for c in customers] + [depot.y]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        span_x = max(max_x - min_x, 1.0)
        span_y = max(max_y - min_y, 1.0)
        max_xy = max(span_x, span_y, 1.0)

        C_cap = float(max(inst.capacity, 1.0))

        all_ready = [c.ready_time for c in customers] + [depot.ready_time]
        all_due = [c.due_time for c in customers] + [depot.due_time]
        min_t = min(all_ready)
        max_t = max(all_due)
        T = float(max(max_t - min_t, 1.0))

        max_dist = 0.0
        for c in customers:
            max_dist = max(
                max_dist,
                math.hypot(float(c.x) - float(depot.x), float(c.y) - float(depot.y)),
            )
        max_dist = float(max(max_dist, 1.0))

        def _build_node_feat(
            x: float,
            y: float,
            demand: float,
            ready_time: float,
            due_time: float,
            service_time: float,
        ) -> List[float]:
            dx = float(x) - float(depot.x)
            dy = float(y) - float(depot.y)
            dist_to_depot = math.hypot(dx, dy)
            tw_width_norm = max(0.0, float(due_time) - float(ready_time)) / T
            out = [
                (float(x) - min_x) / max_xy,
                (float(y) - min_y) / max_xy,
                float(demand) / C_cap,
                (float(ready_time) - min_t) / T,
                (float(due_time) - min_t) / T,
                float(service_time) / T,
                tw_width_norm,
                dist_to_depot / max_dist,
            ]
            if use_angle:
                if dist_to_depot > 1e-8:
                    out.append(dy / dist_to_depot)
                    out.append(dx / dist_to_depot)
                else:
                    out.append(0.0)
                    out.append(0.0)
            return out

        feats = []
        for c in customers:
            feats.append(_build_node_feat(c.x, c.y, c.demand, c.ready_time, c.due_time, c.service_time))
        for _ in range(max_n - n):
            feats.append([0.0] * feat_dim)

        depot_f = _build_node_feat(
            depot.x,
            depot.y,
            0.0,
            depot.ready_time,
            depot.due_time,
            depot.service_time,
        )

        batch_nf.append(feats)
        batch_df.append(depot_f)
        batch_pad.append([1.0] * n + [0.0] * (max_n - n))

        coords[b, 0, 0] = depot.x
        coords[b, 0, 1] = depot.y
        demands[b, 0] = 0
        readys[b, 0] = depot.ready_time
        dues[b, 0] = depot.due_time
        services[b, 0] = 0.0

        for i, c in enumerate(customers):
            coords[b, i + 1, 0] = c.x
            coords[b, i + 1, 1] = c.y
            demands[b, i + 1] = c.demand
            readys[b, i + 1] = c.ready_time
            dues[b, i + 1] = c.due_time
            services[b, i + 1] = c.service_time

        capacities[b] = float(inst.capacity)

    return (
        torch.tensor(batch_nf, dtype=torch.float32),
        torch.tensor(batch_df, dtype=torch.float32),
        torch.tensor(batch_pad, dtype=torch.float32),
        {
            "coords": coords,
            "demand": demands,
            "ready": readys,
            "due": dues,
            "service": services,
            "capacity": capacities,
        },
    )


def _resolve_vehicle_limits_from_config(
    env_data: Dict[str, torch.Tensor],
    pad_mask: torch.Tensor,
) -> Tuple[torch.Tensor, int, str]:
    """
    Resolve per-instance vehicle budget and batch-level M_max from config.
    Minimal config:
    - train_defaults.vehicle_max > 0: fixed vehicle limit.
    - otherwise: fallback to per-instance real customer count.

    Returns:
    - vehicle_limit: [B] int64, per-instance max usable vehicles
    - vehicle_max: int, batch M_max used for tensor shapes [B, M_max, ...]
    - source: "fixed" or "n_customers(auto)"
    """
    if pad_mask.dim() != 2:
        raise ValueError(f"pad_mask must be [B,N], got {tuple(pad_mask.shape)}")
    if "demand" not in env_data or "capacity" not in env_data:
        raise KeyError("env_data must contain keys: demand, capacity")
    demand = env_data["demand"]
    capacity = env_data["capacity"]
    if demand.dim() != 2 or capacity.dim() != 1:
        raise ValueError(
            f"env_data shape mismatch: demand={tuple(demand.shape)}, capacity={tuple(capacity.shape)}"
        )

    B = pad_mask.size(0)
    if demand.size(0) != B or capacity.size(0) != B:
        raise ValueError(
            f"batch mismatch: pad_mask={B}, demand={demand.size(0)}, capacity={capacity.size(0)}"
        )

    real_counts = (pad_mask > 0.5).sum(dim=1).to(dtype=torch.long).clamp_min(1)  # [B]
    fixed_v = getattr(train_defaults, "vehicle_max", None)
    if fixed_v is None or int(fixed_v) <= 0:
        vehicle_limit = real_counts.clone()
        source = "n_customers(auto)"
    else:
        vehicle_limit = torch.full_like(real_counts, int(fixed_v))
        source = "fixed"

    vehicle_limit = vehicle_limit.clamp_min(1)
    vehicle_max = int(vehicle_limit.max().item()) if vehicle_limit.numel() > 0 else 1
    vehicle_max = max(1, vehicle_max)
    return vehicle_limit, vehicle_max, source




class BatchVRPTWEnv:
    def __init__(
        self,
        env_data: Dict[str, torch.Tensor],
        device: str = "cuda",
        track_routes: bool = True,
        vehicle_limit: Optional[torch.Tensor] = None,
        vehicle_max: Optional[int] = None,
        vehicle_limit_source: str = "fixed",
    ):

        self.device = device
        self.track_routes = bool(track_routes)
        self.coords = env_data["coords"].to(device)  # [B, N+1, 2]
        self.demand = env_data["demand"].to(device)  # [B, N+1]
        self.ready = env_data["ready"].to(device)  # [B, N+1]
        self.due = env_data["due"].to(device)  # [B, N+1]
        self.service = env_data["service"].to(device)  # [B, N+1]
        self.capacity = env_data["capacity"].to(device)  # [B]

        self.B = self.coords.size(0)
        self.N_plus_1 = self.coords.size(1)
        self.N = self.N_plus_1 - 1
        self.vehicle_limit_source = str(vehicle_limit_source)

        if vehicle_limit is not None:
            veh_limit_tensor = vehicle_limit.to(device=device, dtype=torch.long).reshape(-1)
            if veh_limit_tensor.numel() != self.B:
                raise ValueError(f"vehicle_limit expects B={self.B}, got {veh_limit_tensor.numel()}")
            self.vehicle_limit = veh_limit_tensor.clamp_min(1)
        else:
            vm = int(vehicle_max) if vehicle_max is not None else int(self.N)
            self.vehicle_limit = torch.full((self.B,), max(1, vm), dtype=torch.long, device=device)
        if vehicle_max is None:
            self.vehicle_max = int(self.vehicle_limit.max().item())
        else:
            self.vehicle_max = int(max(1, int(vehicle_max)))
        if self.vehicle_max < int(self.vehicle_limit.max().item()):
            raise ValueError(
                f"vehicle_max={self.vehicle_max} < max(vehicle_limit)={int(self.vehicle_limit.max().item())}"
            )
        self.vehicle_limit = self.vehicle_limit.clamp(max=self.vehicle_max)

        
        self.dist_matrix = torch.cdist(self.coords, self.coords)  # [B, N+1, N+1]
        self.dist_to_depot = self.dist_matrix[:, 1:, 0]  # [B, N]
        self.max_dist = float(self.dist_matrix.max().item() + 1e-6)
        self.batch_idx = torch.arange(self.B, device=device)
        self.customer_coords = self.coords[:, 1:, :]
        self.customer_demand = self.demand[:, 1:]
        self.customer_ready = self.ready[:, 1:]
        self.customer_due = self.due[:, 1:]
        self.customer_service = self.service[:, 1:]
        self.customer_dist_matrix = self.dist_matrix[:, 1:, 1:]
        self.depot_due = self.due[:, 0]
        self.depot_xy = self.coords[:, 0, :]
        depot_vec = self.coords - self.depot_xy.unsqueeze(1)
        self.node_angle_from_depot = torch.atan2(depot_vec[..., 1], depot_vec[..., 0])
        self.cand_angle_from_depot = self.node_angle_from_depot[:, 1:]
        self.max_dist_row = self.dist_matrix.amax(dim=-1).amax(dim=-1).unsqueeze(1)
        sentinel = _depot_due_sentinel_threshold()
        self.has_depot_due = ((self.depot_due > 0.0) & (self.depot_due < sentinel)).unsqueeze(1)
        self.dist_scale = max(float(self.max_dist), 1.0)
        self._cand_cache_key: tuple[int, tuple[int, ...], torch.device] | None = None
        self._cand_valid_cust: Optional[torch.Tensor] = None
        self._cand_max_customer_due: Optional[torch.Tensor] = None
        self._cand_knn_nearest_dist_norm: Optional[torch.Tensor] = None
        self._cand_knn_mean_dist_norm: Optional[torch.Tensor] = None
        self._cand_knn_min_dist_norm: Optional[torch.Tensor] = None

        
        self.visited = torch.zeros(self.B, self.N, dtype=torch.bool, device=device)
        self.finished = torch.zeros(self.B, dtype=torch.bool, device=device)

        self.loc = torch.zeros(self.B, dtype=torch.long, device=device)
        self.load = torch.zeros(self.B, dtype=torch.float32, device=device)
        self.curr_dist = torch.zeros(self.B, dtype=torch.float32, device=device)  
        self.total_dist = torch.zeros(self.B, dtype=torch.float32, device=device)  
        self.route_len = torch.zeros(self.B, dtype=torch.long, device=device)  
        self.veh_count = torch.zeros(self.B, dtype=torch.long, device=device)  
        self.infeasible = torch.zeros(self.B, dtype=torch.bool, device=device)  

        self.time = self.ready[:, 0].clone()

        if self.track_routes:
            self.routes: List[List[List[int]]] = [[] for _ in range(self.B)]
            self.current_route: List[List[int]] = [[] for _ in range(self.B)]
        else:
            self.routes = []
            self.current_route = []

    def _get_candidate_cache_key(self, pad_mask: torch.Tensor) -> tuple[int, tuple[int, ...], torch.device]:
        return (int(pad_mask.data_ptr()), tuple(int(s) for s in pad_mask.shape), pad_mask.device)

    def _ensure_candidate_static_cache(self, pad_mask: torch.Tensor) -> torch.Tensor:
        cache_key = self._get_candidate_cache_key(pad_mask)
        if self._cand_cache_key == cache_key and self._cand_valid_cust is not None:
            return self._cand_valid_cust

        valid_cust = (pad_mask > 0.5).to(device=self.device, dtype=torch.bool)
        due_for_max = torch.where(valid_cust, self.customer_due, torch.zeros_like(self.customer_due))
        max_customer_due = due_for_max.max(dim=1, keepdim=True).values
        knn_nearest, knn_mean, knn_min = _compute_knn_stats(self.customer_dist_matrix, valid_cust)

        self._cand_cache_key = cache_key
        self._cand_valid_cust = valid_cust
        self._cand_max_customer_due = max_customer_due
        self._cand_knn_nearest_dist_norm = knn_nearest / self.dist_scale
        self._cand_knn_mean_dist_norm = knn_mean / self.dist_scale
        self._cand_knn_min_dist_norm = knn_min / self.dist_scale
        return valid_cust

    def reset_vehicle(self, b_indices: torch.Tensor):
        if len(b_indices) == 0:
            return
        self.loc[b_indices] = 0
        self.load[b_indices] = 0.0
        self.curr_dist[b_indices] = 0.0
        self.route_len[b_indices] = 0
        self.time[b_indices] = self.ready[b_indices, 0]

    def step(self, actions: torch.Tensor):
        is_depot = actions == self.N
        is_cust = ~is_depot

        if is_cust.any():
            cust_indices = actions[is_cust]
            env_indices = cust_indices + 1
            batch_indices = torch.nonzero(is_cust).squeeze(1)

            prev_locs = self.loc[batch_indices]
            curr_locs = env_indices

            dist = self.dist_matrix[batch_indices, prev_locs, curr_locs]

            self.load[batch_indices] += self.demand[batch_indices, curr_locs]
            self.curr_dist[batch_indices] += dist
            self.total_dist[batch_indices] += dist
            self.route_len[batch_indices] += 1

            arrival = self.time[batch_indices] + dist
            start_service = torch.max(arrival, self.ready[batch_indices, curr_locs])
            self.time[batch_indices] = start_service + self.service[batch_indices, curr_locs]

            self.loc[batch_indices] = curr_locs
            self.visited[batch_indices, cust_indices] = True

            if self.track_routes:
                for idx, act in zip(batch_indices.cpu().numpy(), cust_indices.cpu().numpy()):
                    self.current_route[idx].append(int(act))

        if is_depot.any():
            batch_indices = torch.nonzero(is_depot).squeeze(1)

            prev_locs = self.loc[batch_indices]
            dist_back = self.dist_matrix[batch_indices, prev_locs, 0]
            arrive_time = self.time[batch_indices] + dist_back
            self.curr_dist[batch_indices] += dist_back
            self.total_dist[batch_indices] += dist_back

            completed_route = self.route_len[batch_indices] > 0
            if completed_route.any():
                # vehicle_count definition: vehicles that served at least one customer.
                self.veh_count[batch_indices[completed_route]] += 1

            if self.track_routes:
                for idx in batch_indices.cpu().numpy():
                    r = self.current_route[idx]
                    if len(r) > 0:
                        self.routes[idx].append(r)
                    self.current_route[idx] = []

            
            depot_due = self.due[batch_indices, 0]
            
            violated = (depot_due > 0) & (arrive_time > depot_due + 1e-9)
            ok_mask = ~violated

            if ok_mask.any():
                ok_indices = batch_indices[ok_mask]
                self.reset_vehicle(ok_indices)

            if violated.any():
                bad_indices = batch_indices[violated]
                self.load[bad_indices] = 0.0
                self.loc[bad_indices] = 0
                self.curr_dist[bad_indices] = 0.0
                self.route_len[bad_indices] = 0
                self.infeasible[bad_indices] = True
                
                self.finished[bad_indices] = True


    def get_used_vehicle_count(self, include_open_route: bool = True) -> torch.Tensor:
        """
        vehicle_count := number of vehicles that have served >=1 customer.
        In legacy mode, completed routes are tracked by self.veh_count; optionally include
        the currently open route when it already contains customers.
        """
        used = self.veh_count.clone()
        if include_open_route:
            used = used + (self.route_len > 0).to(dtype=torch.long)
        return used

    def compute_terminal_mask(self, pad_mask: torch.Tensor) -> torch.Tensor:
        """
        Rollout termination mask per instance [B].
        Single-route mode: all real customers served AND the active route is back at depot.
        """
        real_counts = (pad_mask > 0.5).sum(dim=1)
        visited_counts = self.visited.sum(dim=1)
        all_served = visited_counts >= real_counts
        return all_served & (self.loc == 0)

    def get_mask(self, pad_mask: torch.Tensor):
        
        active_mask = ~self.finished
        cust_mask = (~self.visited) & (pad_mask > 0.5) & active_mask.unsqueeze(1)
        
        cap_mask = (self.load.unsqueeze(1) + self.customer_demand) <= self.capacity.unsqueeze(1)
        
        dists = self.dist_matrix[self.batch_idx, self.loc][:, 1:]
        arrival_time = self.time.unsqueeze(1) + dists
        start_service = torch.maximum(arrival_time, self.customer_ready)
        tw_mask = start_service <= self.customer_due + 1e-9
        
        finish_service = start_service + self.customer_service
        return_time = finish_service + self.dist_to_depot
        depot_due = self.depot_due.unsqueeze(1)
        depot_ok = (depot_due <= 0) | (return_time <= depot_due + 1e-9)
        
        valid_cust = cust_mask & cap_mask & tw_mask & depot_ok
        
        at_depot = self.loc == 0
        dist_now = self.dist_matrix[self.batch_idx, self.loc, 0]
        arrive_depot_now = self.time + dist_now
        depot_now_ok = (self.depot_due <= 0) | (arrive_depot_now <= self.depot_due + 1e-9)
        allow_depot = (~at_depot) & depot_now_ok & active_mask
        full_mask = torch.cat([valid_cust, allow_depot.unsqueeze(1)], dim=1)
        return full_mask


    def get_candidate_features(self, pad_mask: torch.Tensor) -> torch.Tensor:
        """
        Build per-customer dynamic candidate features for key augmentation.
        Output shape: [B, N, F], customers only (no depot-action slot).
        """
        eps = 1e-6
        valid_cust = self._ensure_candidate_static_cache(pad_mask)

        dist_to_i = self.dist_matrix[self.batch_idx, self.loc][:, 1:]
        arrival = self.time.unsqueeze(1) + dist_to_i

        wait = torch.relu(self.customer_ready - arrival)
        start = torch.maximum(arrival, self.customer_ready)
        tw_slack = self.customer_due - start
        finish = start + self.customer_service

        fallback_horizon = torch.maximum(
            self._cand_max_customer_due,
            self.time.unsqueeze(1) + 2.0 * self.max_dist_row,
        )
        depot_due = self.depot_due.unsqueeze(1)
        horizon = torch.where(self.has_depot_due, depot_due, fallback_horizon).clamp_min(1.0)

        travel_dist_norm = dist_to_i / self.dist_scale
        wait_norm = wait / (horizon + eps)
        tw_slack_ratio = tw_slack / (horizon + eps)
        arrival_time_norm = arrival / (horizon + eps)
        departure_time_norm = finish / (horizon + eps)

        angle_cur = self.node_angle_from_depot[self.batch_idx, self.loc]
        angle_diff = torch.remainder(
            self.cand_angle_from_depot - angle_cur.unsqueeze(1) + math.pi,
            2 * math.pi,
        ) - math.pi
        depot_angle_diff_norm = angle_diff.abs() / math.pi

        cand_phi = torch.stack(
            [
                travel_dist_norm,
                wait_norm,
                tw_slack_ratio,
                arrival_time_norm,
                departure_time_norm,
                depot_angle_diff_norm,
                self._cand_knn_nearest_dist_norm,
                self._cand_knn_mean_dist_norm,
                self._cand_knn_min_dist_norm,
            ],
            dim=-1,
        )
        cand_phi = torch.nan_to_num(cand_phi, nan=0.0, posinf=0.0, neginf=0.0)
        cand_phi = cand_phi * valid_cust.unsqueeze(-1).to(cand_phi.dtype)
        # quick self-check: expected feature size for dynamic key augmentation
        assert cand_phi.size(-1) == get_cand_phi_feature_dim()
        return cand_phi





class AttentionVRPTW(nn.Module):

    def __init__(
        self,
        node_dim: Optional[int] = None,
        dyn_dim: Optional[int] = None,
        embed_dim: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        ff_dim: Optional[int] = None,
        latent_dim: int = 0,
        cand_phi_dim: Optional[int] = None,
        cand_phi_hidden_dim: int = 0,
        use_raw_feature_bias: bool = False,
        latent_injection_mode: str = "film",
        use_gated_attn: bool = True,
        gated_attn_init_bias: float = 2.0,
    ):
        super().__init__()
        if node_dim is None:
            node_dim = get_planned_node_feature_dim()
        if dyn_dim is None:
            dyn_dim = _resolve_model_dyn_dim()
        if cand_phi_dim is None:
            cand_phi_dim = get_cand_phi_feature_dim()
        self.dim = embed_dim
        self.dyn_dim = int(dyn_dim)
        self.latent_dim = max(0, int(latent_dim))
        latent_mode = str(latent_injection_mode).strip().lower()
        if latent_mode not in {"film", "legacy_concat"}:
            raise ValueError(f"Unsupported latent_injection_mode: {latent_injection_mode}")
        if self.latent_dim <= 0:
            latent_mode = "film"
        self.latent_injection_mode = latent_mode
        self.cand_phi_dim = max(0, int(cand_phi_dim))
        self.cand_phi_hidden_dim = max(0, int(cand_phi_hidden_dim))
        self.use_raw_feature_bias = bool(use_raw_feature_bias)
        self.alpha_attn_gate: Optional[float] = None

        self.node_dim = int(node_dim)
        self.node_enc = nn.Linear(self.node_dim, embed_dim)
        self.depot_enc = nn.Linear(self.node_dim, embed_dim)

        ff_dim = ff_dim or embed_dim * 4
        enc_layer = GatedTransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            batch_first=True,
            activation="relu",
            use_gated_attn=use_gated_attn,
            gated_attn_init_bias=gated_attn_init_bias,
        )
        self.encoder = EncoderStack(enc_layer, num_layers=n_layers)

        self.graph_proj = nn.Linear(embed_dim, embed_dim)
        self.dyn_proj = nn.Linear(self.dyn_dim, embed_dim)

        base_ctx_dim = embed_dim * 3
        ctx_dim = base_ctx_dim + (self.latent_dim if self.latent_injection_mode == "legacy_concat" else 0)
        self.ctx_dim = ctx_dim
        self.vehicle_ctx_dim = base_ctx_dim
        self.W_q = nn.Linear(ctx_dim, embed_dim)
        self.W_k = nn.Linear(embed_dim, embed_dim)
        self.scale = math.sqrt(embed_dim)
        self.end_scorer = nn.Linear(ctx_dim, 1)
        self.vehicle_scorer = nn.Linear(self.vehicle_ctx_dim, 1)
        if self.latent_dim > 0 and self.latent_injection_mode != "legacy_concat":
            self.latent_context_proj = nn.Linear(self.latent_dim, ctx_dim, bias=False)
            self.latent_film = nn.Sequential(
                nn.Linear(self.latent_dim, ctx_dim),
                nn.SiLU(),
                nn.Linear(ctx_dim, 2 * ctx_dim),
            )
            nn.init.zeros_(self.latent_film[-1].weight)
            nn.init.zeros_(self.latent_film[-1].bias)
        else:
            self.latent_context_proj = None
            self.latent_film = None
        if self.cand_phi_dim > 0:
            if self.cand_phi_hidden_dim > 0:
                self.phi_proj = nn.Sequential(
                    nn.Linear(self.cand_phi_dim, self.cand_phi_hidden_dim),
                    nn.ReLU(),
                    nn.Linear(self.cand_phi_hidden_dim, embed_dim),
                )
            else:
                self.phi_proj = nn.Linear(self.cand_phi_dim, embed_dim)
        else:
            self.phi_proj = None

        # B-2 for dynamic derived features cand_phi
        if self.cand_phi_dim > 0:
            alpha_hidden = max(32, embed_dim // 2)
            # Factorized first layer to avoid allocating cat([ctx_expand, node_emb]) of shape [B,N,ctx_dim+embed_dim]
            self.alpha_ctx_proj = nn.Linear(ctx_dim, alpha_hidden, bias=False)
            self.alpha_node_proj = nn.Linear(embed_dim, alpha_hidden, bias=True)
            self.alpha_out = nn.Linear(alpha_hidden, self.cand_phi_dim)
            self.phi_bias_norm = nn.LayerNorm(self.cand_phi_dim, elementwise_affine=False)
        else:
            self.alpha_ctx_proj = None
            self.alpha_node_proj = None
            self.alpha_out = None
            self.phi_bias_norm = None
        self.phi_bias_scale = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))

        # B-2 for raw static node features (dim follows config or node_dim)
        self.raw_feat_dim = _resolve_raw_feat_dim(self.node_dim)
        if self.use_raw_feature_bias:
            raw_hidden = max(32, embed_dim // 2)
            self.raw_alpha_ctx_proj = nn.Linear(ctx_dim, raw_hidden, bias=False)
            self.raw_alpha_node_proj = nn.Linear(embed_dim, raw_hidden, bias=True)
            self.raw_alpha_out = nn.Linear(raw_hidden, self.raw_feat_dim)
            self.raw_bias_norm = nn.LayerNorm(self.raw_feat_dim, elementwise_affine=False)
        else:
            self.raw_alpha_ctx_proj = None
            self.raw_alpha_node_proj = None
            self.raw_alpha_out = None
            self.raw_bias_norm = None
        self.raw_bias_scale = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))

        self.gate_node = nn.Linear(embed_dim, 1)
        self.gate_ctx = nn.Linear(ctx_dim, 1)
        nn.init.zeros_(self.gate_node.weight)
        nn.init.zeros_(self.gate_node.bias)
        nn.init.zeros_(self.gate_ctx.weight)
        try:
            self.gate_ctx.bias.data.fill_(float(getattr(train_defaults, "gate_init_bias", 2.0)))
        except Exception:
            self.gate_ctx.bias.data.fill_(2.0)

    def encode(
        self,
        node_feats: torch.Tensor,
        depot_feat: torch.Tensor,
        pad_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[Dict[str, torch.Tensor]]]:
        node_emb = self.node_enc(node_feats)
        key_padding_mask = None
        if pad_mask is not None:
            key_padding_mask = (pad_mask < 0.5)

        model_alpha_attn_gate = getattr(self, "alpha_attn_gate", None)
        if model_alpha_attn_gate is not None and hasattr(self.encoder, "layers"):
            for enc_layer in self.encoder.layers:
                setattr(enc_layer, "alpha_attn_gate", float(model_alpha_attn_gate))
        node_emb, aux_loss, enc_stats = self.encoder(
            node_emb,
            src_key_padding_mask=key_padding_mask,
        )

        depot_emb = self.depot_enc(depot_feat)
        if pad_mask is not None:
            valid = (pad_mask > 0.5).to(node_emb.dtype).unsqueeze(-1)
            node_emb = node_emb * valid
            graph_pool = (node_emb * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
            graph_emb = self.graph_proj(graph_pool)
        else:
            graph_emb = self.graph_proj(node_emb.mean(dim=1))
        return node_emb, depot_emb, graph_emb, aux_loss, enc_stats

    def _expand_batch_latent(
        self,
        latent: Optional[torch.Tensor],
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        if self.latent_dim <= 0:
            return None
        if latent is None:
            latent = torch.zeros(batch_size, self.latent_dim, device=device, dtype=dtype)
        latent = latent.to(device=device, dtype=dtype)
        if latent.dim() == 1:
            latent = latent.unsqueeze(0)
        if latent.dim() != 2:
            raise ValueError(f"latent must be [B,z] or [z], got {tuple(latent.shape)}")
        if latent.size(0) == 1 and batch_size > 1:
            latent = latent.expand(batch_size, -1)
        elif latent.size(0) != batch_size:
            raise ValueError(f"latent batch mismatch: expected {batch_size}, got {latent.size(0)}")
        if latent.size(1) != self.latent_dim:
            raise ValueError(f"latent dim mismatch: expected {self.latent_dim}, got {latent.size(1)}")
        return latent

    def _apply_latent_film(self, h: torch.Tensor, latent_rep: Optional[torch.Tensor]) -> torch.Tensor:
        if latent_rep is None or self.latent_film is None:
            return h
        z = latent_rep.to(dtype=h.dtype, device=h.device)
        if z.dim() == 2 and h.dim() == 3:
            z = z.unsqueeze(1)
        film = self.latent_film(z)
        gamma, beta = film.chunk(2, dim=-1)
        gamma = torch.tanh(gamma)
        return (1.0 + gamma) * h + beta

    def _inject_latent_into_ctx(self, ctx: torch.Tensor, latent_rep: Optional[torch.Tensor]) -> torch.Tensor:
        if latent_rep is None:
            return ctx
        if self.latent_injection_mode == "legacy_concat":
            return torch.cat([ctx, latent_rep.to(device=ctx.device, dtype=ctx.dtype)], dim=-1)
        ctx = self._apply_latent_film(ctx, latent_rep)
        if self.latent_context_proj is not None:
            ctx = ctx + self.latent_context_proj(latent_rep.to(device=ctx.device, dtype=ctx.dtype))
        return ctx

    def decode_step(
        self,
        node_emb: torch.Tensor,
        node_feats: Optional[torch.Tensor],
        depot_emb: torch.Tensor,
        graph_emb: torch.Tensor,
        dyn_feats: torch.Tensor,
        last_action: torch.Tensor,
        mask: torch.Tensor,
        latent: Optional[torch.Tensor] = None,
        k_precomputed: Optional[torch.Tensor] = None,
        cand_phi: Optional[torch.Tensor] = None,
        return_extra: bool = False,
        return_alpha: bool = False,
    ) -> torch.Tensor:
        # Single-route output: logits [B, N+1].
        B, N, _ = node_emb.size()
        device = node_emb.device
        if dyn_feats.dim() != 2 or dyn_feats.size(0) != B:
            raise ValueError(f"dyn_feats must be [B,D]={B,self.dyn_dim}, got {tuple(dyn_feats.shape)}")
        if dyn_feats.size(-1) != self.dyn_dim:
            raise ValueError(
                f"dyn_feats last dim={dyn_feats.size(-1)} != expected dyn_dim={self.dyn_dim}"
            )

        safe_idx = last_action.clamp(max=N - 1)
        cust_emb = node_emb[torch.arange(B, device=device), safe_idx]
        use_depot = last_action == N
        last_emb = torch.where(use_depot.unsqueeze(1), depot_emb, cust_emb)

        dyn_emb = self.dyn_proj(dyn_feats)
        latent_rep = self._expand_batch_latent(
            latent,
            batch_size=B,
            device=device,
            dtype=node_emb.dtype,
        )
        ctx = torch.cat([graph_emb, last_emb, dyn_emb], dim=-1)  # [B,3d]
        ctx = self._inject_latent_into_ctx(ctx, latent_rep)

        
        q_raw = self.W_q(ctx)
        q = F.normalize(q_raw, dim=-1).unsqueeze(1)
        if k_precomputed is not None:
            k_static = k_precomputed
        else:
            k_static = self.W_k(node_emb)
        if cand_phi is not None and self.cand_phi_dim > 0 and self.phi_proj is not None:
            if cand_phi.size(-1) != self.cand_phi_dim:
                raise ValueError(f"cand_phi last dim={cand_phi.size(-1)} != cand_phi_dim={self.cand_phi_dim}")
            cand_phi = cand_phi.to(device=node_emb.device, dtype=node_emb.dtype)
            k_dyn = self.phi_proj(cand_phi)
            k = k_static + k_dyn
        else:
            k = k_static

        gate_node = self.gate_node(node_emb)  # [B,N,1]
        gate_ctx = self.gate_ctx(ctx).unsqueeze(1)  # [B,1,1]
        gate = torch.sigmoid(gate_node + gate_ctx).squeeze(-1)  # [B,N]

        scores = torch.matmul(q, k.transpose(1, 2)).squeeze(1)
        scores_base = scores
        scores_gate = scores_base * gate
        alpha = float(getattr(self, "alpha_gate", 1.0))
        scores = (1.0 - alpha) * scores_base + alpha * scores_gate

        extra = {} if return_extra else None
        if cand_phi is not None and self.cand_phi_dim > 0 and self.alpha_out is not None:
            if cand_phi.size(-1) != self.cand_phi_dim:
                raise ValueError(f"cand_phi last dim={cand_phi.size(-1)} != cand_phi_dim={self.cand_phi_dim}")
            cand_phi_for_bias = cand_phi.to(device=node_emb.device, dtype=node_emb.dtype)
            if self.phi_bias_norm is not None:
                cand_phi_for_bias = self.phi_bias_norm(cand_phi_for_bias)

            # Factorized first layer: W_ctx(ctx_t) + W_node(node_j), then projection to F
            alpha_hidden = self.alpha_ctx_proj(ctx).unsqueeze(1) + self.alpha_node_proj(node_emb)  # [B,N,H]
            alpha_logits = self.alpha_out(F.relu(alpha_hidden))  # [B,N,F]
            alpha_weights = F.softmax(alpha_logits, dim=-1)  # [B,N,F]
            phi_bias = (alpha_weights * cand_phi_for_bias).sum(dim=-1)  # [B,N]
            scores = scores + self.phi_bias_scale.to(dtype=scores.dtype) * phi_bias

            if return_extra:
                extra.update(
                    {
                        "phi_bias_mean": phi_bias.mean().detach(),
                        "phi_bias_max": phi_bias.amax().detach(),
                        "phi_bias_scale": self.phi_bias_scale.detach(),
                    }
                )
                if return_alpha:
                    extra["alpha"] = alpha_weights.detach()

        if node_feats is not None and self.use_raw_feature_bias and self.raw_alpha_out is not None:
            if node_feats.size(1) != N:
                raise ValueError(f"node_feats dim1={node_feats.size(1)} != N={N}")
            if node_feats.size(-1) < self.raw_feat_dim:
                raise ValueError(
                    f"node_feats last dim={node_feats.size(-1)} < raw_feat_dim={self.raw_feat_dim}"
                )
            node_feats_for_bias = node_feats[..., : self.raw_feat_dim].to(device=node_emb.device, dtype=node_emb.dtype)
            if self.raw_bias_norm is not None:
                node_feats_for_bias = self.raw_bias_norm(node_feats_for_bias)

            raw_alpha_hidden = self.raw_alpha_ctx_proj(ctx).unsqueeze(1) + self.raw_alpha_node_proj(node_emb)  # [B,N,H]
            raw_alpha_logits = self.raw_alpha_out(F.relu(raw_alpha_hidden))  # [B,N,R]
            raw_alpha_weights = F.softmax(raw_alpha_logits, dim=-1)  # [B,N,R]
            raw_bias = (raw_alpha_weights * node_feats_for_bias).sum(dim=-1)  # [B,N]
            scores = scores + self.raw_bias_scale.to(dtype=scores.dtype) * raw_bias

            if return_extra:
                extra.update(
                    {
                        "raw_bias_mean": raw_bias.mean().detach(),
                        "raw_bias_max": raw_bias.amax().detach(),
                        "raw_bias_scale": self.raw_bias_scale.detach(),
                    }
                )
                if return_alpha:
                    extra["raw_alpha"] = raw_alpha_weights.detach()

        cust_mask = mask[:, :N] > 0.5
        scores = scores.masked_fill(~cust_mask, -1e9)

        
        end_logit = self.end_scorer(ctx)  # [B,1]
        logits = torch.cat([scores, end_logit], dim=1)
        if return_extra:
            return logits, (extra or {})
        return logits

def build_dyn_features(env: "BatchVRPTWEnv") -> torch.Tensor:
    eps = 1e-6
    B = env.B
    device = env.device

    coords = env.coords  # [B, N+1, 2]
    valid_cust = env.demand[:, 1:] > 0
    valid_mask = torch.cat(
        [torch.ones((B, 1), device=device, dtype=torch.bool), valid_cust.to(device=device)], dim=1
    )
    min_x, min_y, max_xy = _compute_coord_norm_params(coords, valid_mask)

    batch_idx = torch.arange(B, device=device)
    cur_xy = coords[batch_idx, env.loc]  # [B, 2]
    curr_x_norm = (cur_xy[:, 0] - min_x) / (max_xy + eps)
    curr_y_norm = (cur_xy[:, 1] - min_y) / (max_xy + eps)

    depot_due = env.due[:, 0]
    max_due = env.due.max(dim=1).values
    horizon = torch.where(
        depot_due > 0,
        depot_due - env.ready[:, 0],
        max_due - env.ready[:, 0],
    ).clamp_min(1.0)
    curr_time_ratio = env.time / (horizon + eps)

    remaining_cap_ratio = (env.capacity - env.load) / (env.capacity + eps)

    dyn = torch.stack(
        [
            curr_x_norm,
            curr_y_norm,
            curr_time_ratio,
            remaining_cap_ratio,
        ],
        dim=1,
    )
    dyn = torch.nan_to_num(dyn, nan=0.0, posinf=0.0, neginf=0.0)
    dyn = torch.clamp(dyn, min=-10.0, max=10.0)
    assert dyn.size(-1) == get_dyn_feature_dim(), f"build_dyn_features dim mismatch: {dyn.size(-1)}"
    return dyn






def build_scheduler(optimizer, total_updates: int, lr: float, lr_schedule: str, last_epoch: int = -1):
    name = (lr_schedule or "cosine").strip().lower()
    total_updates = max(1, int(total_updates))
    if name == "fixed":
        return LambdaLR(
            optimizer,
            lr_lambda=lambda _s: 1.0,
            last_epoch=last_epoch,
        )
    if name == "linear":
        return LambdaLR(
            optimizer,
            lr_lambda=lambda s: max(0.0, 1.0 - s / float(total_updates)),
            last_epoch=last_epoch,
        )
    if name == "cosine":
        return CosineAnnealingLR(optimizer, T_max=total_updates, eta_min=lr * 0.05, last_epoch=last_epoch)
    if name == "multistep":
        raw_milestones = tuple(getattr(train_defaults, "lr_milestones", (0.8, 0.95)) or ())
        milestones: List[int] = []
        for raw in raw_milestones:
            try:
                val = float(raw)
            except Exception:
                continue
            if val <= 1.0:
                step = int(round(max(0.0, val) * float(total_updates)))
            else:
                step = int(round(val))
            step = max(1, min(total_updates - 1, step))
            milestones.append(step)
        milestones = sorted(set(m for m in milestones if 0 < m < total_updates))
        if not milestones:
            return LambdaLR(
                optimizer,
                lr_lambda=lambda _s: 1.0,
                last_epoch=last_epoch,
            )
        gamma = float(max(1e-6, getattr(train_defaults, "lr_gamma", 0.1)))
        return MultiStepLR(optimizer, milestones=milestones, gamma=gamma, last_epoch=last_epoch)
    raise ValueError(f"Unsupported lr_schedule='{name}'. Expected one of: fixed, cosine, linear, multistep.")


def three_phase_schedule(
    epoch: int,
    total_epochs: int,
    sigma_0: float = 1.0,
    sigma_mid: float = 0.3,
    sigma_min: float = 0.1,
    ent_0: float = 1.0,
    ent_mid: float = 0.3,
    ent_min: float = 0.05,
    gate_mid: float = 0.7,
    phase1_end: float = 0.2,
    phase2_end: float = 0.6,
):
    """
    Returns (sigma, entropy_coef, alpha_gate_schedule).
    The same alpha value is used for both:
    - decoder logit gate blend (`model.alpha_gate`)
    - encoder gated-attention blend (`model.alpha_attn_gate`)
    """
    x = epoch / float(max(total_epochs, 1))
    if x <= phase1_end:
        sigma = sigma_0
        ent = ent_0
        alpha_gate = 0.0
    elif x <= phase2_end:
        t = (x - phase1_end) / max(1e-9, (phase2_end - phase1_end))
        sigma = sigma_0 + t * (sigma_mid - sigma_0)
        ent = ent_0 + t * (ent_mid - ent_0)
        alpha_gate = 0.0 + t * (gate_mid - 0.0)
    else:
        t = (x - phase2_end) / max(1e-9, (1.0 - phase2_end))
        sigma = sigma_mid + t * (sigma_min - sigma_mid)
        ent = ent_mid + t * (ent_min - ent_mid)
        alpha_gate = gate_mid + t * (1.0 - gate_mid)
    return float(sigma), float(ent), float(alpha_gate)


def _set_model_gate_alphas(model: nn.Module, alpha_gate: float) -> float:
    alpha = float(alpha_gate)
    # Keep decoder and encoder gate schedules aligned.
    setattr(model, "alpha_gate", alpha)
    setattr(model, "alpha_attn_gate", alpha)
    return alpha


def alpha_schedule(progress: float, alpha_start: float = 350.0, alpha_end: float = 250.0, warmup_pct: float = 0.3) -> float:
    """Example vehicle-penalty schedule for training: linear warmup to target alpha."""
    p = float(max(0.0, min(1.0, progress)))
    w = float(max(1e-9, warmup_pct))
    if p < w:
        t = p / w
        return float(alpha_start + t * (alpha_end - alpha_start))
    return float(alpha_end)


def create_csv_logger(log_path: str):
    if not log_path:
        return None, None
    header_needed = not os.path.exists(log_path)
    log_f = open(log_path, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(
        log_f,
        fieldnames=[
            "epoch",
            "update",
            "loss",
            "mean_cost",
            "distance",
            "vehicles",  # vehicle_count: number of used vehicles (served >=1 customer)
            "entropy",
            "grad_norm",
            "best_mean_cost",
            "lr",
            "grad_ema",
            "latent_sigma",
            "latent_noise_mode",
            "alpha_gate",
        ],
    )
    if header_needed:
        writer.writeheader()
    return writer, log_f


def resume_training(model, optimizer, resume_path: Optional[str], device: str, total_updates: int, steps_per_epoch: int, lr: float, lr_schedule: str):
    resume_update = 0
    resume_epoch = 0
    best_mean_cost = float("inf")
    scheduler = build_scheduler(optimizer, total_updates, lr, lr_schedule)

    def _load_state_compat(state_obj) -> bool:
        state_obj = normalize_checkpoint_state_dict(state_obj)
        try:
            model.load_state_dict(state_obj)
            return True
        except Exception as e:
            model.load_state_dict(state_obj, strict=False)
            print(f"[resume] loaded with strict=False due to key mismatch: {e}")
            return True

    if resume_path:
        try:
            ckpt = torch.load(resume_path, map_location=device, weights_only=False)
            loaded = False
            if isinstance(ckpt, dict):
                if "model" in ckpt:
                    loaded = _load_state_compat(ckpt["model"])
                elif "model_state_dict" in ckpt:
                    loaded = _load_state_compat(ckpt["model_state_dict"])
                else:
                    try:
                        loaded = _load_state_compat(ckpt)
                    except Exception:
                        loaded = False
                if loaded:
                    if "optimizer" in ckpt:
                        try:
                            optimizer.load_state_dict(ckpt["optimizer"])
                        except Exception as e:
                            print(f"[resume] skip optimizer state due to mismatch: {e}")
                    resume_update = int(ckpt.get("update", ckpt.get("step", 0)) or 0)
                    resume_epoch = int(ckpt.get("epoch", resume_update // max(1, steps_per_epoch)))
                    best_mean_cost = float(ckpt.get("best_mean_cost", float("inf")))
                    scheduler = build_scheduler(optimizer, total_updates + resume_update, lr, lr_schedule, last_epoch=resume_update - 1)
                    if "scheduler" in ckpt:
                        try:
                            scheduler.load_state_dict(ckpt["scheduler"])
                        except Exception:
                            pass
            if not loaded:
                _load_state_compat(ckpt)
            print(f"[resume] loaded {resume_path}, start epoch={resume_epoch}, update={resume_update}")
        except Exception as e:
            print(f"[resume] failed to load {resume_path}, using fresh init: {e}")
    return scheduler, resume_update, resume_epoch, best_mean_cost


def lexicographic_key(infeasible: bool, unserved: int, veh: int, dist: float) -> tuple:
    """Solomon-style lexicographic objective key: infeasible -> unserved -> vehicles -> distance."""
    return (bool(infeasible), int(unserved), int(veh), float(dist))




def log_epoch_summary(
    epoch: int,
    total_epochs: int,
    ep_count: int,
    ep_loss_sum: float,
    ep_cost_sum: float,
    ep_dist_sum: float,
    ep_veh_sum: float,
    ep_gn_sum: float,
    ep_ent_sum: float,
    ep_ent_count: int,
    best_mean_cost: float,
    current_lr: float,
    writer,
    log_f,
    global_update: int,
    model,
    optimizer,
    scheduler,
    grad_ema: float | None = None,
    latent_sigma: float | None = None,
    latent_noise_mode: str | None = None,
    alpha_gate: float | None = None,
):
    avg_loss = ep_loss_sum / ep_count
    avg_cost = ep_cost_sum / ep_count
    avg_dist = ep_dist_sum / ep_count
    avg_veh = ep_veh_sum / ep_count
    avg_gn = ep_gn_sum / ep_count
    avg_ent = ep_ent_sum / ep_ent_count if ep_ent_count > 0 else None
    ent_for_print = avg_ent if avg_ent is not None else 0.0
    if avg_cost < best_mean_cost:
        best_mean_cost = avg_cost
        os.makedirs("pt", exist_ok=True)
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch + 1,
                "update": global_update,
                "best_mean_cost": best_mean_cost,
            },
            "pt/best.pt",
        )
    msg_parts = [
        f"[epoch {epoch+1}/{total_epochs} summary] "
        f"loss={avg_loss:.4f}, cost={avg_cost:.2f}, veh={avg_veh:.2f}, "
        f"dist={avg_dist:.2f}, grad_norm={avg_gn:.4f}, lr={current_lr:.2e}, "
        f"mean_best_cost={best_mean_cost:.2f}, entropy={ent_for_print:.4f}"
    ]
    if latent_sigma is not None:
        msg_parts.append(
            f"latent_sigma={float(latent_sigma):.4f}, latent_noise_mode={str(latent_noise_mode or 'schedule')}"
        )
    if writer is not None:
        writer.writerow(
            {
                "epoch": epoch + 1,
                "update": global_update,
                "loss": avg_loss,
                "mean_cost": avg_cost,
                "distance": avg_dist,
                "vehicles": avg_veh,
                "entropy": avg_ent,
                "grad_norm": avg_gn,
                "best_mean_cost": best_mean_cost,
                "lr": current_lr,
                "grad_ema": grad_ema,
                "latent_sigma": latent_sigma,
                "latent_noise_mode": latent_noise_mode,
                "alpha_gate": alpha_gate,
            }
        )
        log_f.flush()
    return msg_parts, best_mean_cost


def _build_round_robin_start_actions(
    *,
    model,
    b_nf_base: torch.Tensor,
    b_df_base: torch.Tensor,
    b_pm_base: torch.Tensor,
    b_env_base: Dict[str, torch.Tensor],
    nodes_enc_base: torch.Tensor,
    depot_emb_base: torch.Tensor,
    graph_emb_base: torch.Tensor,
    k_pre_base: torch.Tensor,
    latent_multi_k: int,
    device: str,
    target_dyn_dim: int,
    target_cand_dim: int,
    use_bf16: bool,
    vehicle_limit_base: torch.Tensor,
    vehicle_max_base: int,
    vehicle_limit_source_base: str,
) -> torch.Tensor:
    start_pools = _extract_greedy_start_pools(
        model=model,
        b_nf_base=b_nf_base,
        b_df_base=b_df_base,
        b_pm_base=b_pm_base,
        b_env_base=b_env_base,
        nodes_enc_base=nodes_enc_base,
        depot_emb_base=depot_emb_base,
        graph_emb_base=graph_emb_base,
        k_pre_base=k_pre_base,
        device=device,
        target_dyn_dim=target_dyn_dim,
        target_cand_dim=target_cand_dim,
        use_bf16=use_bf16,
        vehicle_limit_base=vehicle_limit_base,
        vehicle_max_base=vehicle_max_base,
        vehicle_limit_source_base=vehicle_limit_source_base,
    )
    return _assign_round_robin_start_actions(
        start_pools=start_pools,
        repeats_per_instance=latent_multi_k,
        device=device,
        view_major=False,
    )


def _extract_greedy_start_pools(
    *,
    model,
    b_nf_base: torch.Tensor,
    b_df_base: torch.Tensor,
    b_pm_base: torch.Tensor,
    b_env_base: Dict[str, torch.Tensor],
    nodes_enc_base: torch.Tensor,
    depot_emb_base: torch.Tensor,
    graph_emb_base: torch.Tensor,
    k_pre_base: torch.Tensor,
    device: str,
    target_dyn_dim: int,
    target_cand_dim: int,
    use_bf16: bool,
    vehicle_limit_base: torch.Tensor,
    vehicle_max_base: int,
    vehicle_limit_source_base: str,
) -> List[List[int]]:
    """
    One greedy rollout per base instance -> collect each vehicle route's first customer.
    """
    B = int(b_nf_base.size(0))
    if B <= 0:
        return []
    env_greedy = BatchVRPTWEnv(
        b_env_base,
        device=device,
        track_routes=True,
        vehicle_limit=vehicle_limit_base,
        vehicle_max=vehicle_max_base,
        vehicle_limit_source=vehicle_limit_source_base,
    )
    prev_action = torch.full((B,), env_greedy.N, dtype=torch.long, device=device)
    if int(getattr(model, "latent_dim", 0)) > 0:
        latent_greedy = torch.zeros(B, int(model.latent_dim), device=device, dtype=b_nf_base.dtype)
    else:
        latent_greedy = None

    with torch.no_grad():
        for _ in range(env_greedy.N * 2 + 100):
            mask = env_greedy.get_mask(b_pm_base)
            active = (~env_greedy.finished) & (mask.sum(dim=1) > 0.5)
            if not active.any():
                break
            safe_mask = mask.clone()
            safe_mask[~active] = 0
            safe_mask[~active, env_greedy.N] = 1.0
            row_sum = safe_mask.sum(dim=1, keepdim=True)
            zero_rows = row_sum.squeeze(1) <= 0
            if zero_rows.any():
                safe_mask[zero_rows] = 0
                safe_mask[zero_rows, env_greedy.N] = 1.0

            dyn_global = build_dyn_features(env_greedy)
            dyn_global = _match_dyn_feat_dim(dyn_global, target_dyn_dim)
            cand_phi = env_greedy.get_candidate_features(b_pm_base)
            cand_phi = _match_cand_phi_dim(cand_phi, target_cand_dim)
            with bf16_autocast(use_bf16):
                logits = model.decode_step(
                    nodes_enc_base,
                    b_nf_base,
                    depot_emb_base,
                    graph_emb_base,
                    dyn_global,
                    prev_action,
                    safe_mask,
                    latent=latent_greedy,
                    k_precomputed=k_pre_base,
                    cand_phi=cand_phi,
                )
            greedy_logits = logits.masked_fill(safe_mask < 0.5, -1e9)
            actions = torch.argmax(greedy_logits, dim=1)
            safe_actions = actions.clone()
            safe_actions[~active] = env_greedy.N
            env_greedy.step(safe_actions)
            prev_action = safe_actions
            fully_done = env_greedy.compute_terminal_mask(b_pm_base)
            if fully_done.any():
                env_greedy.finished[fully_done] = True

    start_pools: List[List[int]] = []
    for b in range(B):
        routes = _collect_final_routes(env_greedy, b)
        starts = [int(r[0]) for r in routes if len(r) > 0]
        if not starts:
            fallback = torch.nonzero(b_pm_base[b] > 0.5, as_tuple=False).squeeze(1)
            if int(fallback.numel()) > 0:
                starts = [int(fallback[0].item())]
        start_pools.append(starts)
    return start_pools


def _assign_round_robin_start_actions(
    *,
    start_pools: List[List[int]],
    repeats_per_instance: int,
    device: str,
    view_major: bool,
) -> torch.Tensor:
    B = len(start_pools)
    R = max(1, int(repeats_per_instance))
    start_actions = torch.zeros((B * R,), dtype=torch.long, device=device)
    for b, starts in enumerate(start_pools):
        if not starts:
            continue
        for r in range(R):
            idx = (r * B + b) if bool(view_major) else (b * R + r)
            start_actions[idx] = int(starts[r % len(starts)])
    return start_actions


def train_one_batch(
    indices: torch.Tensor,
    latent_multi_k: int,
    device: str,
    model,
    optimizer,
    scheduler,
    all_node_feats: torch.Tensor,
    all_depot_feats: torch.Tensor,
    all_pad_masks: torch.Tensor,
    all_env_data: Dict[str, torch.Tensor],
    instances: List[Instance],
    max_grad_norm: float,
    update_idx: int,
    ema_state: 'EMAState',
    latent_sigma: float,
    per_inst_best: Dict[str, Dict[str, object]],
    entropy_coef: float,
    progress: float = 1.0,
    use_bf16: bool = False,
):
    indices_rep = indices.repeat_interleave(latent_multi_k) if latent_multi_k > 1 else indices
    BK = len(indices_rep)

    # Encode static graph once per original instance (B), then expand to BK for rollout.
    b_nf_base = all_node_feats[indices].to(device)   # [B,N,d]
    b_df_base = all_depot_feats[indices].to(device)  # [B,d]
    b_pm_base = all_pad_masks[indices].to(device)    # [B,N]
    b_env_base = {k: v[indices].to(device) for k, v in all_env_data.items()}

    if latent_multi_k > 1:
        b_nf = b_nf_base.repeat_interleave(latent_multi_k, dim=0)
        b_pm = b_pm_base.repeat_interleave(latent_multi_k, dim=0)
        b_env_data = {k: v.repeat_interleave(latent_multi_k, dim=0) for k, v in b_env_base.items()}
    else:
        b_nf = b_nf_base
        b_pm = b_pm_base
        b_env_data = b_env_base

    vehicle_limit_base, vehicle_max_base, vehicle_limit_source_base = _resolve_vehicle_limits_from_config(
        b_env_base,
        b_pm_base,
    )
    if latent_multi_k > 1:
        vehicle_limit = vehicle_limit_base.repeat_interleave(latent_multi_k, dim=0)
    else:
        vehicle_limit = vehicle_limit_base
    vehicle_max = int(vehicle_max_base)
    vehicle_limit_source = str(vehicle_limit_source_base)
    target_dyn_dim = int(getattr(getattr(model, "dyn_proj", None), "in_features", get_dyn_feature_dim()))
    target_cand_dim = int(getattr(model, "cand_phi_dim", get_cand_phi_feature_dim()))
    env = BatchVRPTWEnv(
        b_env_data,
        device=device,
        track_routes=False,
        vehicle_limit=vehicle_limit,
        vehicle_max=vehicle_max,
        vehicle_limit_source=vehicle_limit_source,
    )
    with bf16_autocast(use_bf16):
        nodes_enc_base, depot_emb_base, graph_emb_base, _enc_aux, _ = model.encode(
            b_nf_base, b_df_base, b_pm_base
        )
        k_pre_base = model.W_k(nodes_enc_base)
    if latent_multi_k > 1:
        nodes_enc = nodes_enc_base.repeat_interleave(latent_multi_k, dim=0)
        depot_emb = depot_emb_base.repeat_interleave(latent_multi_k, dim=0)
        graph_emb = graph_emb_base.repeat_interleave(latent_multi_k, dim=0)
        k_pre = k_pre_base.repeat_interleave(latent_multi_k, dim=0)
    else:
        nodes_enc = nodes_enc_base
        depot_emb = depot_emb_base
        graph_emb = graph_emb_base
        k_pre = k_pre_base
    log_probs: List[torch.Tensor] = []
    entropies: List[torch.Tensor] = []
    exact_best_every = int(getattr(train_defaults, "train_exact_best_every", 0))
    do_exact_best = bool(exact_best_every > 0 and ((update_idx + 1) % exact_best_every == 0))
    action_trace: List[torch.Tensor] = []
    prev_action = torch.full((BK,), env.N, dtype=torch.long, device=device)
    alpha_veh = float(alpha_schedule(progress))
    if getattr(model, "latent_dim", 0) > 0:
        latent = (latent_sigma * torch.randn(BK, model.latent_dim, device=device))
    else:
        latent = None
    use_greedy_start_pool = bool(getattr(train_defaults, "use_greedy_start_pool", False))
    forced_start_actions = None
    if use_greedy_start_pool:
        forced_start_actions = _build_round_robin_start_actions(
            model=model,
            b_nf_base=b_nf_base,
            b_df_base=b_df_base,
            b_pm_base=b_pm_base,
            b_env_base=b_env_base,
            nodes_enc_base=nodes_enc_base,
            depot_emb_base=depot_emb_base,
            graph_emb_base=graph_emb_base,
            k_pre_base=k_pre_base,
            latent_multi_k=latent_multi_k,
            device=device,
            target_dyn_dim=target_dyn_dim,
            target_cand_dim=target_cand_dim,
            use_bf16=use_bf16,
            vehicle_limit_base=vehicle_limit_base,
            vehicle_max_base=vehicle_max_base,
            vehicle_limit_source_base=vehicle_limit_source_base,
        )

    for step_idx in range(env.N + 50):
        mask = env.get_mask(b_pm)
        active = (~env.finished) & (mask.sum(dim=1) > 0.5)
        if not active.any():
            break

        safe_mask = mask.clone()
        safe_mask[~active] = 0
        safe_mask[~active, env.N] = 1.0

        row_sum = safe_mask.sum(dim=1, keepdim=True)
        zero_rows = row_sum.squeeze(1) <= 0
        if zero_rows.any():
            safe_mask[zero_rows] = 0
            safe_mask[zero_rows, env.N] = 1.0
        dyn_global = build_dyn_features(env)
        dyn_global = _match_dyn_feat_dim(dyn_global, target_dyn_dim)
        cand_phi = env.get_candidate_features(b_pm)
        cand_phi = _match_cand_phi_dim(cand_phi, target_cand_dim)
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
                cand_phi=cand_phi,
            )
            raw_probs = F.softmax(logits, dim=1)
            probs = _normalize_probs(raw_probs, safe_mask)
        dist = Categorical(probs.float())
        if step_idx == 0 and forced_start_actions is not None:
            actions = forced_start_actions.clone()
            actions[~active] = env.N
            picked_ok = safe_mask.gather(1, actions.unsqueeze(1)).squeeze(1) > 0.5
            if (~picked_ok).any():
                fallback = torch.argmax(probs, dim=1)
                actions[~picked_ok] = fallback[~picked_ok]
        else:
            actions = dist.sample()
        log_p = dist.log_prob(actions)
        entropies.append(dist.entropy())
        log_probs.append(log_p)
        safe_actions = actions.clone()
        safe_actions[~active] = env.N
        env.step(safe_actions)
        if do_exact_best:
            action_trace.append(safe_actions.detach())
        prev_action = safe_actions
        fully_done = env.compute_terminal_mask(b_pm)
        if fully_done.any():
            env.finished[fully_done] = True
    # Tensor-only training cost for speed: distance + alpha*vehicles + unserved_penalty + infeasible_penalty.
    visited_counts = env.visited.sum(dim=1).to(dtype=torch.float32)
    real_counts = b_pm.sum(dim=1).to(dtype=torch.float32)
    unserved = (real_counts - visited_counts).clamp_min(0.0)
    dist_tensor = env.total_dist.to(dtype=torch.float32)
    # vehicle_count definition: vehicles used by serving >=1 customer.
    veh_tensor = env.get_used_vehicle_count(include_open_route=True).to(dtype=torch.float32)
    infeasible_tensor = env.infeasible.to(dtype=torch.float32)
    costs_tensor = (
        dist_tensor
        + alpha_veh * veh_tensor
        + 500.0 * unserved
        + 5000.0 * infeasible_tensor
    )

    # Optional low-frequency exact recomputation to refresh per-instance best records.
    if do_exact_best:
        env_exact = BatchVRPTWEnv(
            b_env_data,
            device=device,
            track_routes=True,
            vehicle_limit=vehicle_limit,
            vehicle_max=vehicle_max,
            vehicle_limit_source=vehicle_limit_source,
        )
        for a in action_trace:
            env_exact.step(a)
        for i in range(BK):
            inst_idx = indices_rep[i].item()
            inst = instances[inst_idx]
            final_routes = _collect_final_routes(env_exact, i)
            dist_val = 0.0
            feas_all = True
            served_set = set()
            for r in final_routes:
                d, feas, _ = route_cost_and_feasible(inst, r)
                if not feas:
                    feas_all = False
                    dist_val = 1e6
                    served_set.update(r)
                    break
                dist_val += d
                served_set.update(r)
            n_cust = len(inst.customers)
            served_count = len(served_set)
            unserved_i = max(0, n_cust - served_count)
            veh_i = len(final_routes)
            cost = compute_cost(
                dist_val,
                final_routes,
                alpha=250.0,
                unserved=unserved_i,
                unserved_penalty=500.0,
                infeasible=not feas_all,
                infeasible_penalty=5000.0,
            )
            if unserved_i == 0 and feas_all:
                name = inst.name
                if name not in per_inst_best or cost < per_inst_best[name]["cost"]:
                    per_inst_best[name] = {"cost": cost, "dist": dist_val, "vehicles": veh_i, "routes": final_routes}
    reward = _compute_reward(costs_tensor)
    inst_ids = indices_rep.to(device)
    adv, ema_state, _stats = compute_advantage(
        reward=reward,
        group_ids=inst_ids,
        ema_state=ema_state,
        eps=1e-8,
    )
    log_probs_stack = torch.stack(log_probs, dim=0)
    sum_log_probs = log_probs_stack.sum(dim=0)
    policy_loss = -(adv * sum_log_probs).mean()
    entropy_term = torch.stack(entropies, dim=0).mean() if entropies else None
    step_entropy = float(entropy_term.item()) if entropy_term is not None else None
    total_loss = policy_loss
    if entropy_coef > 0.0 and entropy_term is not None:
        total_loss = total_loss - entropy_coef * entropy_term
    optimizer.zero_grad()
    total_loss.backward()
    grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm))
    optimizer.step()
    scheduler.step()
    _stats["alpha_veh"] = float(alpha_veh)
    update_idx += 1
    mean_cost = costs_tensor.mean().item()
    mean_veh = float(veh_tensor.mean().item())
    mean_dist = float(dist_tensor.mean().item())
    return update_idx, total_loss.item(), mean_cost, mean_dist, mean_veh, grad_norm, step_entropy, ema_state, _stats



def train_neural(
    instances: List[Instance],
    epochs: int = 20,
    batch_size: int = 56,
    device: str = "cuda",
    use_bf16: bool = False,
    log_path: Optional[str] = None,
    lr: float = 5e-4,
    lr_schedule: str = "cosine",
    entropy_coef: float = 0.0,
    resume_path: Optional[str] = None,
    latent_multi_k: int = 1,
    max_grad_norm: float = 21.0,
    latent_dim: int = 0,
    epoch_instance_provider: Optional[Callable[[int], List[Instance]]] = None,
    epoch_instance_count: Optional[int] = None,
    batch_instance_provider: Optional[Callable[[int, int, int], List[Instance]]] = None,
    batch_instance_count: Optional[int] = None,
):
    """
    ??????? epoch?REINFORCE + ?? latent-multi??
    - ?? epoch ???????????
    - ?? = ?? + ???? + ???/?????
    - latent_multi_k>1 ??????? latent 多样化并行 ??
    """
    if epoch_instance_provider is not None and batch_instance_provider is not None:
        raise ValueError("Enable only one of epoch_instance_provider or batch_instance_provider.")
    if not instances and batch_instance_provider is None:
        raise ValueError("???????")

    cand_phi_dim = int(getattr(train_defaults, "cand_phi_dim", get_cand_phi_feature_dim())) if bool(getattr(train_defaults, "use_dynamic_key_aug", False)) else 0
    cand_phi_hidden_dim = int(getattr(train_defaults, "cand_phi_hidden_dim", 0))
    use_raw_feature_bias = bool(getattr(train_defaults, "use_raw_feature_bias", False))
    embed_dim, n_heads, n_layers, ff_dim = _resolve_model_arch_defaults()
    model = AttentionVRPTW(
        node_dim=get_planned_node_feature_dim(),
        dyn_dim=_resolve_model_dyn_dim(),
        embed_dim=embed_dim,
        n_heads=n_heads,
        n_layers=n_layers,
        ff_dim=ff_dim,
        latent_dim=latent_dim,
        cand_phi_dim=cand_phi_dim,
        cand_phi_hidden_dim=cand_phi_hidden_dim,
        use_raw_feature_bias=use_raw_feature_bias,
    ).to(device)
    use_bf16 = resolve_bf16_mode(device, use_bf16, scope="train", verbose=True)
    if use_bf16:
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    # 仅保留基础学习率策略（fixed/cosine/linear）；不再叠加二次自适应控制器。

    total_instances = len(instances)
    planned_epoch_instances: Optional[int] = None
    if batch_instance_provider is not None:
        planned_epoch_instances = int(batch_instance_count or epoch_instance_count or total_instances)
        if planned_epoch_instances <= 0:
            raise ValueError("batch_instance_count must be positive when batch_instance_provider is enabled")
        steps_per_epoch = max(1, math.ceil(planned_epoch_instances / batch_size))
    elif epoch_instance_provider is not None:
        planned_epoch_instances = int(epoch_instance_count or total_instances)
        if planned_epoch_instances <= 0:
            raise ValueError("epoch_instance_count must be positive when epoch_instance_provider is enabled")
        steps_per_epoch = max(1, math.ceil(planned_epoch_instances / batch_size))
    else:
        steps_per_epoch = max(1, math.ceil(total_instances / batch_size))
    total_updates = steps_per_epoch * epochs

    resume_update = 0
    resume_epoch = 0
    best_mean_cost = float("inf")
    ema_state = EMAState(beta=0.9)
    scheduler, resume_update, resume_epoch, best_mean_cost = resume_training(
        model, optimizer, resume_path, device, total_updates, steps_per_epoch, lr, lr_schedule
    )

    if batch_instance_provider is None:
        all_node_feats, all_depot_feats, all_pad_masks, all_env_data = pad_instances(instances)
    else:
        all_node_feats, all_depot_feats, all_pad_masks, all_env_data = None, None, None, {}
    per_inst_best: Dict[str, Dict[str, object]] = {}

    model.train()
    writer, log_f = create_csv_logger(log_path)
    diag_log_every = int(getattr(train_defaults, "diag_log_every", 100) or 0)
    latent_noise_mode = str(getattr(train_defaults, "latent_noise_mode", "schedule")).strip().lower()
    if latent_noise_mode not in {"schedule", "adaptive"}:
        raise ValueError(
            f"Unsupported latent_noise_mode='{latent_noise_mode}'. Expected one of: schedule, adaptive."
        )
    adaptive_sigma_current = float(getattr(train_defaults, "latent_adaptive_sigma_init", 0.0))
    if adaptive_sigma_current <= 0.0:
        adaptive_sigma_current = float(getattr(train_defaults, "latent_max_sigma", 0.7))
    adaptive_sigma_min = float(max(0.0, getattr(train_defaults, "latent_adaptive_sigma_min", 0.1)))
    adaptive_sigma_max = float(
        max(adaptive_sigma_min, getattr(train_defaults, "latent_adaptive_sigma_max", adaptive_sigma_current))
    )
    adaptive_improve_tol = float(max(0.0, getattr(train_defaults, "latent_adaptive_improve_tol", 0.002)))
    adaptive_downscale = float(max(1e-6, getattr(train_defaults, "latent_adaptive_downscale", 0.94)))
    adaptive_upscale = float(max(1e-6, getattr(train_defaults, "latent_adaptive_upscale", 1.06)))
    adaptive_patience = int(max(1, getattr(train_defaults, "latent_adaptive_patience", 1)))
    adaptive_bad_epochs = 0
    adaptive_best_cost = float(best_mean_cost)
    adaptive_sigma_current = float(max(adaptive_sigma_min, min(adaptive_sigma_max, adaptive_sigma_current)))

    global_update = resume_update
    start_epoch = resume_epoch
    # When resuming, interpret `epochs` as additional epochs to train.
    # Example: checkpoint at epoch=100 and epochs=50 -> train [100, 150).
    total_epochs = (resume_epoch + epochs) if (resume_path and resume_epoch > 0) else epochs

    for epoch in tqdm(range(start_epoch, total_epochs), desc="Training"):
        if batch_instance_provider is not None:
            total_instances = int(planned_epoch_instances or batch_size)
            epoch_batches = math.ceil(total_instances / batch_size)
            # Dynamic instances use unique names per epoch; avoid unbounded growth.
            per_inst_best.clear()
        else:
            if epoch_instance_provider is not None:
                epoch_instances = epoch_instance_provider(epoch)
                if not epoch_instances:
                    raise ValueError("epoch_instance_provider returned empty instances")
                instances = epoch_instances
                all_node_feats, all_depot_feats, all_pad_masks, all_env_data = pad_instances(instances)
                total_instances = len(instances)
                # Dynamic instances use unique names per epoch; avoid unbounded growth.
                per_inst_best.clear()

            perm = torch.randperm(total_instances)
            epoch_batches = math.ceil(total_instances / batch_size)

        ep_loss_sum = 0.0
        ep_cost_sum = 0.0
        ep_dist_sum = 0.0
        ep_veh_sum = 0.0
        ep_gn_sum = 0.0
        ep_ent_sum = 0.0
        ep_ent_count = 0
        ep_count = 0
        epoch_last_latent_sigma: float | None = None

        for batch_idx in tqdm(range(epoch_batches), desc=f"Epoch {epoch+1}", leave=False):
            if batch_instance_provider is not None:
                batch_start = batch_idx * batch_size
                batch_expected = min(batch_size, max(0, total_instances - batch_start))
                if batch_expected <= 0:
                    continue
                batch_instances = batch_instance_provider(epoch, batch_idx, batch_expected)
                if not batch_instances:
                    raise ValueError("batch_instance_provider returned empty instances")
                if len(batch_instances) != batch_expected:
                    raise ValueError(
                        f"batch_instance_provider returned {len(batch_instances)} instances, expected {batch_expected}"
                    )
                b_node_feats, b_depot_feats, b_pad_masks, b_env_data = pad_instances(batch_instances)
                indices = torch.arange(batch_expected, dtype=torch.long)
                batch_instances_ref = batch_instances
            else:
                batch_start = batch_idx * batch_size
                indices = perm[batch_start : batch_start + batch_size]
                b_node_feats = all_node_feats
                b_depot_feats = all_depot_feats
                b_pad_masks = all_pad_masks
                b_env_data = all_env_data
                batch_instances_ref = instances
            sigma_0 = float(train_defaults.latent_max_sigma)
            ent_base = float(train_defaults.entropy_coef)
            scheduled_sigma, entropy_coef_s, alpha_gate = three_phase_schedule(
                epoch,
                total_epochs,
                sigma_0=sigma_0,
                ent_0=ent_base,
            )
            if latent_noise_mode == "adaptive":
                latent_sigma = float(adaptive_sigma_current)
            else:
                latent_sigma = float(scheduled_sigma)
            epoch_last_latent_sigma = float(latent_sigma)
            entropy_coef = entropy_coef_s
            alpha_gate = _set_model_gate_alphas(model, alpha_gate)
            progress_in_epoch = (batch_idx / max(1, epoch_batches))
            progress = (epoch + progress_in_epoch) / float(max(1, total_epochs))
            (global_update, loss_item, mean_cost, mean_dist, mean_veh, grad_norm, step_entropy, ema_state, batch_stats) = train_one_batch(
                indices,
                latent_multi_k,
                device,
                model,
                optimizer,
                scheduler,
                b_node_feats,
                b_depot_feats,
                b_pad_masks,
                b_env_data,
                batch_instances_ref,
                max_grad_norm,
                global_update,
                ema_state,
                float(latent_sigma),
                per_inst_best,
                entropy_coef,
                progress=float(progress),
                use_bf16=bool(use_bf16),
            )
            ep_loss_sum += loss_item
            ep_cost_sum += mean_cost
            ep_dist_sum += mean_dist
            ep_veh_sum += mean_veh
            ep_gn_sum += grad_norm
            ep_count += 1
            if step_entropy is not None:
                ep_ent_sum += step_entropy
                ep_ent_count += 1
            if diag_log_every > 0 and (global_update % diag_log_every == 0):
                diag_extra: List[str] = []
                adv_mode_diag = batch_stats.get("adv_mode")
                if adv_mode_diag is not None:
                    diag_extra.append(f"adv_mode={adv_mode_diag}")
                for key in ("loss_ps", "loss_ss", "loss_inv"):
                    if key in batch_stats and batch_stats[key] is not None:
                        try:
                            diag_extra.append(f"{key}={float(batch_stats[key]):.4f}")
                        except Exception:
                            pass
                print(
                    f"[diag @ step {global_update}] "
                    f"loss={loss_item:.4f}, mean_cost={mean_cost:.2f}, "
                    f"veh={mean_veh:.2f}, dist={mean_dist:.2f}, "
                    f"grad_norm={grad_norm:.4f}, entropy={step_entropy if step_entropy is not None else 'None'}"
                    + (", " + ", ".join(diag_extra) if diag_extra else "")
                )

        current_lr = optimizer.param_groups[0]["lr"]
        if ep_count > 0:
            epoch_mean_cost = ep_cost_sum / ep_count
            if latent_noise_mode == "adaptive":
                improved = epoch_mean_cost < (adaptive_best_cost * (1.0 - adaptive_improve_tol))
                action = "hold"
                old_sigma = float(adaptive_sigma_current)
                if improved:
                    adaptive_best_cost = float(epoch_mean_cost)
                    adaptive_bad_epochs = 0
                    adaptive_sigma_current = max(adaptive_sigma_min, adaptive_sigma_current * adaptive_downscale)
                    action = "down"
                else:
                    adaptive_bad_epochs += 1
                    if adaptive_bad_epochs >= adaptive_patience:
                        adaptive_sigma_current = min(adaptive_sigma_max, adaptive_sigma_current * adaptive_upscale)
                        adaptive_bad_epochs = 0
                        action = "up"
                adaptive_sigma_current = float(max(adaptive_sigma_min, min(adaptive_sigma_max, adaptive_sigma_current)))
                if action != "hold":
                    print(
                        f"[latent_adaptive] epoch={epoch+1}, mean_cost={epoch_mean_cost:.4f}, "
                        f"action={action}, sigma={old_sigma:.4f}->{adaptive_sigma_current:.4f}"
                    )
            msg_parts, best_mean_cost = log_epoch_summary(
                epoch,
                total_epochs,
                ep_count,
                ep_loss_sum,
                ep_cost_sum,
                ep_dist_sum,
                ep_veh_sum,
                ep_gn_sum,
                ep_ent_sum,
                ep_ent_count,
                best_mean_cost,
                current_lr,
                writer,
                log_f,
                global_update,
                model,
                optimizer,
                scheduler,
                grad_ema=None,
                latent_sigma=epoch_last_latent_sigma,
                latent_noise_mode=latent_noise_mode,
                alpha_gate=float(getattr(model, "alpha_gate", 1.0)),
            )

        if ep_count > 0:
            from tqdm import tqdm as _tqdm_alias
            _tqdm_alias.write(" | ".join(msg_parts))

    if log_f is not None:
        log_f.close()

    return model, per_inst_best

def neural_construct_single(
    inst: Instance,
    greedy: bool = True,
    latent_multi_k: int = 1,
    steps_limit: int | None = None,
    device: str = "cuda",
    model: Optional[nn.Module] = None,
    latent_dim: int = 0,
    use_bf16: bool = False,
) -> Tuple[List[List[int]], float]:
    node_feats, depot_feat, pad_mask, env_data = pad_instances([inst])
    node_feats = node_feats.to(device)
    depot_feat = depot_feat.to(device)
    pad_mask = pad_mask.to(device)
    env_data = {k: v.to(device) for k, v in env_data.items()}

    if model is None:
        cand_phi_dim = int(getattr(train_defaults, "cand_phi_dim", get_cand_phi_feature_dim())) if bool(getattr(train_defaults, "use_dynamic_key_aug", False)) else 0
        cand_phi_hidden_dim = int(getattr(train_defaults, "cand_phi_hidden_dim", 0))
        use_raw_feature_bias = bool(getattr(train_defaults, "use_raw_feature_bias", False))
        embed_dim, n_heads, n_layers, ff_dim = _resolve_model_arch_defaults()
        model = AttentionVRPTW(
            node_dim=int(node_feats.size(-1)),
            dyn_dim=_resolve_model_dyn_dim(),
            embed_dim=embed_dim,
            n_heads=n_heads,
            n_layers=n_layers,
            ff_dim=ff_dim,
            latent_dim=latent_dim,
            cand_phi_dim=cand_phi_dim,
            cand_phi_hidden_dim=cand_phi_hidden_dim,
            use_raw_feature_bias=use_raw_feature_bias,
        ).to(device)
    target_node_dim = int(getattr(getattr(model, "node_enc", None), "in_features", node_feats.size(-1)))
    node_feats, depot_feat = _match_model_node_feat_dim(node_feats, depot_feat, target_node_dim)
    target_dyn_dim = int(getattr(getattr(model, "dyn_proj", None), "in_features", get_dyn_feature_dim()))
    target_cand_dim = int(getattr(model, "cand_phi_dim", get_cand_phi_feature_dim()))
    was_training = model.training
    model.eval()
    use_bf16 = resolve_bf16_mode(device, use_bf16, scope="construct_single", verbose=False)
    vehicle_limit, vehicle_max, vehicle_limit_source = _resolve_vehicle_limits_from_config(env_data, pad_mask)

    def decode_once() -> Tuple[List[List[int]], float, int, int, bool]:
        env = BatchVRPTWEnv(
            env_data,
            device=device,
            track_routes=True,
            vehicle_limit=vehicle_limit,
            vehicle_max=vehicle_max,
            vehicle_limit_source=vehicle_limit_source,
        )
        prev_action = torch.full((1,), env.N, dtype=torch.long, device=device)
        latent = torch.randn(1, model.latent_dim, device=device) if getattr(model, "latent_dim", 0) > 0 else None
        max_steps = steps_limit or (env.N * 3 + 50)

        with torch.no_grad():
            with bf16_autocast(use_bf16):
                nodes_enc, depot_emb, graph_emb, _enc_aux, _ = model.encode(node_feats, depot_feat, pad_mask)
                k_pre = model.W_k(nodes_enc)
            for _ in range(max_steps):
                mask = env.get_mask(pad_mask)
                active = (~env.finished) & (mask.sum(dim=1) > 0.5)
                if not active.any():
                    break
                safe_mask = mask.clone()
                safe_mask[~active] = 0
                safe_mask[~active, env.N] = 1.0
                row_sum = safe_mask.sum(dim=1, keepdim=True)
                zero_rows = row_sum.squeeze(1) <= 0
                if zero_rows.any():
                    safe_mask[zero_rows] = 0
                    safe_mask[zero_rows, env.N] = 1.0
                dyn_global = build_dyn_features(env)
                dyn_global = _match_dyn_feat_dim(dyn_global, target_dyn_dim)
                cand_phi = env.get_candidate_features(pad_mask)
                cand_phi = _match_cand_phi_dim(cand_phi, target_cand_dim)
                with bf16_autocast(use_bf16):
                    logits = model.decode_step(
                        nodes_enc,
                        node_feats,
                        depot_emb,
                        graph_emb,
                        dyn_global,
                        prev_action,
                        safe_mask,
                        latent=latent,
                        k_precomputed=k_pre,
                        cand_phi=cand_phi,
                    )
                if greedy:
                    logits = logits.masked_fill(safe_mask < 0.5, -1e9)
                    actions = torch.argmax(logits, dim=1)
                else:
                    raw_probs = F.softmax(logits, dim=1)
                    probs = _normalize_probs(raw_probs, safe_mask)
                    dist_full = Categorical(probs.float())
                    actions = dist_full.sample()
                safe_actions = actions.clone()
                safe_actions[~active] = env.N
                env.step(safe_actions)
                prev_action = safe_actions

                fully_done = env.compute_terminal_mask(pad_mask)
                if fully_done.any():
                    env.finished[fully_done] = True

        routes = _collect_final_routes(env, 0)

        total_dist = 0.0
        feas_all = True
        served = 0
        for r in routes:
            d, feas, _ = route_cost_and_feasible(inst, r)
            total_dist += d
            feas_all = feas_all and feas
            served += len(r)
        unserved = max(0, len(inst.customers) - served)
        veh = len(routes)
        infeasible = (not feas_all)
        return routes, total_dist, veh, unserved, infeasible

    best_key = None
    best_routes: List[List[int]] = []
    best_dist = 0.0
    for _ in range(max(1, latent_multi_k)):
        r_k, dist_k, veh_k, unserved_k, infeasible_k = decode_once()
        key_k = lexicographic_key(infeasible_k, unserved_k, veh_k, dist_k)
        if (best_key is None) or (key_k < best_key):
            best_key = key_k
            best_routes = r_k
            best_dist = dist_k

    if was_training:
        model.train()

    return best_routes, best_dist


def neural_construct(
    instances: List[Instance],
    subset_ids=None,
    greedy: bool = True,
    device: str = "cuda",
    model: Optional[nn.Module] = None,
    apply_feasibility_mask: bool = False,
    latent_dim: int = 0,
    latent_override: Optional[torch.Tensor] = None,
    use_bf16: bool = False,
) -> Tuple[List[List[List[int]]], List[float]]:
    if not instances:
        return [], []

    all_node_feats, all_depot_feats, all_pad_masks, all_env_data = pad_instances(instances)

    if model is None:
        cand_phi_dim = int(getattr(train_defaults, "cand_phi_dim", get_cand_phi_feature_dim())) if bool(getattr(train_defaults, "use_dynamic_key_aug", False)) else 0
        cand_phi_hidden_dim = int(getattr(train_defaults, "cand_phi_hidden_dim", 0))
        use_raw_feature_bias = bool(getattr(train_defaults, "use_raw_feature_bias", False))
        embed_dim, n_heads, n_layers, ff_dim = _resolve_model_arch_defaults()
        model = AttentionVRPTW(
            node_dim=int(all_node_feats.size(-1)),
            dyn_dim=_resolve_model_dyn_dim(),
            embed_dim=embed_dim,
            n_heads=n_heads,
            n_layers=n_layers,
            ff_dim=ff_dim,
            latent_dim=latent_dim,
            cand_phi_dim=cand_phi_dim,
            cand_phi_hidden_dim=cand_phi_hidden_dim,
            use_raw_feature_bias=use_raw_feature_bias,
        ).to(device)
    target_node_dim = int(getattr(getattr(model, "node_enc", None), "in_features", all_node_feats.size(-1)))
    all_node_feats, all_depot_feats = _match_model_node_feat_dim(all_node_feats, all_depot_feats, target_node_dim)
    target_dyn_dim = int(getattr(getattr(model, "dyn_proj", None), "in_features", get_dyn_feature_dim()))
    target_cand_dim = int(getattr(model, "cand_phi_dim", get_cand_phi_feature_dim()))
    was_training = model.training
    model.eval()
    use_bf16 = resolve_bf16_mode(device, use_bf16, scope="construct_batch", verbose=False)

    b_nf = all_node_feats.to(device)
    b_df = all_depot_feats.to(device)
    b_pm = all_pad_masks.to(device)
    b_env_data = {k: v.to(device) for k, v in all_env_data.items()}
    vehicle_limit, vehicle_max, vehicle_limit_source = _resolve_vehicle_limits_from_config(b_env_data, b_pm)

    env = BatchVRPTWEnv(
        b_env_data,
        device=device,
        track_routes=True,
        vehicle_limit=vehicle_limit,
        vehicle_max=vehicle_max,
        vehicle_limit_source=vehicle_limit_source,
    )
    with torch.no_grad():
        with bf16_autocast(use_bf16):
            nodes_enc, depot_emb, graph_emb, _enc_aux, _ = model.encode(b_nf, b_df, b_pm)
            k_pre = model.W_k(nodes_enc)

    B = len(instances)
    prev_action = torch.full((B,), env.N, dtype=torch.long, device=device)
    model_latent_dim = int(getattr(model, "latent_dim", 0))
    if model_latent_dim > 0:
        if latent_override is None:
            latent = torch.randn(B, model_latent_dim, device=device)
        else:
            latent = latent_override.to(device=device, dtype=b_nf.dtype)
            if latent.dim() == 1:
                latent = latent.unsqueeze(0)
            if latent.size(0) == 1 and B > 1:
                latent = latent.expand(B, -1)
            elif latent.size(0) != B:
                raise ValueError(f"latent_override batch mismatch: got {latent.size(0)}, expected {B}")
            if latent.size(1) != model_latent_dim:
                raise ValueError(
                    f"latent_override dim mismatch: got {latent.size(1)}, expected {model_latent_dim}"
                )
    else:
        latent = None

    with torch.no_grad():
        for _ in range(env.N * 2 + 100):
            mask = env.get_mask(b_pm)
            active = (~env.finished) & (mask.sum(dim=1) > 0.5)
            if not active.any():
                break

            safe_mask = mask.clone()
            safe_mask[~active] = 0
            safe_mask[~active, env.N] = 1.0
            row_sum = safe_mask.sum(dim=1, keepdim=True)
            zero_rows = row_sum.squeeze(1) <= 0
            if zero_rows.any():
                safe_mask[zero_rows] = 0
                safe_mask[zero_rows, env.N] = 1.0

            dyn_global = build_dyn_features(env)
            dyn_global = _match_dyn_feat_dim(dyn_global, target_dyn_dim)
            cand_phi = env.get_candidate_features(b_pm)
            cand_phi = _match_cand_phi_dim(cand_phi, target_cand_dim)
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
                    cand_phi=cand_phi,
                )

            if greedy:
                logits = logits.masked_fill(safe_mask < 0.5, -1e9)
                actions = torch.argmax(logits, dim=1)
            else:
                raw_probs = F.softmax(logits, dim=1)
                probs = _normalize_probs(raw_probs, safe_mask)
                dist = Categorical(probs.float())
                actions = dist.sample()

            safe_actions = actions.clone()
            safe_actions[~active] = env.N
            env.step(safe_actions)
            prev_action = safe_actions

            fully_done = env.compute_terminal_mask(b_pm)
            if fully_done.any():
                env.finished[fully_done] = True

    results_routes = []
    results_dists = []

    for i in range(B):
        inst = instances[i]
        final_routes = _collect_final_routes(env, i)

        
        total_dist = 0.0
        for r in final_routes:
            d, _, _ = route_cost_and_feasible(inst, r)
            total_dist += d

        
        results_routes.append(final_routes)
        results_dists.append(total_dist)

    if was_training:
        model.train()
    return results_routes, results_dists


























def _collect_final_routes(env, row: int) -> List[List[int]]:
    """
    Collect completed and open routes from a single-route env.
    Returns List[List[int]] (customer ids only, no depot).
    """
    final_routes_raw: List[List[int]] = []
    if hasattr(env, "routes") and len(env.routes) > row:
        final_routes_raw.extend(list(env.routes[row]))
    if hasattr(env, "current_route") and len(env.current_route) > row:
        curr = env.current_route[row]
        if isinstance(curr, list):
            if len(curr) > 0 and isinstance(curr[0], list):
                for r in curr:
                    if len(r) > 0:
                        final_routes_raw.append(list(r))
            elif len(curr) > 0:
                final_routes_raw.append(list(curr))
    final_routes: List[List[int]] = []
    for r in final_routes_raw:
        if len(r) == 0:
            continue
        final_routes.append([int(x) for x in r])
    return final_routes


def _aggregate_dyn_for_value(dyn_feats, vehicle_limit: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Convert dynamic features to value-head input [B,D].
    Accepts either [B,D] or [B,M,D].
    """
    if dyn_feats.dim() == 2:
        return torch.nan_to_num(dyn_feats, nan=0.0, posinf=0.0, neginf=0.0)
    if dyn_feats.dim() != 3:
        raise ValueError(f"dyn_feats must be [B,D] or [B,M,D], got {tuple(dyn_feats.shape)}")
    if vehicle_limit is None:
        return torch.nan_to_num(dyn_feats.mean(dim=1), nan=0.0, posinf=0.0, neginf=0.0)
    B, M, _ = dyn_feats.shape
    if vehicle_limit.numel() != B:
        raise ValueError(f"vehicle_limit expects B={B}, got {vehicle_limit.numel()}")
    slot_mask = (torch.arange(M, device=dyn_feats.device).unsqueeze(0) < vehicle_limit.to(dyn_feats.device).unsqueeze(1))
    den = slot_mask.sum(dim=1, keepdim=True).clamp_min(1).to(dtype=dyn_feats.dtype)
    num = (dyn_feats * slot_mask.unsqueeze(-1).to(dtype=dyn_feats.dtype)).sum(dim=1)
    out = num / den
    return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def _normalize_probs(probs: torch.Tensor, mask: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    
    probs = probs * mask
    probs_sum = probs.sum(dim=1, keepdim=True)
    probs = probs / (probs_sum + eps)

    
    invalid_rows = (~torch.isfinite(probs)).any(dim=1) | (probs_sum.squeeze(1) <= eps)
    if invalid_rows.any():
        safe_mask = mask[invalid_rows]
        safe_den = safe_mask.sum(dim=1, keepdim=True)
        
        zero_den = safe_den.squeeze(1) <= eps
        if zero_den.any():
            safe_mask = safe_mask.clone()
            safe_mask[zero_den, -1] = 1.0
            safe_den = safe_mask.sum(dim=1, keepdim=True)
        safe_den = safe_den.clamp_min(1.0)
        fill = safe_mask / safe_den
        probs[invalid_rows] = fill

    probs = probs.clamp(min=0.0, max=1.0)
    probs = probs / (probs.sum(dim=1, keepdim=True) + eps)
    return probs


def _compute_reward(costs_tensor: torch.Tensor) -> torch.Tensor:
    med_cost = float(costs_tensor.median().item())
    reward_scale = max(100.0, med_cost)
    reward = -costs_tensor / reward_scale
    return reward


class EMAState:
    def __init__(self, beta: float = 0.9):
        self.beta = beta
        self.mean: Optional[torch.Tensor] = None
        self.std: Optional[torch.Tensor] = None

    def update(self, batch_mean: torch.Tensor, batch_std: torch.Tensor, eps: float = 1e-8):
        if self.mean is None:
            self.mean = batch_mean
            self.std = batch_std
        else:
            self.mean = self.beta * self.mean + (1 - self.beta) * batch_mean
            self.std = self.beta * self.std + (1 - self.beta) * batch_std
        return self.mean, self.std


def compute_advantage(
    reward: torch.Tensor,
    group_ids: torch.Tensor,
    ema_state: EMAState,
    eps: float = 1e-8,
):
    uniq, inverse = torch.unique(group_ids, return_inverse=True)
    group_count = torch.zeros(len(uniq), device=reward.device, dtype=reward.dtype)
    group_sum = torch.zeros(len(uniq), device=reward.device, dtype=reward.dtype)
    ones = torch.ones_like(reward, dtype=reward.dtype)
    group_count.scatter_add_(0, inverse, ones)
    group_sum.scatter_add_(0, inverse, reward)
    # Leave-one-out group baseline:
    # for sample i in a group, baseline excludes reward_i itself.
    group_count_i = group_count[inverse]
    group_sum_i = group_sum[inverse]
    has_peer = group_count_i > 1.0
    loo_group_mean = torch.where(
        has_peer,
        (group_sum_i - reward) / (group_count_i - 1.0),
        reward,
    )
    a0 = reward - loo_group_mean
    raw_adv_std = float(a0.std(unbiased=False).item())
    batch_mean = a0.mean()
    batch_std = a0.std(unbiased=False) + eps
    a_norm = (a0 - batch_mean) / batch_std

    ema_mean, ema_std = ema_state.update(batch_mean.detach(), batch_std.detach())
    adv = (a_norm - ema_mean) / (ema_std + eps)

    stats = {
        "adv_mode": "group_mean_only",
        "var_a1": float(a0.var(unbiased=False).item()),
        "loo_singleton_ratio": float((~has_peer).float().mean().item()),
        "batch_mean_a0": float(batch_mean.item()),
        "batch_std_a0": float(batch_std.item()),
        "ema_mean": float(ema_mean.item()),
        "ema_std": float(ema_std.item()),
        "adv_raw_std": raw_adv_std,
    }

    return adv, ema_state, stats


def _apply_feasibility_mask(
    env: "BatchVRPTWEnv",
    safe_mask: torch.Tensor,
    instances: List[Instance],
    batch_indices: torch.Tensor,
) -> torch.Tensor:
    
    
    if safe_mask.numel() == 0:
        return safe_mask

    refined = safe_mask.clone()
    row_cap = 16  
    cand_cap = 24  

    row_items = list(enumerate(batch_indices.tolist()))
    if len(row_items) > row_cap:
        row_items = random.sample(row_items, row_cap)

    for row, b_idx in row_items:
        if env.finished[row]:
            continue
        inst = instances[b_idx]
        cand = torch.nonzero(refined[row, : env.N] > 0.5, as_tuple=False).squeeze(1)
        if cand.numel() == 0:
            continue
        cand_list = cand.tolist()
        if len(cand_list) > cand_cap:
            cand_list = random.sample(cand_list, cand_cap)
        curr_route = env.current_route[row]
        for cid in cand_list:
            route_prefix = curr_route + [int(cid)]
            _, feas, _ = route_cost_and_feasible(inst, route_prefix)
            if not feas:
                refined[row, cid] = 0.0

    row_sum = refined.sum(dim=1, keepdim=True)
    zero_rows = row_sum.squeeze(1) <= 0
    if zero_rows.any():
        refined[zero_rows] = 0.0
        refined[zero_rows, env.N] = 1.0
    return refined


def self_test_pomo_instance_norm_equivalence_no_pad(device: Optional[str] = None):
    """
    Verify the no-pad path matches original POMO transpose+InstanceNorm1d behavior.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    torch.manual_seed(0)
    B, N, E = 4, 13, 32
    x = torch.randn(B, N, E, device=device, dtype=torch.float32)

    norm = POMOInstanceNorm(d_model=E, eps=1e-5, batch_first=True).to(device=device, dtype=x.dtype)
    ref_norm = nn.InstanceNorm1d(E, affine=True, track_running_stats=False, eps=1e-5).to(device=device, dtype=x.dtype)

    with torch.no_grad():
        ref_norm.weight.copy_(norm.weight)
        ref_norm.bias.copy_(norm.bias)

        out = norm(x, padding_mask=None)
        out_ref = ref_norm(x.transpose(1, 2)).transpose(1, 2)
        max_abs = float((out - out_ref).abs().max().item())
        assert max_abs < 1e-6, f"POMOInstanceNorm no-pad mismatch, max_abs_diff={max_abs:.6e}"

        norm_t = POMOInstanceNorm(d_model=E, eps=1e-5, batch_first=False).to(device=device, dtype=x.dtype)
        norm_t.weight.copy_(norm.weight)
        norm_t.bias.copy_(norm.bias)
        out_t = norm_t(x.transpose(0, 1), padding_mask=None).transpose(0, 1)
        max_abs_t = float((out_t - out_ref).abs().max().item())
        assert max_abs_t < 1e-6, f"POMOInstanceNorm batch_first=False mismatch, max_abs_diff={max_abs_t:.6e}"


def self_test_encoder_norm_checkpoint_compat(device: Optional[str] = None):
    """
    Ensure encoder norm keys stay strict-load compatible for legacy checkpoints.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    torch.manual_seed(0)
    node_dim = get_planned_node_feature_dim()
    model_src = AttentionVRPTW(node_dim=node_dim, embed_dim=32, n_heads=4, n_layers=2).to(device)
    state = model_src.state_dict()

    norm_suffixes = (".norm1.weight", ".norm1.bias", ".norm2.weight", ".norm2.bias")
    norm_keys = [k for k in state.keys() if k.startswith("encoder.layers.") and k.endswith(norm_suffixes)]
    assert norm_keys, "No encoder norm keys found for compatibility self-test."

    with torch.no_grad():
        for idx, key in enumerate(sorted(norm_keys)):
            numel = state[key].numel()
            shape = state[key].shape
            base = torch.linspace(-0.5, 0.5, steps=numel, device=state[key].device, dtype=state[key].dtype)
            if key.endswith("bias"):
                base = -base
            state[key] = (base + 0.01 * idx).view(shape)

    model_dst = AttentionVRPTW(node_dim=node_dim, embed_dim=32, n_heads=4, n_layers=2).to(device)
    model_dst.load_state_dict(state, strict=True)
    loaded_state = model_dst.state_dict()

    for key in norm_keys:
        if not torch.allclose(loaded_state[key], state[key], atol=0.0, rtol=0.0):
            raise AssertionError(f"Norm compatibility load mismatch at key: {key}")


def self_test_encode_padding_invariance(device: str = "cpu"):
    """
    Minimal sanity check:
    For the same valid customers, adding extra padding tokens should keep graph_emb nearly unchanged.
    """
    torch.manual_seed(0)
    node_dim = get_planned_node_feature_dim()
    model = AttentionVRPTW(node_dim=node_dim, embed_dim=64, n_heads=4, n_layers=2).to(device)
    model.eval()

    with torch.no_grad():
        valid_n = 7
        n_small = 12
        n_large = 20

        valid_nodes = torch.randn(1, valid_n, node_dim, device=device)
        pad_small = torch.randn(1, n_small - valid_n, node_dim, device=device) * 10.0
        pad_large = torch.randn(1, n_large - valid_n, node_dim, device=device) * 10.0
        depot = torch.randn(1, node_dim, device=device)

        nf_small = torch.cat([valid_nodes, pad_small], dim=1)
        nf_large = torch.cat([valid_nodes, pad_large], dim=1)
        pm_small = torch.cat(
            [torch.ones(1, valid_n, device=device), torch.zeros(1, n_small - valid_n, device=device)], dim=1
        )
        pm_large = torch.cat(
            [torch.ones(1, valid_n, device=device), torch.zeros(1, n_large - valid_n, device=device)], dim=1
        )

        _, _, g_small, _, _ = model.encode(nf_small, depot, pm_small)
        _, _, g_large, _, _ = model.encode(nf_large, depot, pm_large)

        max_abs = float((g_small - g_large).abs().max().item())
        assert max_abs < 1e-5, f"padding invariance check failed, max_abs_diff={max_abs:.6e}"


def self_test_gated_encoder_attention(device: Optional[str] = None):
    """
    Minimal sanity check for encoder gated self-attention:
    1) use_gated_attn=False: shape is correct and output has no NaN.
    2) use_gated_attn=True with alpha_attn_gate=1.0: shape is correct, output has no NaN,
       and gate mean is in (0, 1).
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    def _run_once(dtype: torch.dtype, use_gated_attn: bool, alpha_attn_gate: float):
        layer = GatedTransformerEncoderLayer(
            d_model=32,
            nhead=4,
            dim_feedforward=64,
            dropout=0.0,
            batch_first=True,
            use_gated_attn=use_gated_attn,
            gated_attn_init_bias=2.0,
        ).to(device=device, dtype=dtype)
        setattr(layer, "alpha_attn_gate", float(alpha_attn_gate))
        layer.eval()

        src = torch.randn(3, 11, 32, device=device, dtype=dtype)
        with torch.no_grad():
            out, _aux_loss, stats = layer(src, src_mask=None, src_key_padding_mask=None)
        assert out.shape == src.shape, "shape mismatch in encoder layer output"
        assert out.device == src.device, "device mismatch in encoder layer output"
        assert out.dtype == src.dtype, "dtype mismatch in encoder layer output"
        assert torch.isfinite(out).all(), "non-finite values found in encoder layer output"
        return stats

    _run_once(torch.float32, use_gated_attn=False, alpha_attn_gate=0.0)
    stats_gated = _run_once(torch.float32, use_gated_attn=True, alpha_attn_gate=1.0)
    gate_mean = float(stats_gated["gated_attn_mean"].detach().float().item())
    assert 0.0 < gate_mean < 1.0, f"gated_attn_mean is out of range: {gate_mean}"

    # Low-precision path check on CUDA where fp16/bf16 attention kernels are expected.
    if torch.cuda.is_available():
        _run_once(torch.float16, use_gated_attn=True, alpha_attn_gate=1.0)
        _run_once(torch.bfloat16, use_gated_attn=True, alpha_attn_gate=1.0)


def self_test_gate_schedule_wiring(device: Optional[str] = None):
    """
    Ensure schedule wiring updates both decoder and encoder gate controls.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    torch.manual_seed(0)
    node_dim = get_planned_node_feature_dim()
    model = AttentionVRPTW(node_dim=node_dim, embed_dim=32, n_heads=4, n_layers=2).to(device)
    model.eval()

    alpha = _set_model_gate_alphas(model, 0.37)
    assert abs(float(getattr(model, "alpha_gate", -1.0)) - alpha) < 1e-12
    assert abs(float(getattr(model, "alpha_attn_gate", -1.0)) - alpha) < 1e-12

    with torch.no_grad():
        nf = torch.randn(2, 9, node_dim, device=device)
        df = torch.randn(2, node_dim, device=device)
        pm = torch.ones(2, 9, device=device)
        model.encode(nf, df, pm)

    for idx, enc_layer in enumerate(model.encoder.layers):
        layer_alpha = float(getattr(enc_layer, "alpha_attn_gate", -1.0))
        assert abs(layer_alpha - alpha) < 1e-12, f"encoder layer {idx} alpha_attn_gate mismatch: {layer_alpha}"


def self_test_depth_attn_residual_behavior(device: Optional[str] = None):
    """
    Check that depth residuals:
    - reduce to a mean when the depth query is zeroed
    - ignore extra padding tokens in the history
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    torch.manual_seed(0)
    B, N, E = 3, 9, 32
    history = [torch.randn(B, N, E, device=device) for _ in range(4)]
    depth_res = DepthAttnResidual(d_model=E, layer_norm_eps=1e-5, batch_first=True).to(device)
    depth_res.eval()

    with torch.no_grad():
        depth_res.depth_query.zero_()
        depth_res.depth_scale.fill_(1.0)
        out = depth_res(history, padding_mask=None)
        expected = torch.stack(history, dim=1).mean(dim=1)
        max_abs = float((out - expected).abs().max().item())
        assert max_abs < 1e-6, f"DepthAttnResidual mean-path mismatch, max_abs_diff={max_abs:.6e}"

        valid_n = 5
        n_small = 8
        n_large = 12
        mask_small = torch.cat(
            [torch.zeros(B, valid_n, device=device), torch.ones(B, n_small - valid_n, device=device)], dim=1
        ).to(dtype=torch.bool)
        mask_large = torch.cat(
            [torch.zeros(B, valid_n, device=device), torch.ones(B, n_large - valid_n, device=device)], dim=1
        ).to(dtype=torch.bool)
        hist_small = [torch.randn(B, n_small, E, device=device) for _ in range(3)]
        hist_large = [torch.cat([h[:, :valid_n], torch.randn(B, n_large - valid_n, E, device=device)], dim=1) for h in hist_small]
        out_small = depth_res(hist_small, padding_mask=mask_small)
        out_large = depth_res(hist_large, padding_mask=mask_large)
        pad_free_diff = float((out_small[:, :valid_n] - out_large[:, :valid_n]).abs().max().item())
        assert pad_free_diff < 1e-6, f"DepthAttnResidual padding sensitivity, max_abs_diff={pad_free_diff:.6e}"


def _smoke_test_head_gated_mha(device: Optional[str] = None):
    """
    Smoke test for head-wise SDPA output gating:
    - alpha_attn_gate=1.0 path
    - bool key_padding_mask semantics (True means disallow in MHA API)
    - bool attn_mask semantics (True means disallow in MHA API)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    torch.manual_seed(0)
    B, L, E = 3, 12, 32
    H = 4
    x = torch.randn(B, L, E, device=device)
    layer = GatedTransformerEncoderLayer(
        d_model=E,
        nhead=H,
        dim_feedforward=64,
        dropout=0.0,
        batch_first=True,
        use_gated_attn=True,
        gated_attn_init_bias=2.0,
    ).to(device)
    setattr(layer, "alpha_attn_gate", 1.0)
    layer.eval()

    with torch.no_grad():
        out, _aux, stats = layer(x, src_mask=None, src_key_padding_mask=None)
    assert out.shape == x.shape, "shape mismatch without masks"
    assert torch.isfinite(out).all(), "non-finite output without masks"
    gate_mean = float(stats["gated_attn_mean"].detach().float().item())
    assert 0.0 < gate_mean < 1.0, f"gate mean out of range: {gate_mean}"

    key_padding_mask = torch.zeros(B, L, dtype=torch.bool, device=device)
    key_padding_mask[:, -3:] = True
    with torch.no_grad():
        out_kpm, _aux_kpm, stats_kpm = layer(
            x,
            src_mask=None,
            src_key_padding_mask=key_padding_mask,
        )
    assert out_kpm.shape == x.shape, "shape mismatch with key_padding_mask"
    assert torch.isfinite(out_kpm).all(), "non-finite output with key_padding_mask"
    assert 0.0 < float(stats_kpm["gated_attn_mean"].detach().float().item()) < 1.0

    # MHA bool attn_mask semantics: True means disallow.
    attn_mask = torch.zeros(L, L, dtype=torch.bool, device=device)
    attn_mask[:, -1] = True
    with torch.no_grad():
        out_am, _aux_am, stats_am = layer(
            x,
            src_mask=attn_mask,
            src_key_padding_mask=None,
        )
    assert out_am.shape == x.shape, "shape mismatch with attn_mask"
    assert torch.isfinite(out_am).all(), "non-finite output with attn_mask"
    assert 0.0 < float(stats_am["gated_attn_mean"].detach().float().item()) < 1.0


