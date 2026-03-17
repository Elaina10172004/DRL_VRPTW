


import argparse
from concurrent.futures import Future, ThreadPoolExecutor
import json
import math
import os
from pathlib import Path
import random
from typing import Callable

import torch

from config import train_defaults
from neural_policy import train_neural
from solo import generate_der_solomon_batch
from vrptw_data import Customer, Instance, read_solomon



def _resolve_train_latent_multi_k_default() -> int:
    return max(1, int(getattr(train_defaults, "latent_multi_k", 1)))


def load_all_solomon(folder: str, max_customers: int | None = None):
    
    instances = []
    folder_path = Path(folder)
    for path in sorted(folder_path.glob("*.txt")):
        inst = read_solomon(str(path))
        if max_customers is not None:
            inst.customers = inst.customers[:max_customers]
        instances.append(inst)
    return instances





def _dict_to_instance(data: dict, name: str) -> Instance:
    depot_raw = data["depot"]
    depot = Customer(
        idx=0,
        x=float(depot_raw["x"]),
        y=float(depot_raw["y"]),
        demand=int(depot_raw.get("demand", 0)),
        ready_time=float(depot_raw.get("ready_time", 0.0)),
        due_time=float(depot_raw.get("due_date", depot_raw.get("due_time", 0.0))),
        service_time=float(depot_raw.get("service_time", 0.0)),
    )
    customers = []
    for idx, c in enumerate(data["customers"]):
        customers.append(
            Customer(
                idx=idx,
                x=float(c["x"]),
                y=float(c["y"]),
                demand=int(c["demand"]),
                ready_time=float(c["ready_time"]),
                due_time=float(c.get("due_date", c.get("due_time", 0.0))),
                service_time=float(c["service_time"]),
            )
        )
    return Instance(name=name, capacity=int(data.get("capacity", 0)), depot=depot, customers=customers)





def build_generated_instances(
    count: int,
    *,
    dist_type: str,
    tw_type: str,
    seed0: int,
    n_customers: int,
    prefix: str = "GEN",
) -> list[Instance]:
    if count <= 0:
        return []
    raw_batch = generate_der_solomon_batch(
        n_instances=count,
        dist_type=dist_type,
        tw_type=tw_type,
        seed0=seed0,
        n_customers=n_customers,
    )
    instances = []
    for i, raw in enumerate(raw_batch):
        
        name = f"{prefix}_{dist_type}{tw_type}_{seed0 + i}"
        inst = _dict_to_instance(raw, name=name)
        
        inst.customers = inst.customers[:n_customers]
        instances.append(inst)
    return instances
class AsyncBatchInstanceProvider:
    def __init__(
        self,
        generator: Callable[[int, int, int], list[Instance]],
        epoch_instance_count: int,
        batch_size: int,
        prefetch_ahead: int = 1,
    ):
        self._generator = generator
        self._epoch_instance_count = int(epoch_instance_count)
        self._batch_size = int(batch_size)
        self._epoch_batches = max(1, math.ceil(self._epoch_instance_count / max(1, self._batch_size)))
        self._prefetch_ahead = max(1, int(prefetch_ahead))
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="der_gen")
        self._futures: dict[tuple[int, int, int], Future[list[Instance]]] = {}

    def _batch_count(self, batch_idx: int) -> int:
        start = int(batch_idx) * self._batch_size
        if start >= self._epoch_instance_count:
            return 0
        return min(self._batch_size, self._epoch_instance_count - start)

    def _next_key(self, epoch_idx: int, batch_idx: int) -> tuple[int, int, int]:
        next_epoch = int(epoch_idx)
        next_batch = int(batch_idx) + 1
        if next_batch >= self._epoch_batches:
            next_epoch += 1
            next_batch = 0
        return next_epoch, next_batch, self._batch_count(next_batch)

    def _submit(self, epoch_idx: int, batch_idx: int, expected_count: int):
        key = (int(epoch_idx), int(batch_idx), int(expected_count))
        if key not in self._futures:
            self._futures[key] = self._pool.submit(
                self._generator,
                int(epoch_idx),
                int(batch_idx),
                int(expected_count),
            )

    def _prefetch_from(self, epoch_idx: int, batch_idx: int):
        cur_epoch, cur_batch = int(epoch_idx), int(batch_idx)
        for _ in range(self._prefetch_ahead):
            next_epoch, next_batch, next_count = self._next_key(cur_epoch, cur_batch)
            if next_count <= 0:
                break
            self._submit(next_epoch, next_batch, next_count)
            cur_epoch, cur_batch = next_epoch, next_batch

    def prime_batch(self, epoch_idx: int, batch_idx: int, expected_count: int, instances: list[Instance]):
        key = (int(epoch_idx), int(batch_idx), int(expected_count))
        fut: Future[list[Instance]] = Future()
        fut.set_result(instances)
        self._futures[key] = fut
        self._prefetch_from(int(epoch_idx), int(batch_idx))

    def __call__(self, epoch_idx: int, batch_idx: int, expected_count: int) -> list[Instance]:
        epoch_idx = int(epoch_idx)
        batch_idx = int(batch_idx)
        expected_count = int(expected_count)
        key = (epoch_idx, batch_idx, expected_count)
        self._submit(epoch_idx, batch_idx, expected_count)
        fut = self._futures.pop(key)
        instances = fut.result()
        stale_keys = [k for k in list(self._futures.keys()) if (k[0] < epoch_idx) or (k[0] == epoch_idx and k[1] < batch_idx)]
        for old_key in stale_keys:
            self._futures.pop(old_key).cancel()
        self._prefetch_from(epoch_idx, batch_idx)
        return instances

    def shutdown(self):
        for fut in self._futures.values():
            fut.cancel()
        self._futures.clear()
        self._pool.shutdown(wait=False, cancel_futures=True)


def discover_train_shards(shards_dir: str | None, max_shards: int = 100) -> tuple[list[Path], int | None]:
    if shards_dir is None:
        return [], None
    if not str(shards_dir).strip():
        return [], None
    shard_root = Path(shards_dir)
    if (not shard_root.exists()) or (not shard_root.is_dir()):
        return [], None
    max_take = max(1, int(max_shards))
    files: list[Path] = []
    shard_size: int | None = None
    manifest_path = shard_root / "manifest.json"
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            try:
                shard_size = int(manifest.get("shard_size")) if manifest.get("shard_size") is not None else None
            except Exception:
                shard_size = None
            for item in manifest.get("files", []):
                rel = item.get("file")
                if not rel:
                    continue
                fp = shard_root / str(rel)
                if fp.exists():
                    files.append(fp)
                if len(files) >= max_take:
                    break
        except Exception:
            files = []
            shard_size = None
    if not files:
        files = sorted(shard_root.glob("shard_*.pt"))[:max_take]
    return files, shard_size


class ShardBatchInstanceProvider:
    def __init__(
        self,
        shard_files: list[Path],
        batch_size: int,
        max_customers: int | None = None,
        shuffle_each_epoch: bool = True,
    ):
        if not shard_files:
            raise ValueError("ShardBatchInstanceProvider requires non-empty shard_files.")
        self._shard_files = list(shard_files)
        self._batch_size = max(1, int(batch_size))
        self._max_customers = max_customers
        self._shuffle_each_epoch = bool(shuffle_each_epoch)
        self._cached_epoch: int | None = None
        self._cached_instances: list[Instance] = []
        self._cached_shard_idx: int | None = None

    def _load_epoch(self, epoch_idx: int):
        shard_idx = int(epoch_idx) % len(self._shard_files)
        shard_path = self._shard_files[shard_idx]
        loaded = torch.load(shard_path, map_location="cpu", weights_only=False)
        instances: list[Instance] = []
        if isinstance(loaded, list):
            for i, raw in enumerate(loaded):
                inst: Instance | None = None
                if isinstance(raw, Instance):
                    inst = raw
                elif isinstance(raw, dict):
                    name = str(raw.get("name") or f"SHARD_{shard_idx:05d}_{i}")
                    try:
                        inst = _dict_to_instance(raw, name=name)
                    except Exception:
                        inst = None
                if inst is None:
                    continue
                if self._max_customers is not None:
                    inst.customers = inst.customers[: int(self._max_customers)]
                instances.append(inst)
        if not instances:
            raise RuntimeError(f"[train_shards] empty/invalid shard: {shard_path}")
        if self._shuffle_each_epoch:
            rng = random.Random(10_000_019 * int(epoch_idx) + shard_idx)
            rng.shuffle(instances)
        self._cached_epoch = int(epoch_idx)
        self._cached_shard_idx = shard_idx
        self._cached_instances = instances
        print(
            f"[train_shards] epoch={int(epoch_idx)} -> {shard_path.name}, "
            f"instances={len(instances)}"
        )

    def __call__(self, epoch_idx: int, batch_idx: int, expected_count: int) -> list[Instance]:
        epoch_idx = int(epoch_idx)
        batch_idx = int(batch_idx)
        expected_count = int(expected_count)
        if expected_count <= 0:
            return []
        if self._cached_epoch != epoch_idx:
            self._load_epoch(epoch_idx)
        start = batch_idx * self._batch_size
        end = start + expected_count
        if start >= len(self._cached_instances):
            raise ValueError(
                f"[train_shards] batch start out of range: start={start}, "
                f"instances={len(self._cached_instances)}, epoch={epoch_idx}"
            )
        batch = self._cached_instances[start:end]
        if len(batch) != expected_count:
            raise ValueError(
                f"[train_shards] batch size mismatch: got {len(batch)}, expected {expected_count} "
                f"(epoch={epoch_idx}, batch={batch_idx}, shard_idx={self._cached_shard_idx})"
            )
        return batch


def train():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_folder", type=str, default=train_defaults.train_folder, help="Training folder; ignored when dynamic generation keeps fixed epoch size")
    ap.add_argument(
        "--gen_instances",
        type=int,
        default=train_defaults.gen_instances,
        help="Number of generated instances per epoch (0 disables generation)",
    )
    ap.add_argument(
        "--gen_dist_type",
        type=str,
        default=train_defaults.gen_dist_type,
        choices=["auto", "C", "R", "RC"],
        help="Generated spatial type: C/R/RC, auto for uniform mix",
    )
    ap.add_argument(
        "--gen_tw_type",
        type=str,
        default=train_defaults.gen_tw_type,
        choices=["auto", "1", "2"],
        help="Generated time-window type: 1/2, auto for uniform mix",
    )
    ap.add_argument("--gen_seed", type=int, default=train_defaults.gen_seed, help="Base random seed for generated instances")
    ap.add_argument(
        "--gen_customers",
        type=int,
        default=train_defaults.gen_customers,
        help="Generated customer count (defaults to --n_customers)",
    )
    ap.add_argument(
        "--gen_async",
        dest="gen_async",
        action="store_true",
        default=train_defaults.gen_async,
        help="Enable async prefetch (generate epoch t+1 while training epoch t)",
    )
    ap.add_argument(
        "--no_gen_async",
        dest="gen_async",
        action="store_false",
        help="Disable async prefetch and generate each epoch synchronously",
    )
    ap.add_argument("--train_bin", type=str, default=train_defaults.train_bin, help="Load training instances from binary file")
    ap.add_argument("--save_bin", type=str, default=train_defaults.save_bin, help="Save generated epoch-0 instances to binary file")
    ap.add_argument(
        "--train_shards_dir",
        type=str,
        default=getattr(train_defaults, "train_shards_dir", "pt/der640k_shards"),
        help="Preferred shard directory (manifest.json + shard_*.pt).",
    )
    ap.add_argument(
        "--train_shards_count",
        type=int,
        default=int(getattr(train_defaults, "train_shards_count", 100)),
        help="Max shard files to use from train_shards_dir.",
    )
    ap.add_argument(
        "--use_train_shards",
        dest="use_train_shards",
        action="store_true",
        default=bool(getattr(train_defaults, "use_train_shards", True)),
        help="Prefer loading training data from shards before train_bin/generation.",
    )
    ap.add_argument(
        "--no_train_shards",
        dest="use_train_shards",
        action="store_false",
        help="Disable shard-priority loading.",
    )
    ap.add_argument(
        "--train_shard_shuffle",
        dest="train_shard_shuffle",
        action="store_true",
        default=bool(getattr(train_defaults, "train_shard_shuffle", True)),
        help="Shuffle samples inside each shard per epoch.",
    )
    ap.add_argument(
        "--no_train_shard_shuffle",
        dest="train_shard_shuffle",
        action="store_false",
        help="Disable per-epoch shuffle within shard.",
    )
    ap.add_argument("--n_customers", type=int, default=train_defaults.n_customers, help="Max customers used per instance (truncate)")

    ap.add_argument("--epochs", type=int, default=train_defaults.epochs, help="Training epochs")
    ap.add_argument("--batch_size", type=int, default=train_defaults.batch_size, help="Training batch size")
    ap.add_argument(
        "--use_train_folder",
        action="store_true",
        default=train_defaults.use_train_folder,
        help="Include train_folder instances in training set (ignored when dynamic generation is active)",
    )

    ap.add_argument("--device", type=str, default=train_defaults.device)
    ap.add_argument(
        "--use_bf16",
        dest="use_bf16",
        action="store_true",
        default=bool(getattr(train_defaults, "use_bf16", True)),
        help="Enable CUDA BF16 autocast for training/inference (auto fallback when unsupported).",
    )
    ap.add_argument(
        "--no_bf16",
        dest="use_bf16",
        action="store_false",
        help="Disable BF16 autocast.",
    )
    ap.add_argument("--out", type=str, default=train_defaults.out)
    ap.add_argument("--log", type=str, default=train_defaults.log, help="Training log CSV path")
    ap.add_argument("--lr", type=float, default=train_defaults.lr, help="Initial learning rate")
    ap.add_argument(
        "--lr_schedule",
        type=str,
        default=train_defaults.lr_schedule,
        choices=["fixed", "cosine", "linear", "multistep"],
        help="LR decay schedule: fixed / cosine / linear / multistep",
    )
    ap.add_argument("--entropy_coef", type=float, default=train_defaults.entropy_coef, help="Entropy regularization coefficient")
    ap.add_argument("--resume", type=str, default=train_defaults.resume, help="Resume from existing pt/ckpt")
    ap.add_argument(
        "--latent_multi_k",
        type=int,
        default=_resolve_train_latent_multi_k_default(),
        help="latent 多样化并行 rollout 数",
    )
    ap.add_argument("--max_grad_norm", type=float, default=train_defaults.max_grad_norm, help="Gradient clipping threshold")
    ap.add_argument("--latent_dim", type=int, default=train_defaults.latent_dim, help="Decoder latent dimension (0 disables)")
    ap.add_argument(
        "--latent_noise_mode",
        type=str,
        default=str(getattr(train_defaults, "latent_noise_mode", "schedule")),
        choices=["schedule", "adaptive"],
        help="latent noise mode: schedule (existing) or adaptive(mean_cost-driven)",
    )
    ap.add_argument(
        "--latent_adaptive_sigma_init",
        type=float,
        default=float(getattr(train_defaults, "latent_adaptive_sigma_init", 0.0)),
        help="adaptive latent noise init sigma (<=0 means fallback to latent_max_sigma)",
    )
    ap.add_argument(
        "--latent_adaptive_sigma_min",
        type=float,
        default=float(getattr(train_defaults, "latent_adaptive_sigma_min", 0.1)),
        help="adaptive latent noise sigma lower bound",
    )
    ap.add_argument(
        "--latent_adaptive_sigma_max",
        type=float,
        default=float(getattr(train_defaults, "latent_adaptive_sigma_max", 1.0)),
        help="adaptive latent noise sigma upper bound",
    )
    ap.add_argument(
        "--latent_adaptive_improve_tol",
        type=float,
        default=float(getattr(train_defaults, "latent_adaptive_improve_tol", 0.002)),
        help="adaptive: relative mean_cost improvement threshold",
    )
    ap.add_argument(
        "--latent_adaptive_downscale",
        type=float,
        default=float(getattr(train_defaults, "latent_adaptive_downscale", 0.94)),
        help="adaptive: sigma multiplier on meaningful improvement",
    )
    ap.add_argument(
        "--latent_adaptive_upscale",
        type=float,
        default=float(getattr(train_defaults, "latent_adaptive_upscale", 1.06)),
        help="adaptive: sigma multiplier after patience epochs without improvement",
    )
    ap.add_argument(
        "--latent_adaptive_patience",
        type=int,
        default=int(getattr(train_defaults, "latent_adaptive_patience", 1)),
        help="adaptive: epochs without improvement before sigma upscale",
    )
    ap.add_argument(
        "--use_greedy_start_pool",
        dest="use_greedy_start_pool",
        action="store_true",
        default=bool(getattr(train_defaults, "use_greedy_start_pool", False)),
        help="Enable greedy start-pool first action assignment in training rollout.",
    )
    ap.add_argument(
        "--no_greedy_start_pool",
        dest="use_greedy_start_pool",
        action="store_false",
        help="Disable greedy start-pool first action assignment.",
    )
    ap.add_argument(
        "--use_dynamic_key_aug",
        dest="use_dynamic_key_aug",
        action="store_true",
        default=bool(getattr(train_defaults, "use_dynamic_key_aug", False)),
        help="Enable explicit dynamic candidate-feature head; default follows config.py.",
    )
    ap.add_argument(
        "--no_dynamic_key_aug",
        dest="use_dynamic_key_aug",
        action="store_false",
        help="Disable explicit dynamic candidate-feature head.",
    )
    ap.add_argument(
        "--use_raw_feature_bias",
        dest="use_raw_feature_bias",
        action="store_true",
        default=bool(getattr(train_defaults, "use_raw_feature_bias", False)),
        help="Enable explicit raw static-feature bias head; default follows config.py.",
    )
    ap.add_argument(
        "--no_raw_feature_bias",
        dest="use_raw_feature_bias",
        action="store_false",
        help="Disable explicit raw static-feature bias head.",
    )
    ap.add_argument(
        "--use_residual_gate",
        dest="use_residual_gate",
        action="store_true",
        default=bool(getattr(train_defaults, "use_residual_gate", True)),
        help="Enable encoder residual branch gates.",
    )
    ap.add_argument(
        "--no_residual_gate",
        dest="use_residual_gate",
        action="store_false",
        help="Disable encoder residual branch gates.",
    )
    ap.add_argument(
        "--residual_gate_init_bias",
        type=float,
        default=float(getattr(train_defaults, "residual_gate_init_bias", 2.0)),
        help="Initial bias for encoder residual branch gates.",
    )
    args = ap.parse_args()
    train_defaults.use_train_shards = bool(args.use_train_shards)
    train_defaults.train_shards_dir = args.train_shards_dir
    train_defaults.train_shards_count = int(args.train_shards_count)
    train_defaults.train_shard_shuffle = bool(args.train_shard_shuffle)
    train_defaults.use_bf16 = bool(args.use_bf16)
    train_defaults.latent_multi_k = max(1, int(args.latent_multi_k))
    train_defaults.latent_noise_mode = str(args.latent_noise_mode).strip().lower()
    train_defaults.latent_adaptive_sigma_init = float(args.latent_adaptive_sigma_init)
    train_defaults.latent_adaptive_sigma_min = float(max(0.0, args.latent_adaptive_sigma_min))
    train_defaults.latent_adaptive_sigma_max = float(max(train_defaults.latent_adaptive_sigma_min, args.latent_adaptive_sigma_max))
    train_defaults.latent_adaptive_improve_tol = float(max(0.0, args.latent_adaptive_improve_tol))
    train_defaults.latent_adaptive_downscale = float(max(1e-6, args.latent_adaptive_downscale))
    train_defaults.latent_adaptive_upscale = float(max(1e-6, args.latent_adaptive_upscale))
    train_defaults.latent_adaptive_patience = int(max(1, args.latent_adaptive_patience))
    train_defaults.use_greedy_start_pool = bool(args.use_greedy_start_pool)
    train_defaults.use_dynamic_key_aug = bool(args.use_dynamic_key_aug)
    train_defaults.use_raw_feature_bias = bool(args.use_raw_feature_bias)
    train_defaults.use_residual_gate = bool(args.use_residual_gate)
    train_defaults.residual_gate_init_bias = float(args.residual_gate_init_bias)

    training_base_instances: list[Instance] = []
    if args.use_train_folder and args.train_folder is not None:
        training_base_instances.extend(load_all_solomon(args.train_folder, max_customers=args.n_customers))

    total_gen = max(0, args.gen_instances)
    gen_n_customers = args.gen_customers if args.gen_customers is not None else args.n_customers
    train_instances: list[Instance] = []
    dynamic_batch_provider = None
    async_batch_provider: AsyncBatchInstanceProvider | None = None
    using_train_shards = False
    shard_files: list[Path] = []
    if bool(args.use_train_shards):
        shard_files, shard_size = discover_train_shards(args.train_shards_dir, max_shards=args.train_shards_count)
        if shard_files:
            using_train_shards = True
            dynamic_batch_provider = ShardBatchInstanceProvider(
                shard_files=shard_files,
                batch_size=max(1, int(args.batch_size)),
                max_customers=args.n_customers,
                shuffle_each_epoch=bool(args.train_shard_shuffle),
            )
            if shard_size is None or int(shard_size) <= 0:
                try:
                    sample_loaded = torch.load(shard_files[0], map_location="cpu", weights_only=False)
                    shard_size = len(sample_loaded) if isinstance(sample_loaded, list) else 0
                except Exception:
                    shard_size = 0
            total_gen = int(max(1, int(shard_size or 1)))
            print(
                f"[train] Prefer shards: dir={args.train_shards_dir}, use {len(shard_files)} shard files, "
                f"instances/epoch={total_gen}"
            )
        else:
            print(f"[train] No valid shards found under {args.train_shards_dir}, fallback to train_bin/generation.")

    dynamic_generation = (not using_train_shards) and (total_gen > 0 and not args.train_bin)

    if dynamic_generation and int(gen_n_customers) != 100:
        raise ValueError("DER-Solomon 2024 generation requires --gen_customers/--n_customers = 100.")

    if using_train_shards:
        if args.use_train_folder and training_base_instances:
            print("[train] Shard training enabled; --use_train_folder samples are ignored.")
    elif dynamic_generation:
        if args.use_train_folder and training_base_instances:
            print(
                "[train] Dynamic DER generation is enabled; to keep exactly --gen_instances per epoch, extra --use_train_folder samples are ignored."
            )
        use_uniform = args.gen_dist_type == "auto" or args.gen_tw_type == "auto"
        if use_uniform:
            combos = [("C", "1"), ("C", "2"), ("R", "1"), ("R", "2"), ("RC", "1"), ("RC", "2")]
        else:
            combos = [(args.gen_dist_type, args.gen_tw_type)]

        base = total_gen // len(combos)
        rem = total_gen % len(combos)
        combo_spans: list[tuple[int, str, str, int, int]] = []
        cursor = 0
        for idx, (dist, tw) in enumerate(combos):
            cnt = base + (1 if idx < rem else 0)
            if cnt <= 0:
                continue
            combo_spans.append((idx, dist, tw, cursor, cursor + cnt))
            cursor += cnt
        if cursor != total_gen:
            raise RuntimeError(f"combo split mismatch: total_gen={total_gen}, split={cursor}")

        epoch_batches = max(1, math.ceil(total_gen / max(1, args.batch_size)))

        def _expected_batch_count(batch_idx: int) -> int:
            start = int(batch_idx) * int(args.batch_size)
            if start >= total_gen:
                return 0
            return min(int(args.batch_size), total_gen - start)

        def _generate_epoch_batch_instances(epoch_idx: int, batch_idx: int, batch_count: int) -> list[Instance]:
            expected = _expected_batch_count(batch_idx)
            if expected <= 0:
                return []
            if int(batch_count) != expected:
                raise ValueError(f"batch_count mismatch: got {batch_count}, expected {expected} (batch_idx={batch_idx})")
            batch_start = int(batch_idx) * int(args.batch_size)
            batch_end = batch_start + expected
            epoch_seed_shift = int(epoch_idx) * 10_000_000
            batch_instances: list[Instance] = []
            for combo_idx, dist, tw, span_start, span_end in combo_spans:
                overlap_start = max(batch_start, span_start)
                overlap_end = min(batch_end, span_end)
                if overlap_start >= overlap_end:
                    continue
                cnt = overlap_end - overlap_start
                local_offset = overlap_start - span_start
                seed_base = args.gen_seed + epoch_seed_shift + combo_idx * 100000 + local_offset
                batch_instances.extend(
                    build_generated_instances(
                        count=cnt,
                        dist_type=dist,
                        tw_type=tw,
                        seed0=seed_base,
                        n_customers=gen_n_customers,
                        prefix=f"GEN_E{int(epoch_idx)}_B{int(batch_idx)}",
                    )
                )
            if len(batch_instances) != expected:
                raise RuntimeError(
                    f"generated {len(batch_instances)} instances, expected {expected} for epoch={epoch_idx}, batch={batch_idx}"
                )
            return batch_instances

        first_batch_count = _expected_batch_count(0)
        first_batch_instances = _generate_epoch_batch_instances(0, 0, first_batch_count)
        train_instances = []
        if args.gen_async:
            async_batch_provider = AsyncBatchInstanceProvider(
                _generate_epoch_batch_instances,
                epoch_instance_count=total_gen,
                batch_size=max(1, args.batch_size),
                prefetch_ahead=1,
            )
            async_batch_provider.prime_batch(0, 0, first_batch_count, first_batch_instances)
            dynamic_batch_provider = async_batch_provider
            print("[train] Async batch prefetch enabled: while training batch t, CPU prepares batch t+1.")
        else:
            dynamic_batch_provider = _generate_epoch_batch_instances

        if args.save_bin:
            try:
                epoch0_instances: list[Instance] = []
                for bidx in range(epoch_batches):
                    cnt = _expected_batch_count(bidx)
                    if cnt <= 0:
                        continue
                    if bidx == 0:
                        epoch0_instances.extend(first_batch_instances)
                    else:
                        epoch0_instances.extend(_generate_epoch_batch_instances(0, bidx, cnt))
                torch.save(epoch0_instances, args.save_bin)
                print(f"[save_bin] Saved epoch-0 {len(epoch0_instances)} instances to {args.save_bin}")
            except Exception as e:
                print(f"[save_bin] Save failed: {e}")
    else:
        if args.train_bin:
            try:
                loaded = torch.load(args.train_bin, map_location="cpu", weights_only=False)
                if isinstance(loaded, list):
                    if loaded and isinstance(loaded[0], Instance):
                        train_instances = loaded
                    else:
                        for i, raw in enumerate(loaded):
                            try:
                                name = getattr(raw, "name", None) or f"BIN_{i}"
                                inst = _dict_to_instance(raw, name=name)
                                train_instances.append(inst)
                            except Exception:
                                pass
                else:
                    print(f"[train_bin] Non-list format ignored: {type(loaded)}")
            except Exception as e:
                print(f"[train_bin] Load failed: {e}")

        train_instances.extend(training_base_instances)
        if not train_instances:
            raise ValueError(
                "No training data source is available. "
                "Please provide train_shards/train_bin/train_folder or enable dynamic generation via --gen_instances."
            )
        random.shuffle(train_instances)

    if not using_train_shards:
        for inst in train_instances:
            inst.customers = inst.customers[: args.n_customers]
        if not dynamic_generation:
            random.shuffle(train_instances)

    if using_train_shards:
        print(
            f"Training instances per epoch: {total_gen} "
            f"(shards mode, dir={args.train_shards_dir}, n_shards={len(shard_files)})"
        )
    elif dynamic_generation:
        gen_mode = "async prefetch" if args.gen_async else "sync generation"
        print(f"Training instances per epoch: {total_gen} (dynamic by-batch generation, {gen_mode})")
    else:
        print(f"Training instances: {len(train_instances)}")

    try:
        model, per_inst_best = train_neural(
            train_instances,
            epochs=args.epochs,
            batch_size=args.batch_size,
            device=args.device,
            use_bf16=bool(args.use_bf16),
            log_path=args.log,
            lr=args.lr,
            lr_schedule=args.lr_schedule,
            entropy_coef=args.entropy_coef,
            resume_path=args.resume,
            latent_multi_k=max(1, int(args.latent_multi_k)),
            max_grad_norm=args.max_grad_norm,
            latent_dim=max(0, args.latent_dim),
            batch_instance_provider=dynamic_batch_provider,
            batch_instance_count=total_gen if dynamic_batch_provider is not None else None,
        )
    finally:
        if async_batch_provider is not None:
            async_batch_provider.shutdown()

    
    model_path = os.path.splitext(args.out)[0] + ".pt"
    torch.save(model.state_dict(), model_path)
    saved_items = [f"Model: {model_path}"]
    saved_items.append(f"Train log: {args.log}")
    print("Saved files:")
    for item in saved_items:
        print(f"  - {item}")


def main():
    train()


if __name__ == "__main__":
    train()


