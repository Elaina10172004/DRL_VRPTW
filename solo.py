import os
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Literal

import numpy as np

from vrptw_data import read_solomon

DistType = Literal["C", "R", "RC"]
TWType = Literal["1", "2"]


def save_solomon_batch(batch: List[Dict[str, Any]], out_dir: str, prefix: str, vehicle_num: int = 25):
    """
    Save a batch of generated instances in Solomon .txt layout.
    """
    os.makedirs(out_dir, exist_ok=True)
    for i, inst in enumerate(batch):
        name = f"{prefix}{i:03d}"
        txt = to_solomon_txt(inst, name=name, vehicle_num=inst.get("num_vehicles", vehicle_num))
        with open(os.path.join(out_dir, f"{name}.txt"), "w", encoding="utf-8") as f:
            f.write(txt)


def to_solomon_txt(instance: Dict[str, Any], name: str = None, vehicle_num: int = 25) -> str:
    """
    Convert generated instance dict to Solomon-like .txt content.
    """
    if name is None:
        name = instance.get("name", "DER_SOLOMON")

    cap = int(instance["capacity"])
    depot = instance["depot"]
    customers = instance["customers"]

    def fmt_row(row: Dict[str, Any]) -> str:
        return (
            f"{int(row['id']):>5}"
            f"{int(round(row['x'])):>8}"
            f"{int(round(row['y'])):>11}"
            f"{int(round(row['demand'])):>11}"
            f"{int(round(row['ready_time'])):>11}"
            f"{int(round(row['due_date'])):>11}"
            f"{int(round(row['service_time'])):>11}"
        )

    lines = [
        name.upper(),
        "",
        "VEHICLE",
        "NUMBER     CAPACITY",
        f"{int(vehicle_num):>4}{cap:>12}",
        "",
        "CUSTOMER",
        "CUST NO.  XCOORD.   YCOORD.    DEMAND   READY TIME  DUE DATE   SERVICE   TIME",
        "",
        fmt_row(depot),
    ]
    lines.extend(fmt_row(c) for c in customers)
    return "\n".join(lines)


def euclid(a, b) -> float:
    return float(np.linalg.norm(a - b))


def extract_stats(coords: np.ndarray, ready: np.ndarray, due: np.ndarray, depot: np.ndarray) -> Dict[str, float]:
    dmat = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
    np.fill_diagonal(dmat, np.inf)
    nn = dmat.min(axis=1)

    depot_dist = np.linalg.norm(coords - depot, axis=1)
    tw_width = due - ready
    return {
        "mean_nn_dist": float(nn.mean()),
        "std_nn_dist": float(nn.std()),
        "mean_depot_dist": float(depot_dist.mean()),
        "std_depot_dist": float(depot_dist.std()),
        "mean_tw_width": float(tw_width.mean()),
        "std_tw_width": float(tw_width.std()),
        "mean_ready": float(ready.mean()),
        "mean_due": float(due.mean()),
    }


@dataclass(frozen=True)
class SolomonFamilyProfile:
    family: str
    depot_xy: np.ndarray
    customer_xy: np.ndarray
    customer_demands: np.ndarray
    customer_service: np.ndarray
    horizon: float
    capacity: int
    vehicle_num: int
    tw_pool: np.ndarray


FAMILY_PATTERNS = {
    "C1": "c1*.txt",
    "C2": "c2*.txt",
    "R1": "r1*.txt",
    "R2": "r2*.txt",
    "RC1": "rc1*.txt",
    "RC2": "rc2*.txt",
}


def _spec_const(v: float) -> Dict[str, Any]:
    return {"kind": "const", "value": float(v)}


def _spec_beta(a: float, b: float, loc: float, scale: float, lo: float, hi: float) -> Dict[str, Any]:
    return {"kind": "beta", "params": (float(a), float(b), float(loc), float(scale)), "range": (float(lo), float(hi))}


def _spec_gamma(shape: float, loc: float, scale: float, lo: float, hi: float) -> Dict[str, Any]:
    return {"kind": "gamma", "params": (float(shape), float(loc), float(scale)), "range": (float(lo), float(hi))}


def _spec_weibull(shape: float, loc: float, scale: float, lo: float, hi: float) -> Dict[str, Any]:
    return {"kind": "weibull", "params": (float(shape), float(loc), float(scale)), "range": (float(lo), float(hi))}


def _spec_ge(shape: float, loc: float, scale: float, lo: float, hi: float) -> Dict[str, Any]:
    return {"kind": "ge", "params": (float(shape), float(loc), float(scale)), "range": (float(lo), float(hi))}


DER_DENSITIES = (0.25, 0.5, 0.75, 1.0)


# DER-Solomon 2024 Table 4.
DER_GROUP_RULES: Dict[str, List[Dict[str, Any]]] = {
    "C1": [
        {"weight": 3.0 / 8.0, "densities": DER_DENSITIES, "spec": _spec_beta(4.06, 2.09, 4.60, 176.48, 14.57, 90.97)},
        {"weight": 1.0 / 8.0, "spec": _spec_beta(3.66, 5.33, -8.89, 114.33, 11.26, 77.47)},
        {"weight": 1.0 / 8.0, "spec": _spec_gamma(1.52, 20.83, 37.58, 26.95, 95.78)},
        {"weight": 1.0 / 8.0, "spec": _spec_beta(3.73, 4.74, 48.20, 77.20, 52.17, 114.95)},
        {"weight": 1.0 / 4.0, "spec_choices": [_spec_const(90.0), _spec_const(180.0)]},
    ],
    "C2": [
        {"weight": 3.0 / 8.0, "densities": DER_DENSITIES, "spec": _spec_beta(2.26, 1.35, 40.38, 57.31, 44.37, 96.08)},
        {"weight": 1.0 / 8.0, "spec": _spec_beta(1.38, 4.63, 25.15, 87.81, 27.99, 84.47)},
        {"weight": 1.0 / 8.0, "spec": _spec_beta(2.95, 2.57, 102.16, 64.98, 109.87, 160.55)},
        {"weight": 1.0 / 8.0, "spec": _spec_ge(-0.15, 145.56, 186.39, 250.00, 788.59)},
        {"weight": 1.0 / 4.0, "spec_choices": [_spec_const(160.0), _spec_const(320.0)]},
    ],
    "R1": [
        {"weight": 1.0 / 4.0, "densities": DER_DENSITIES, "spec": _spec_const(5.0)},
        {"weight": 1.0 / 4.0, "densities": DER_DENSITIES, "spec": _spec_const(15.0)},
        {"weight": 1.0 / 12.0, "spec": _spec_const(30.0)},
        {"weight": 1.0 / 12.0, "spec": _spec_beta(3.68, 4.92, 22.81, 22.56, 26.86, 40.58)},
        {"weight": 1.0 / 12.0, "spec": _spec_const(45.0)},
        {"weight": 1.0 / 12.0, "spec": _spec_beta(2.76, 5.33, 31.95, 74.43, 35.89, 76.44)},
    ],
    "R2": [
        {"weight": 3.0 / 8.0, "densities": DER_DENSITIES, "spec": _spec_ge(0.37, 42.87, 37.01, 47.71, 94.44)},
        {"weight": 1.0 / 4.0, "densities": DER_DENSITIES, "spec": _spec_const(120.0)},
        {"weight": 1.0 / 8.0, "spec": _spec_const(240.0)},
        {"weight": 1.0 / 8.0, "spec": _spec_ge(-0.70, 78.89, 165.50, 120.08, 563.58)},
        {"weight": 1.0 / 8.0, "spec": _spec_beta(1.78, 9.71, 98.52, 869.87, 101.95, 441.12)},
    ],
    "RC1": [
        {"weight": 1.0 / 2.0, "densities": DER_DENSITIES, "spec": _spec_const(15.0)},
        {
            "weight": 1.0 / 8.0,
            "densities": DER_DENSITIES,
            "mix": [
                {"weight": 1.0 / 4.0, "spec": _spec_const(5.0)},
                {"weight": 1.0 / 4.0, "spec": _spec_const(60.0)},
                {"weight": 1.0 / 2.0, "spec": _spec_beta(1.94, 87.21, 8.89, 663.77, 22.52, 39.25)},
            ],
        },
        {"weight": 1.0 / 8.0, "spec": _spec_const(30.0)},
        {
            "weight": 1.0 / 8.0,
            "mix": [
                {"weight": 1.0 / 2.0, "spec": _spec_beta(2.88, 8.24, 19.28, 40.81, 22.52, 39.25)},
                {"weight": 1.0 / 2.0, "spec": _spec_beta(12.26, 10.26, 16.42, 78.39, 45.65, 72.19)},
            ],
        },
        {"weight": 1.0 / 8.0, "spec": _spec_beta(9.90, 5.49, -27.18, 129.57, 26.63, 95.89)},
    ],
    "RC2": [
        {"weight": 1.0 / 2.0, "densities": DER_DENSITIES, "spec": _spec_const(60.0)},
        {
            "weight": 1.0 / 8.0,
            "densities": DER_DENSITIES,
            "mix": [
                {"weight": 1.0 / 4.0, "spec": _spec_const(30.0)},
                {"weight": 1.0 / 4.0, "spec": _spec_const(240.0)},
                {"weight": 1.0 / 2.0, "spec": _spec_weibull(2.05, 92.65, 31.63, 45.15, 140.14)},
            ],
        },
        {"weight": 1.0 / 8.0, "spec": _spec_const(120.0)},
        {"weight": 1.0 / 8.0, "spec": _spec_beta(1.30, 2.27, 44.52, 359.15, 50.53, 346.38)},
        {"weight": 1.0 / 8.0, "spec": _spec_ge(0.22, 222.48, 34.73, 225.58, 309.35)},
    ],
}


def _family_key(dist_type: DistType, tw_type: TWType) -> str:
    return f"{dist_type}{tw_type}"


def _parse_vehicle_num(path: Path) -> int:
    try:
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines()[:20]:
            m = re.match(r"^\s*(\d+)\s+(\d+)\s*$", line)
            if m:
                return int(m.group(1))
    except Exception:
        pass
    return 25


@lru_cache(maxsize=32)
def _load_family_profile(family: str, data_dir: str = "solomon_data") -> SolomonFamilyProfile:
    pattern = FAMILY_PATTERNS.get(family)
    if pattern is None:
        raise ValueError(f"Unknown Solomon family: {family}")

    files = sorted(Path(data_dir).glob(pattern))
    if not files:
        raise FileNotFoundError(f"No Solomon files found for {family} under {data_dir}")

    base_xy = None
    base_demands = None
    base_service = None
    depot_xy = None
    horizon = None
    capacity = None
    vehicle_num = _parse_vehicle_num(files[0])
    tw_tables: list[np.ndarray] = []

    for path in files:
        inst = read_solomon(str(path))
        xy = np.array([[c.x, c.y] for c in inst.customers], dtype=float)
        demands = np.array([c.demand for c in inst.customers], dtype=int)
        service = np.array([c.service_time for c in inst.customers], dtype=float)
        tw = np.array([[c.ready_time, c.due_time] for c in inst.customers], dtype=float)

        if base_xy is None:
            base_xy = xy
            base_demands = demands
            base_service = service
            depot_xy = np.array([inst.depot.x, inst.depot.y], dtype=float)
            horizon = float(inst.depot.due_time)
            capacity = int(inst.capacity)
        else:
            if not np.allclose(base_xy, xy):
                raise ValueError(f"Family {family} has inconsistent coordinates in {path}")
            if not np.array_equal(base_demands, demands):
                raise ValueError(f"Family {family} has inconsistent demands in {path}")
            if not np.allclose(base_service, service):
                raise ValueError(f"Family {family} has inconsistent service times in {path}")

        tw_tables.append(tw)

    return SolomonFamilyProfile(
        family=family,
        depot_xy=depot_xy,
        customer_xy=base_xy,
        customer_demands=base_demands,
        customer_service=base_service,
        horizon=horizon,
        capacity=capacity,
        vehicle_num=vehicle_num,
        tw_pool=np.stack(tw_tables, axis=0),
    )


def _clamp_tw_pair(ready: float, due: float, depot_dist: float, service: float, horizon: float) -> tuple[float, float]:
    latest_due = horizon - service - depot_dist
    if latest_due < depot_dist:
        latest_due = depot_dist
    due = min(float(due), float(latest_due))
    if due < depot_dist:
        due = float(depot_dist)
    ready = max(0.0, min(float(ready), due - 1.0))
    if ready > due:
        ready = max(0.0, due - 1.0)
    return ready, due


def _normalize_weights(weights: np.ndarray) -> np.ndarray:
    w = np.asarray(weights, dtype=float)
    s = float(w.sum())
    if (not np.isfinite(s)) or s <= 0.0:
        return np.full_like(w, 1.0 / max(1, len(w)))
    return w / s


def _sample_index_by_weight(rng: np.random.Generator, items: List[Dict[str, Any]]) -> int:
    weights = np.array([float(x.get("weight", 0.0)) for x in items], dtype=float)
    return int(rng.choice(len(items), p=_normalize_weights(weights)))


def _allocate_counts(rng: np.random.Generator, total: int, weights: np.ndarray) -> np.ndarray:
    if total <= 0:
        return np.zeros(len(weights), dtype=int)
    w = _normalize_weights(weights)
    expected = w * float(total)
    counts = np.floor(expected).astype(int)
    rem = int(total - counts.sum())
    if rem > 0:
        frac = expected - counts.astype(float)
        picks = rng.choice(len(weights), size=rem, replace=True, p=_normalize_weights(frac))
        for k in picks:
            counts[int(k)] += 1
    return counts


def _sample_ge(rng: np.random.Generator, c: float, loc: float, scale: float, size: int) -> np.ndarray:
    # SciPy genextreme parameterization.
    u = rng.uniform(1e-12, 1.0 - 1e-12, size=size)
    if abs(c) < 1e-12:
        return loc - scale * np.log(-np.log(u))
    return loc + scale * (1.0 - np.power(-np.log(u), c)) / c


def _sample_raw_by_spec(rng: np.random.Generator, spec: Dict[str, Any], size: int) -> np.ndarray:
    kind = str(spec.get("kind", "")).lower()
    if kind == "const":
        return np.full(size, float(spec["value"]), dtype=float)

    params = spec.get("params", ())
    if kind == "beta":
        a, b, loc, scale = params
        return loc + scale * rng.beta(a, b, size=size)
    if kind == "gamma":
        shape, loc, scale = params
        return loc + rng.gamma(shape=shape, scale=scale, size=size)
    if kind == "weibull":
        shape, loc, scale = params
        return loc + scale * rng.weibull(shape, size=size)
    if kind == "ge":
        c, loc, scale = params
        return _sample_ge(rng, c, loc, scale, size=size)
    raise ValueError(f"Unsupported distribution kind: {kind}")


def _sample_from_spec(rng: np.random.Generator, spec: Dict[str, Any], size: int) -> np.ndarray:
    if size <= 0:
        return np.zeros(0, dtype=float)

    lo, hi = spec.get("range", (None, None))
    if lo is None and hi is None:
        return _sample_raw_by_spec(rng, spec, size=size)

    lo_v = -np.inf if lo is None else float(lo)
    hi_v = np.inf if hi is None else float(hi)
    out = np.empty(size, dtype=float)
    filled = 0
    rounds = 0
    while filled < size and rounds < 16:
        need = size - filled
        draw = _sample_raw_by_spec(rng, spec, size=max(need * 4, 64))
        draw = draw[(draw >= lo_v) & (draw <= hi_v)]
        if draw.size > 0:
            take = min(need, draw.size)
            out[filled : filled + take] = draw[:take]
            filled += take
        rounds += 1

    if filled < size:
        out[filled:] = np.clip(_sample_raw_by_spec(rng, spec, size=size - filled), lo_v, hi_v)
    return out


def _sample_density_for_group(rng: np.random.Generator, group: Dict[str, Any]) -> float:
    densities = tuple(group.get("densities", (1.0,)))
    return float(densities[int(rng.integers(0, len(densities)))]) if densities else 1.0


def _sample_half_widths_for_group(rng: np.random.Generator, group: Dict[str, Any], n: int) -> np.ndarray:
    if n <= 0:
        return np.zeros(0, dtype=float)

    mix = group.get("mix")
    if mix:
        weights = np.array([float(item.get("weight", 0.0)) for item in mix], dtype=float)
        counts = _allocate_counts(rng, n, weights)
        parts = []
        for item, cnt in zip(mix, counts):
            if cnt > 0:
                parts.append(_sample_from_spec(rng, item["spec"], int(cnt)))
        arr = np.concatenate(parts, axis=0) if parts else np.zeros(n, dtype=float)
        rng.shuffle(arr)
        return arr[:n]

    spec_choices = group.get("spec_choices")
    if spec_choices:
        pick = int(rng.integers(0, len(spec_choices)))
        return _sample_from_spec(rng, spec_choices[pick], n)

    spec = group.get("spec")
    if spec is None:
        raise ValueError("DER group must define one of: spec / spec_choices / mix")
    return _sample_from_spec(rng, spec, n)


def _route_distance(route: List[int], coords: np.ndarray, depot_xy: np.ndarray) -> float:
    if not route:
        return 0.0
    dist = euclid(depot_xy, coords[route[0]])
    for i in range(1, len(route)):
        dist += euclid(coords[route[i - 1]], coords[route[i]])
    dist += euclid(coords[route[-1]], depot_xy)
    return float(dist)


def _three_opt_candidates(route: List[int], i: int, j: int, k: int) -> List[List[int]]:
    a = route[:i]
    b = route[i:j]
    c = route[j:k]
    d = route[k:]
    return [
        a + b[::-1] + c + d,
        a + b + c[::-1] + d,
        a + b[::-1] + c[::-1] + d,
        a + c + b + d,
        a + c[::-1] + b + d,
        a + c + b[::-1] + d,
        a + c[::-1] + b[::-1] + d,
    ]


def _three_opt_improve_route(
    route: List[int],
    coords: np.ndarray,
    depot_xy: np.ndarray,
    rng: np.random.Generator,
    max_rounds: int = 3,
    max_samples_per_round: int = 256,
) -> List[int]:
    if len(route) < 4:
        return route

    best = list(route)
    best_len = _route_distance(best, coords, depot_xy)

    for _ in range(max_rounds):
        n = len(best)
        if n < 4:
            break
        improved = False
        samples = min(max_samples_per_round, max(64, n * n * n))
        for _ in range(samples):
            i = int(rng.integers(1, n - 2))
            j = int(rng.integers(i + 1, n - 1))
            k = int(rng.integers(j + 1, n + 1))
            for cand in _three_opt_candidates(best, i, j, k):
                cand_len = _route_distance(cand, coords, depot_xy)
                if cand_len + 1e-9 < best_len:
                    best = cand
                    best_len = cand_len
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break
    return best


def _build_initial_cvrp_routes(
    coords: np.ndarray,
    demands: np.ndarray,
    depot_xy: np.ndarray,
    capacity: int,
    rng: np.random.Generator,
) -> List[List[int]]:
    n = int(coords.shape[0])
    if n == 0:
        return []
    if np.any(demands > capacity):
        raise ValueError("Found customer demand larger than vehicle capacity")

    unserved = set(range(n))
    routes: List[List[int]] = []
    while unserved:
        route: List[int] = []
        load = 0
        cur = depot_xy
        while True:
            feasible = [i for i in unserved if load + int(demands[i]) <= capacity]
            if not feasible:
                break
            feas = np.array(feasible, dtype=int)
            d = np.linalg.norm(coords[feas] - cur, axis=1)
            order = np.argsort(d)
            top_k = min(8, len(feas))
            cand = feas[order[:top_k]]
            cand_d = d[order[:top_k]]
            w = 1.0 / (cand_d + 1e-6)
            w = w / w.sum()
            pick = int(rng.choice(cand, p=w))
            route.append(pick)
            unserved.remove(pick)
            load += int(demands[pick])
            cur = coords[pick]

        if not route:
            pick = int(next(iter(unserved)))
            route = [pick]
            unserved.remove(pick)
        routes.append(route)
    return routes


def _build_c_centers_from_cvrp_3opt(
    coords: np.ndarray,
    demands: np.ndarray,
    service: np.ndarray,
    depot_xy: np.ndarray,
    capacity: int,
    rng: np.random.Generator,
    n_starts: int = 3,
) -> np.ndarray:
    best_routes: List[List[int]] = []
    best_dist = float("inf")

    for _ in range(max(1, n_starts)):
        routes = _build_initial_cvrp_routes(coords, demands, depot_xy, capacity, rng)
        routes = [_three_opt_improve_route(r, coords, depot_xy, rng) for r in routes]
        total_dist = float(sum(_route_distance(r, coords, depot_xy) for r in routes))
        if total_dist < best_dist:
            best_dist = total_dist
            best_routes = routes

    centers = np.zeros(len(coords), dtype=float)
    for route in best_routes:
        t = 0.0
        prev = depot_xy
        for cid in route:
            t += euclid(prev, coords[cid])
            centers[cid] = t
            t += float(service[cid])
            prev = coords[cid]
    return centers


def _sample_der_centers(
    profile: SolomonFamilyProfile,
    family: str,
    n_customers: int,
    rng: np.random.Generator,
    *,
    horizon: float,
    capacity: int,
) -> np.ndarray:
    coords = profile.customer_xy[:n_customers]
    service = profile.customer_service[:n_customers]
    depot = profile.depot_xy
    d0 = np.linalg.norm(coords - depot, axis=1)
    lo = d0
    hi = np.maximum(lo + 1.0, horizon - d0 - service)

    if family.startswith("C"):
        centers = _build_c_centers_from_cvrp_3opt(
            coords=coords,
            demands=profile.customer_demands[:n_customers],
            service=service,
            depot_xy=depot,
            capacity=capacity,
            rng=rng,
        )
        return np.clip(centers, lo, hi)

    return rng.uniform(lo, hi)


def _generate_der_solomon_2024_instance(
    n_customers=100,
    dist_type: DistType = "R",
    tw_type: TWType = "1",
    seed=0,
    data_dir: str = "solomon_data",
) -> Dict[str, Any]:
    family = _family_key(dist_type, tw_type)
    rules = DER_GROUP_RULES.get(family)
    if not rules:
        raise ValueError(f"No DER-Solomon 2024 rules for family={family}")

    profile = _load_family_profile(family, data_dir)
    if int(n_customers) != 100:
        raise ValueError("Exact DER-Solomon 2024 reproduction requires n_customers=100.")

    max_customers = int(profile.customer_xy.shape[0])
    if n_customers > max_customers:
        raise ValueError(f"Family {family} only supports up to {max_customers} customers")

    rng = np.random.default_rng(seed)
    horizon_used = float(profile.horizon)
    capacity_used = int(profile.capacity)

    group_idx = _sample_index_by_weight(rng, rules)
    group = rules[group_idx]
    density = _sample_density_for_group(rng, group)

    n_constrained = int(round(float(density) * float(n_customers)))
    n_constrained = int(max(0, min(n_customers, n_constrained)))
    constrained_mask = np.zeros(n_customers, dtype=bool)
    if n_constrained == n_customers:
        constrained_mask[:] = True
    elif n_constrained > 0:
        pick = rng.choice(n_customers, size=n_constrained, replace=False)
        constrained_mask[np.asarray(pick, dtype=int)] = True

    coords = profile.customer_xy[:n_customers].copy()
    demands = profile.customer_demands[:n_customers].copy()
    service = profile.customer_service[:n_customers].copy()
    centers = _sample_der_centers(
        profile=profile,
        family=family,
        n_customers=n_customers,
        rng=rng,
        horizon=horizon_used,
        capacity=capacity_used,
    )

    ready = np.zeros(n_customers, dtype=float)
    due = np.full(n_customers, horizon_used, dtype=float)
    constrained_idx = np.nonzero(constrained_mask)[0]
    if constrained_idx.size > 0:
        half_width = _sample_half_widths_for_group(rng, group, int(constrained_idx.size))
        for j, cid in enumerate(constrained_idx):
            c = float(centers[int(cid)])
            h = float(max(0.5, half_width[j]))
            d0 = euclid(coords[int(cid)], profile.depot_xy)
            r_i, d_i = _clamp_tw_pair(c - h, c + h, d0, float(service[int(cid)]), horizon_used)
            ready[int(cid)] = r_i
            due[int(cid)] = d_i

    stats = extract_stats(coords, ready, due, profile.depot_xy)
    return {
        "depot": {
            "id": 0,
            "x": float(profile.depot_xy[0]),
            "y": float(profile.depot_xy[1]),
            "demand": 0,
            "ready_time": 0.0,
            "due_date": float(horizon_used),
            "service_time": 0,
        },
        "customers": [
            {
                "id": i + 1,
                "x": float(coords[i, 0]),
                "y": float(coords[i, 1]),
                "demand": int(demands[i]),
                "ready_time": float(ready[i]),
                "due_date": float(due[i]),
                "service_time": int(service[i]),
            }
            for i in range(n_customers)
        ],
        "capacity": int(capacity_used),
        "num_vehicles": int(profile.vehicle_num),
        "stats": stats,
        "score": 0.0,
        "family": family,
        "density": float(density),
        "group_index": int(group_idx),
        "method": "der_solomon_2024",
    }


def generate_der_solomon_instance(
    n_customers=100,
    dist_type: DistType = "R",
    tw_type: TWType = "1",
    seed=0,
    data_dir: str = "solomon_data",
) -> Dict[str, Any]:
    return _generate_der_solomon_2024_instance(
        n_customers=n_customers,
        dist_type=dist_type,
        tw_type=tw_type,
        seed=seed,
        data_dir=data_dir,
    )


def generate_der_solomon_batch(
    n_instances: int,
    dist_type: DistType,
    tw_type: TWType,
    seed0=0,
    **kwargs,
) -> List[Dict[str, Any]]:
    return [
        generate_der_solomon_instance(
            dist_type=dist_type,
            tw_type=tw_type,
            seed=seed0 + i,
            **kwargs,
        )
        for i in range(n_instances)
    ]


def main():
    n_instances_per_class = 1000
    common_kwargs = dict(n_customers=100)
    configs = [
        ("C", "1", "C1_train", "C1_", 0),
        ("C", "2", "C2_train", "C2_", 10_000),
        ("R", "1", "R1_train", "R1_", 20_000),
        ("R", "2", "R2_train", "R2_", 30_000),
        ("RC", "1", "RC1_train", "RC1_", 40_000),
        ("RC", "2", "RC2_train", "RC2_", 50_000),
    ]
    for dist_type, tw_type, out_dir, prefix, seed0 in configs:
        batch = generate_der_solomon_batch(
            n_instances=n_instances_per_class,
            dist_type=dist_type,
            tw_type=tw_type,
            seed0=seed0,
            **common_kwargs,
        )
        save_solomon_batch(batch, out_dir=out_dir, prefix=prefix, vehicle_num=25)


if __name__ == "__main__":
    main()
