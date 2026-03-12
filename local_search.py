


import copy
import math
import random
import time
from typing import List, Tuple


random.seed(42)

from cost_utils import compute_cost, CostConfig
from vrptw_data import Instance, route_cost_and_feasible

_NUMBA_JIT_FN = None


def _get_route_cost_fn(inst: Instance):
    if not _NUMBA_AVAILABLE:
        return None
    try:
        import numpy as np

        n = len(inst.customers)
        coords = np.zeros((n + 1, 2), dtype=np.float64)
        demand = np.zeros(n + 1, dtype=np.float64)
        ready = np.zeros(n + 1, dtype=np.float64)
        due = np.zeros(n + 1, dtype=np.float64)
        service = np.zeros(n + 1, dtype=np.float64)

        coords[0, 0] = inst.depot.x
        coords[0, 1] = inst.depot.y
        ready[0] = inst.depot.ready_time
        due[0] = inst.depot.due_time

        for i, c in enumerate(inst.customers, start=1):
            coords[i, 0] = c.x
            coords[i, 1] = c.y
            demand[i] = c.demand
            ready[i] = c.ready_time
            due[i] = c.due_time
            service[i] = c.service_time

        cap = float(inst.capacity)
        depot_ready = float(inst.depot.ready_time)
        depot_due = float(inst.depot.due_time)

        @_numba.njit
        def _nb_cost(route_idx):
            dist = 0.0
            time = depot_ready
            cap_used = 0.0
            prev_idx = 0  
            for cid in route_idx:
                idx = cid + 1  
                cap_used += demand[idx]
                if cap_used > cap:
                    return 1e18, False
                dx = coords[prev_idx, 0] - coords[idx, 0]
                dy = coords[prev_idx, 1] - coords[idx, 1]
                travel = (dx * dx + dy * dy) ** 0.5
                dist += travel
                time += travel
                if time < ready[idx]:
                    time = ready[idx]
                if due[idx] > 0.0 and time > due[idx] + 1e-9:
                    return 1e18, False
                time += service[idx]
                prev_idx = idx

            dx = coords[prev_idx, 0] - coords[0, 0]
            dy = coords[prev_idx, 1] - coords[0, 1]
            back = (dx * dx + dy * dy) ** 0.5
            dist += back
            time += back
            if depot_due > 0.0 and time > depot_due + 1e-9:
                return 1e18, False
            return dist, True

        def _wrapper(route: List[int]):
            arr = np.array(route, dtype=np.int64)
            d, feas = _nb_cost(arr)
            return float(d), bool(feas)

        
        _wrapper([0])
        return _wrapper
    except Exception:
        return None
try:
    import numba as _numba
    _NUMBA_AVAILABLE = True
except Exception:
    _NUMBA_AVAILABLE = False



def repair_routes(
    inst: Instance, routes: List[List[int]], route_cost_fn=None
) -> Tuple[List[List[int]], float, float, int, int, bool]:
    """
    Light repair: deduplicate visit order, append missing customers greedily.
    """
    n_cust = len(inst.customers)
    seen = set()
    order = []
    for r in routes:
        for cid in r:
            if 0 <= cid < n_cust and cid not in seen:
                seen.add(cid)
                order.append(cid)
    for cid in range(n_cust):
        if cid not in seen:
            order.append(cid)

    repaired: List[List[int]] = []
    cur: List[int] = []
    for cid in order:
        trial = cur + [cid]
        if route_cost_fn is None:
            dist, feas, _ = route_cost_and_feasible(inst, trial)
        else:
            dist, feas = route_cost_fn(trial)
        if feas:
            cur = trial
            continue
        if cur:
            repaired.append(cur)
        cur = [cid]
    if cur:
        repaired.append(cur)

    cost, dist, veh, unserved, feas_all = evaluate_solution(inst, repaired, route_cost_fn=route_cost_fn)
    return repaired, cost, dist, veh, unserved, feas_all




def evaluate_solution(
    inst: Instance,
    routes: List[List[int]],
    alpha: float = 250.0,
    unserved_penalty: float = 500.0,
    infeasible_penalty: float = 5000.0,
    cost_cfg: CostConfig | None = None,
    route_cost_fn=None,
) -> Tuple[float, float, int, int, bool]:
    total_dist = 0.0
    feasible_all = True
    served_set = set()
    n_cust = len(inst.customers)

    for r in routes:
        if len(r) == 0:
            continue
        if any((cid < 0 or cid >= n_cust) for cid in r):
            feasible_all = False
            total_dist = 1e6
            break
        if any((cid in served_set) for cid in r):
            feasible_all = False
            total_dist = 1e6
            break

        if route_cost_fn is None:
            d, feas, _ = route_cost_and_feasible(inst, r)
        else:
            d, feas = route_cost_fn(r)
        if not feas or not math.isfinite(d):
            feasible_all = False
            
            d = 1e5 * max(1, len(r))
        total_dist += d
        for cid in r:
            served_set.add(cid)

    unserved = n_cust - len(served_set)
    unserved = max(0, unserved)
    non_empty_routes = [r for r in routes if len(r) > 0]
    num_veh = len(non_empty_routes)

    cost = compute_cost(
        total_dist,
        non_empty_routes,
        alpha=alpha,
        unserved=unserved,
        unserved_penalty=unserved_penalty,
        infeasible=not feasible_all,
        infeasible_penalty=infeasible_penalty,
        config=cost_cfg,
    )
    return cost, total_dist, num_veh, unserved, feasible_all


def _apply_move(routes: List[List[int]], move: tuple) -> List[List[int]]:
    kind = move[0]
    new_routes = [list(r) for r in routes]

    if kind == "2opt":
        _, ridx, i, j = move
        seg = new_routes[ridx][i : j + 1]
        new_routes[ridx][i : j + 1] = reversed(seg)
        return new_routes

    if kind == "relocate":
        _, src, dst, i, insert_pos = move
        if i >= len(new_routes[src]):
            return new_routes
        cust = new_routes[src][i]
        del new_routes[src][i]
        if dst == src and insert_pos > i:
            insert_pos -= 1
        insert_pos = max(0, min(insert_pos, len(new_routes[dst])))
        new_routes[dst].insert(insert_pos, cust)
        return new_routes

    if kind == "oropt":
        _, src, dst, start, length, insert_pos = move
        if start + length > len(new_routes[src]):
            return new_routes
        seg = new_routes[src][start : start + length]
        del new_routes[src][start : start + length]
        if dst == src and insert_pos > start:
            insert_pos -= length
        insert_pos = max(0, min(insert_pos, len(new_routes[dst])))
        new_routes[dst][insert_pos:insert_pos] = seg
        return new_routes

    if kind == "swap":
        _, r1, p1, r2, p2 = move
        if p1 >= len(new_routes[r1]) or p2 >= len(new_routes[r2]):
            return new_routes
        new_routes[r1][p1], new_routes[r2][p2] = new_routes[r2][p2], new_routes[r1][p1]
        return new_routes

    return new_routes


def _sample_moves(routes: List[List[int]], max_candidates: int = 24) -> List[tuple]:
    cand = []
    non_empty = [i for i, r in enumerate(routes) if len(r) > 0]
    total_nodes = sum(len(r) for r in routes)
    k = max(4, total_nodes // 10)  
    if max_candidates is None:
        max_candidates = max(24, k * 8)

    
    for _ in range(k):
        if not non_empty:
            break
        ridx = random.choice(non_empty)
        rlen = len(routes[ridx])
        if rlen < 3:
            continue
        i = random.randint(0, rlen - 3)
        j = random.randint(i + 1, rlen - 2)
        cand.append(("2opt", ridx, i, j))

    
    for _ in range(k):
        if not non_empty:
            break
        src = random.choice(non_empty)
        if len(routes[src]) == 0:
            continue
        dst = random.choice(non_empty) if len(non_empty) > 1 else src
        if src == dst and len(routes[src]) <= 1:
            continue
        i = random.randrange(len(routes[src]))
        insert_pos = random.randrange(len(routes[dst]) + 1)
        cand.append(("relocate", src, dst, i, insert_pos))

    
    for _ in range(k):
        if not non_empty:
            break
        src = random.choice(non_empty)
        rlen = len(routes[src])
        if rlen < 2:
            continue
        seg_len = 2 if rlen == 2 else random.choice([2, 3])
        if rlen < seg_len:
            continue
        start = random.randrange(0, rlen - seg_len + 1)
        dst = random.choice(non_empty) if len(non_empty) > 1 else src
        insert_pos = random.randrange(len(routes[dst]) + 1)
        cand.append(("oropt", src, dst, start, seg_len, insert_pos))

    
    for _ in range(k):
        if not non_empty:
            break
        r1 = random.choice(non_empty)
        r2 = random.choice(non_empty)
        if len(routes[r1]) == 0 or len(routes[r2]) == 0:
            continue
        p1 = random.randrange(len(routes[r1]))
        p2 = random.randrange(len(routes[r2]))
        if r1 == r2 and p1 == p2:
            continue
        cand.append(("swap", r1, p1, r2, p2))

    random.shuffle(cand)
    return cand[:max_candidates]


def two_opt_on_route(
    inst: Instance,
    routes: List[List[int]],
    route_idx: int,
    best_cost: float,
    best_feas: bool,
    max_try: int = 50,
    allow_repair: bool = False,
    route_cost_fn=None,
) -> Tuple[List[List[int]], float, bool]:
    if route_idx < 0 or route_idx >= len(routes):
        return routes, best_cost, best_feas
    route = routes[route_idx]
    n = len(route)
    if n < 3:
        return routes, best_cost, best_feas

    best_routes = routes
    cur_best = best_cost
    cur_feas = best_feas

    for _ in range(max_try):
        i = random.randint(0, n - 3)
        j = random.randint(i + 1, n - 2)
        new_routes = copy.deepcopy(best_routes)
        new_r = new_routes[route_idx]
        new_r[i : j + 1] = reversed(new_r[i : j + 1])

        new_cost, _, _, _, feas = evaluate_solution(inst, new_routes, route_cost_fn=route_cost_fn)
        if (feas or allow_repair) and new_cost + 1e-9 < cur_best:
            best_routes = new_routes
            cur_best = new_cost
            cur_feas = feas
            route = best_routes[route_idx]
            n = len(route)

    return best_routes, cur_best, cur_feas


def relocate_between_routes(
    inst: Instance,
    routes: List[List[int]],
    best_cost: float,
    best_feas: bool,
    max_try: int = 50,
    allow_repair: bool = False,
    route_cost_fn=None,
) -> Tuple[List[List[int]], float, bool]:
    if len(routes) < 2:
        return routes, best_cost, best_feas

    best_routes = routes
    cur_best = best_cost
    cur_feas = best_feas

    for _ in range(max_try):
        src_idx, dst_idx = random.sample(range(len(routes)), 2)
        src = best_routes[src_idx]
        dst = best_routes[dst_idx]
        if len(src) == 0:
            continue

        i = random.randrange(len(src))
        customer = src[i]

        new_routes = copy.deepcopy(best_routes)
        new_src = new_routes[src_idx]
        new_dst = new_routes[dst_idx]

        del new_src[i]
        insert_pos = random.randrange(len(new_dst) + 1)
        new_dst.insert(insert_pos, customer)

        new_cost, _, _, _, feas = evaluate_solution(inst, new_routes, route_cost_fn=route_cost_fn)
        if (feas or allow_repair) and new_cost + 1e-9 < cur_best:
            best_routes = new_routes
            cur_best = new_cost
            cur_feas = feas

    return best_routes, cur_best, cur_feas




def local_search_improve(
    inst: Instance,
    routes: List[List[int]],
    alpha: float = 250.0,
    max_iters: int = 200,
    allow_repair: bool = True,
    infeasible_penalty_ls: float = 1e4,
    cost_cfg: CostConfig | None = None,
    route_cost_fn=None,
) -> Tuple[List[List[int]], float, float, int, int, bool]:
    best_routes = [list(r) for r in routes]
    best_cost, best_dist, best_veh, best_unserved, best_feas = evaluate_solution(
        inst,
        best_routes,
        alpha=alpha,
        infeasible_penalty=infeasible_penalty_ls,
        cost_cfg=cost_cfg,
        route_cost_fn=route_cost_fn,
    )
    curr_routes = [list(r) for r in best_routes]
    curr_cost, curr_dist, curr_veh, curr_unserved, curr_feas = evaluate_solution(
        inst,
        curr_routes,
        alpha=alpha,
        infeasible_penalty=infeasible_penalty_ls,
        cost_cfg=cost_cfg,
        route_cost_fn=route_cost_fn,
    )

    if not best_feas and not allow_repair:
        return best_routes, best_cost, best_dist, best_veh, best_unserved, best_feas

    temp = max(50.0, 0.01 * abs(best_cost))
    temp_decay = 0.96

    consec_infeas = 0
    last_repair_it = -999

    for it in range(max_iters):
        cand_moves = _sample_moves(curr_routes, max_candidates=24)
        if not cand_moves:
            break

        best_cand = None  # (mv, new_routes, metrics)
        for mv in cand_moves:
            new_routes = _apply_move(curr_routes, mv)
            cost_m, dist_m, veh_m, unserved_m, feas_m = evaluate_solution(
                inst,
                new_routes,
                alpha=alpha,
                infeasible_penalty=infeasible_penalty_ls,
                cost_cfg=cost_cfg,
                route_cost_fn=route_cost_fn,
            )
            if not feas_m and not allow_repair:
                continue
            
            if best_cand is None:
                best_cand = (mv, new_routes, (cost_m, dist_m, veh_m, unserved_m, feas_m))
                continue
            _, _, (c_prev, _, _, _, feas_prev) = best_cand
            if (feas_m and not feas_prev) or (feas_m == feas_prev and cost_m + 1e-9 < c_prev):
                best_cand = (mv, new_routes, (cost_m, dist_m, veh_m, unserved_m, feas_m))

        if best_cand is None:
            break

        mv, new_routes, best_cand_metrics = best_cand
        cost_m, dist_m, veh_m, unserved_m, feas_m = best_cand_metrics
        delta_curr = cost_m - curr_cost

        
        accept = False
        if (feas_m or allow_repair) and delta_curr < -1e-9:
            accept = True
        else:
            if allow_repair and (feas_m or allow_repair):
                prob = math.exp(-max(0.0, delta_curr) / max(1e-6, temp))
                if random.random() < prob:
                    accept = True

        temp *= temp_decay

        if not accept:
            consec_infeas = consec_infeas + 1 if not curr_feas else 0
            continue

        # Accept candidate
        curr_routes = new_routes
        curr_cost, curr_dist, curr_veh, curr_unserved, curr_feas = best_cand_metrics
        consec_infeas = consec_infeas + 1 if not curr_feas else 0

        # Update global best
        if (curr_feas and not best_feas) or (curr_feas and curr_cost + 1e-9 < best_cost):
            best_routes = [list(r) for r in curr_routes]
            best_cost, best_dist, best_veh, best_unserved, best_feas = (
                curr_cost,
                curr_dist,
                curr_veh,
                curr_unserved,
                curr_feas,
            )

        need_repair = False
        if consec_infeas >= 30:
            need_repair = True
        if (it - last_repair_it) >= 50 and not curr_feas:
            need_repair = True

        if allow_repair and need_repair:
            repaired, cost_r, dist_r, veh_r, unserved_r, feas_r = repair_routes(
                inst, curr_routes, route_cost_fn=route_cost_fn
            )
            last_repair_it = it
            if feas_r and cost_r < curr_cost + 1e-9:
                curr_routes = repaired
                curr_cost, curr_dist, curr_veh, curr_unserved, curr_feas = (
                    cost_r,
                    dist_r,
                    veh_r,
                    unserved_r,
                    feas_r,
                )
                if (curr_feas and not best_feas) or (curr_feas and curr_cost + 1e-9 < best_cost):
                    best_routes = [list(r) for r in curr_routes]
                    best_cost, best_dist, best_veh, best_unserved, best_feas = (
                        curr_cost,
                        curr_dist,
                        curr_veh,
                        curr_unserved,
                        curr_feas,
                    )
                consec_infeas = 0

    best_cost, best_dist, best_veh, best_unserved, best_feas = evaluate_solution(
        inst,
        best_routes,
        alpha=alpha,
        infeasible_penalty=infeasible_penalty_ls,
        cost_cfg=cost_cfg,
        route_cost_fn=route_cost_fn,
    )
    if allow_repair and (not best_feas or best_unserved > 0):
        repaired, cost_r, dist_r, veh_r, unserved_r, feas_r = repair_routes(
            inst, best_routes, route_cost_fn=route_cost_fn
        )
        if feas_r and cost_r < best_cost + 1e-9:
            return repaired, cost_r, dist_r, veh_r, unserved_r, feas_r
    return best_routes, best_cost, best_dist, best_veh, best_unserved, best_feas




def time_budget_search(
    inst: Instance,
    candidates: List[List[List[int]]],
    budget_sec: float = 0.5,
    alpha: float = 250.0,
    base_iters: int = 30,
    allow_repair: bool = True,
    infeasible_penalty_ls: float = 5000.0,
    cost_cfg: CostConfig | None = None,
    parallel_workers: int = 1,
) -> Tuple[List[List[int]], float, float, int, int, bool]:
    if not candidates:
        return [], float("inf"), 0.0, 0, 0, False

    route_cost_fn = _get_route_cost_fn(inst)

    best_routes = None
    best_metric = (float("inf"), 0.0, 0, 0, False)

    scored = []
    for routes in candidates:
        cost, dist, veh, unserved, feas = evaluate_solution(
            inst,
            routes,
            alpha=alpha,
            infeasible_penalty=infeasible_penalty_ls,
            cost_cfg=cost_cfg,
            route_cost_fn=route_cost_fn,
        )
        if not feas and not allow_repair:
            continue
        scored.append((cost, dist, veh, unserved, feas, routes))

    if not scored:
        return [], float("inf"), 0.0, 0, 0, False

    scored.sort(key=lambda x: x[0])
    keep = min(16, len(scored))
    candidates = [r for _, _, _, _, _, r in scored[:keep]]

    for cost, dist, veh, unserved, feas, routes in scored[:keep]:
        if cost < best_metric[0]:
            best_metric = (cost, dist, veh, unserved, feas)
            best_routes = routes

    
    if parallel_workers and parallel_workers > 1:
        try:
            import multiprocessing as mp

            with mp.get_context("spawn").Pool(processes=parallel_workers) as pool:
                results = pool.map(
                    _ls_worker_args,
                    [
                        (
                            inst,
                            routes,
                            alpha,
                            base_iters,
                            allow_repair,
                            infeasible_penalty_ls,
                            cost_cfg,
                        )
                        for routes in candidates
                    ],
                )
            for improved, cost_i, dist_i, veh_i, unserved_i, feas_i in results:
                if cost_i < best_metric[0]:
                    best_metric = (cost_i, dist_i, veh_i, unserved_i, feas_i)
                    best_routes = improved
        except Exception as e:
            print(f"[警告] 并行本地搜索失败（{e}），退回到顺序执行")

    start = time.time()
    idx = 0
    n = len(candidates)

    while time.time() - start < budget_sec:
        cand_idx = idx % n
        routes = candidates[cand_idx]
        idx += 1

        remaining = budget_sec - (time.time() - start)
        if remaining <= 0:
            break
        iters = max(5, int(base_iters * (remaining / budget_sec)))

        improved, cost_i, dist_i, veh_i, unserved_i, feas_i = local_search_improve(
            inst,
            routes,
            alpha=alpha,
            max_iters=iters,
            allow_repair=allow_repair,
            infeasible_penalty_ls=infeasible_penalty_ls,
            cost_cfg=cost_cfg,
            route_cost_fn=route_cost_fn,
        )

        candidates[cand_idx] = improved

        if cost_i < best_metric[0]:
            best_metric = (cost_i, dist_i, veh_i, unserved_i, feas_i)
            best_routes = improved

    best_cost, best_dist, best_veh, best_unserved, best_feas = best_metric
    if allow_repair and (not best_feas or best_unserved > 0):
        repaired, cost_r, dist_r, veh_r, unserved_r, feas_r = repair_routes(
            inst, best_routes, route_cost_fn=route_cost_fn
        )
        if feas_r and cost_r < best_cost + 1e-9:
            return repaired, cost_r, dist_r, veh_r, unserved_r, feas_r
    return best_routes, best_cost, best_dist, best_veh, best_unserved, best_feas


# --- multiprocessing helper ---
def _ls_worker_args(args):
    inst, routes, alpha, base_iters, allow_repair, infeasible_penalty_ls, cost_cfg = args
    route_cost_fn = _get_route_cost_fn(inst)
    improved, cost_i, dist_i, veh_i, unserved_i, feas_i = local_search_improve(
        inst,
        routes,
        alpha=alpha,
        max_iters=base_iters,
        allow_repair=allow_repair,
        infeasible_penalty_ls=infeasible_penalty_ls,
        cost_cfg=cost_cfg,
        route_cost_fn=route_cost_fn,
    )
    return improved, cost_i, dist_i, veh_i, unserved_i, feas_i
