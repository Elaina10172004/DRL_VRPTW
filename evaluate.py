from typing import List

try:
    from .vrptw_data import Instance, route_cost_and_feasible
except Exception:
    from vrptw_data import Instance, route_cost_and_feasible




def evaluate(inst: Instance, routes: List[List[int]]):
    non_empty_routes = [r for r in routes if len(r) > 0]
    total_dist = 0.0
    served = set()
    n = len(inst.customers)

    for r in non_empty_routes:
        for cid in r:
            if cid < 0 or cid >= n:
                return {"feasible": False}
            if cid in served:
                
                return {"feasible": False}
            served.add(cid)

        dist, feasible, _ = route_cost_and_feasible(inst, r)
        if not feasible:
            return {"feasible": False}
        total_dist += dist

    if len(served) != n:
        return {"feasible": False}

    return {
        "feasible": True,
        "vehicles": len(non_empty_routes),
        "total_distance": total_dist,
    }



def pretty_routes(routes: List[List[int]]):
    return " | ".join(["->".join(map(str, r)) for r in routes])
