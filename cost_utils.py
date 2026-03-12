

from dataclasses import dataclass
from typing import Optional, Sequence


@dataclass
class CostConfig:
    alpha: float = 250.0
    unserved_penalty: float = 500.0  
    infeasible_penalty: float = 5000.0  




def compute_cost(
    distance: float,
    routes: Sequence[Sequence[int]],
    *,
    alpha: Optional[float] = None,
    unserved: int = 0,
    unserved_penalty: Optional[float] = None,
    infeasible: bool = False,
    infeasible_penalty: Optional[float] = None,
    vehicles_override: int | None = None,
    config: Optional[CostConfig] = None,
) -> float:
    cfg = config or CostConfig()
    alpha_eff = cfg.alpha if alpha is None else alpha
    unserved_penalty_eff = cfg.unserved_penalty if unserved_penalty is None else unserved_penalty
    infeasible_penalty_eff = cfg.infeasible_penalty if infeasible_penalty is None else infeasible_penalty

    
    if vehicles_override is not None:
        vehicles = int(vehicles_override)
    else:
        non_empty_routes = [r for r in routes if len(r) > 0]
        vehicles = len(non_empty_routes)
    cost = distance + alpha_eff * vehicles
    if unserved > 0:
        cost += unserved_penalty_eff * unserved
    if infeasible:
        cost += infeasible_penalty_eff
    return cost
