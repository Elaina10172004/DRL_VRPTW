import math
import re
from dataclasses import dataclass
from typing import List, Tuple





@dataclass
class Customer:
    idx: int
    x: float
    y: float
    demand: int
    ready_time: float
    due_time: float
    service_time: float


@dataclass
class Instance:
    name: str
    capacity: int
    depot: Customer
    customers: List[Customer]


def _parse_solomon_lines(lines: list) -> Instance:
    # Find name
    name = lines[0].strip()

    # Capacity line usually on line index ~4 (after headers)
    cap = None
    for i in range(min(15, len(lines))):
        if "CAPACITY" in lines[i].upper():
            tokens = re.findall(r"[-+]?\d+", lines[i + 1] if i + 1 < len(lines) else lines[i])
            if tokens:
                cap = int(tokens[-1])
            break

    # Find the table start: line containing 'CUST' and 'XCOORD'
    tstart = None
    for i, ln in enumerate(lines):
        u = ln.upper()
        if "CUST" in u and "XCOORD" in u and "YCOORD" in u:
            tstart = i + 1
            break
    if tstart is None:
        raise ValueError("Could not locate Solomon table header")

    rows = []
    for ln in lines[tstart:]:
        if not ln.strip():
            continue
        toks = ln.split()
        if len(toks) < 7:
            continue
        try:
            rid, x, y, q, r, d, s = toks[:7]
            rid = int(rid)
            x = float(x)
            y = float(y)
            q = int(float(q))
            r = float(r)
            d = float(d)
            s = float(s)
            rows.append((rid, x, y, q, r, d, s))
        except Exception:
            continue

    if not rows:
        raise ValueError("No rows parsed from Solomon file")

    # First row is depot
    rid, x, y, q, r, d, s = rows[0]
    depot = Customer(rid, x, y, q, r, d, s)
    customers = [Customer(rid, x, y, q, r, d, s) for rid, x, y, q, r, d, s in rows[1:]]

    if cap is None:
        cap = 9_999_999  # fallback, though Solomon should provide capacity

    return Instance(name=name, capacity=cap, depot=depot, customers=customers)




def read_solomon(path: str) -> Instance:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()
    return _parse_solomon_lines(lines)


def euclid(a: Customer, b: Customer) -> float:
    return math.hypot(a.x - b.x, a.y - b.y)




def route_cost_and_feasible(inst: Instance, route: list) -> Tuple[float, bool, list]:
    """
    route: list of customer indices (relative to customers list) meaning serve those in order, start/end at depot.
    Returns (distance, feasible, arrival_times). Enforces capacity, time windows, and service times.
    """
    cap_used = 0
    
    time = inst.depot.ready_time
    dist = 0.0
    arr = []
    prev = inst.depot

    for cid in route:
        c = inst.customers[cid]
        cap_used += c.demand
        if cap_used > inst.capacity:
            return float("inf"), False, []

        travel = euclid(prev, c)
        dist += travel
        time += travel

        if time < c.ready_time:
            time = c.ready_time
        if time > c.due_time + 1e-9:
            return float("inf"), False, []

        arr.append(time)
        time += c.service_time
        prev = c

    dist += euclid(prev, inst.depot)
    time += euclid(prev, inst.depot)

    if inst.depot.due_time > 0 and time > inst.depot.due_time + 1e-9:
        return float("inf"), False, []

    return dist, True, arr
