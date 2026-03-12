

import csv
import json
import argparse
import matplotlib.pyplot as plt
import config
from vrptw_data import read_solomon


def _median(data):
    if not data:
        return 0.0
    s = sorted(data)
    n = len(s)
    mid = n // 2
    if n % 2 == 1:
        return float(s[mid])
    return (s[mid - 1] + s[mid]) / 2.0


def filter_outliers(pairs, threshold=5.0):
    if len(pairs) < 3:
        return pairs
    ys = [p[1] for p in pairs]
    med = _median(ys)
    devs = [abs(y - med) for y in ys]
    mad = _median(devs)
    if mad == 0:
        return pairs
    keep = []
    for (x, y), d in zip(pairs, devs):
        if d <= threshold * mad:
            keep.append((x, y))
    return keep




def plot_loss_and_distance(log_path: str):
    updates = []
    losses = []
    distances = []
    mean_costs = []
    entropies = []
    grad_norms = []
    lrs = []
    grad_emas = []
    alpha_gates = []
    epochs = []
    epoch_mean_costs = []
    epoch_losses = []
    epoch_entropies = []
    epoch_grad_norms = []
    epoch_eff_steps = []
    epoch_best_costs = []
    mon_epochs = []
    mon_distances = []
    mon_costs = []

    with open(log_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            step_raw = (row.get("step") or "").strip()
            update_raw = (row.get("update") or "").strip()
            epoch_raw = (row.get("epoch") or "").strip()
            
            if step_raw.startswith("eval_ep"):
                try:
                    ep = int(step_raw.replace("eval_ep", ""))
                except Exception:
                    continue
                mon_epochs.append(ep)
                mc = row.get("mean_cost")
                mon_costs.append(float(mc) if mc not in (None, "") else None)
                d = row.get("distance")
                mon_distances.append(float(d) if d not in (None, "") else None)
                continue
            
            if (step_raw and not step_raw.isdigit()) or (update_raw and not update_raw.isdigit()):
                continue
            
            if update_raw:
                updates.append(int(update_raw))
            elif step_raw:
                updates.append(int(step_raw))
            
            if epoch_raw and epoch_raw.isdigit():
                ep = int(epoch_raw)
                epochs.append(ep)
                mc = row.get("mean_cost")
                epoch_mean_costs.append(float(mc) if mc not in (None, "") else None)
                loss_val = row.get("loss")
                epoch_losses.append(float(loss_val) if loss_val not in (None, "") else None)
                gn = row.get("grad_norm")
                epoch_grad_norms.append(float(gn) if gn not in (None, "") else None)
                ent = row.get("entropy")
                epoch_entropies.append(float(ent) if ent not in (None, "") else None)
                eff = row.get("effective_step")
                epoch_eff_steps.append(float(eff) if eff not in (None, "") else None)
                bmc = row.get("best_mean_cost")
                epoch_best_costs.append(float(bmc) if bmc not in (None, "") else None)
            
            loss_val = row.get("loss")
            losses.append(float(loss_val) if loss_val not in (None, "") else None)
            d = row.get("distance")
            distances.append(float(d) if d not in (None, "") else None)
            mc = row.get("mean_cost")
            mean_costs.append(float(mc) if mc not in (None, "") else None)
            ent = row.get("entropy")
            entropies.append(float(ent) if ent not in (None, "") else None)
            gn = row.get("grad_norm")
            grad_norms.append(float(gn) if gn not in (None, "") else None)
            lr = row.get("lr")
            lrs.append(float(lr) if lr not in (None, "") else None)
            ge = row.get("grad_ema")
            grad_emas.append(float(ge) if ge not in (None, "") else None)
            ag = row.get("alpha_gate")
            alpha_gates.append(float(ag) if ag not in (None, "") else None)

    plt.figure()
    loss_pairs = [(s, l) for s, l in zip(updates, losses) if l is not None]
    loss_pairs = filter_outliers(loss_pairs)
    if loss_pairs:
        xs = [p[0] for p in loss_pairs]
        ys = [p[1] for p in loss_pairs]
        plt.plot(xs, ys)
    plt.xlabel("Update")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.savefig("loss.png")
    plt.tight_layout()

    
    cost_pairs = filter_outliers([(s, c) for s, c in zip(updates, mean_costs) if c is not None])
    if cost_pairs:
        plt.figure()
        xs = [p[0] for p in cost_pairs]
        ys = [p[1] for p in cost_pairs]
        plt.plot(xs, ys, label="Training mean_cost (update-based)")
        plt.xlabel("Update")
        plt.ylabel("mean_cost")
        plt.title("mean_cost (update-based)")
        plt.savefig("mean_cost_update.png")
        plt.tight_layout()
    
    epoch_cost_pairs = filter_outliers([(e, c) for e, c in zip(epochs, epoch_mean_costs) if c is not None])
    if epoch_cost_pairs:
        plt.figure()
        xs = [p[0] for p in epoch_cost_pairs]
        ys = [p[1] for p in epoch_cost_pairs]
        plt.plot(xs, ys, marker="o", label="epoch mean_cost")
        # overlay best_mean_cost
        best_pairs = filter_outliers([(e, c) for e, c in zip(epochs, epoch_best_costs) if c is not None])
        if best_pairs:
            bxs = [p[0] for p in best_pairs]
            bys = [p[1] for p in best_pairs]
            plt.plot(bxs, bys, linestyle="--", label="Epoch best mean_cost")
        plt.xlabel("epoch")
        plt.ylabel("mean_cost")
        plt.title("Epoch mean_cost")
        plt.legend()
        plt.savefig("mean_cost_epoch.png")
        plt.tight_layout()
    
    mon_cost_pairs = filter_outliers([(e, c) for e, c in zip(mon_epochs, mon_costs) if c is not None])
    if mon_cost_pairs:
        plt.figure()
        mx = [p[0] for p in mon_cost_pairs]
        my = [p[1] for p in mon_cost_pairs]
        plt.plot(mx, my, linestyle="--", label="Monitor mean_cost")
        plt.xlabel("epoch")
        plt.ylabel("mean_cost (monitor)")
        plt.title("Monitor mean_cost")
        plt.legend()
        plt.savefig("mean_cost_monitor.png")
        plt.tight_layout()

    
    if any(d is not None for d in distances):
        valid_pairs = filter_outliers([(s, d) for s, d in zip(updates, distances) if d is not None])
        if valid_pairs:
            plt.figure()
            xs = [p[0] for p in valid_pairs]
            ys = [p[1] for p in valid_pairs]
            plt.plot(xs, ys, label="Training distance")
            plt.xlabel("Update")
            plt.ylabel("distance")
            plt.title("Distance")
            plt.savefig("distance.png")
            plt.tight_layout()
    
    mon_valid = filter_outliers([(e, d) for e, d in zip(mon_epochs, mon_distances) if d is not None])
    if mon_valid:
        plt.figure()
        mxs = [p[0] for p in mon_valid]
        mys = [p[1] for p in mon_valid]
        plt.plot(mxs, mys, linestyle="--", label="Monitor distance")
        plt.xlabel("epoch")
        plt.ylabel("distance (monitor)")
        plt.title("Monitor distance")
        plt.legend()
        plt.savefig("distance_monitor.png")
        plt.tight_layout()

    
    ent_pairs = filter_outliers([(s, e) for s, e in zip(updates, entropies) if e is not None])
    if ent_pairs:
        plt.figure()
        xs = [p[0] for p in ent_pairs]
        ys = [p[1] for p in ent_pairs]
        plt.plot(xs, ys)
        plt.xlabel("Update")
        plt.ylabel("entropy")
        plt.title("Policy entropy")
        plt.savefig("entropy.png")
        plt.tight_layout()

    gn_pairs = filter_outliers([(s, g) for s, g in zip(updates, grad_norms) if g is not None])
    if gn_pairs:
        plt.figure()
        xs = [p[0] for p in gn_pairs]
        ys = [p[1] for p in gn_pairs]
        plt.plot(xs, ys, label="grad_norm")
        try:
            clip = float(config.train_defaults.max_grad_norm)
            plt.axhline(clip, color="red", linestyle="--", label=f"clip={clip}")
        except Exception:
            pass
        plt.xlabel("Update")
        plt.ylabel("grad_norm")
        plt.title("Gradient norm")
        plt.legend()
        plt.savefig("grad_norm.png")
        plt.tight_layout()

    lr_pairs = [(s, v) for s, v in zip(updates, lrs) if v is not None]
    if lr_pairs:
        plt.figure()
        xs = [p[0] for p in lr_pairs]
        ys = [p[1] for p in lr_pairs]
        plt.plot(xs, ys)
        plt.xlabel("Update")
        plt.ylabel("lr")
        plt.title("Learning rate")
        plt.savefig("lr.png")
        plt.tight_layout()

    ge_pairs = [(s, v) for s, v in zip(updates, grad_emas) if v is not None]
    if ge_pairs:
        plt.figure()
        xs = [p[0] for p in ge_pairs]
        ys = [p[1] for p in ge_pairs]
        plt.plot(xs, ys)
        plt.xlabel("Update")
        plt.ylabel("grad_ema")
        plt.title("Grad EMA")
        plt.savefig("grad_ema.png")
        plt.tight_layout()

    ag_pairs = [(s, v) for s, v in zip(updates, alpha_gates) if v is not None]
    if ag_pairs:
        plt.figure()
        xs = [p[0] for p in ag_pairs]
        ys = [p[1] for p in ag_pairs]
        plt.plot(xs, ys)
        plt.xlabel("Update")
        plt.ylabel("alpha_gate")
        plt.title("Alpha gate")
        plt.savefig("alpha_gate.png")
        plt.tight_layout()
    
    es_pairs = filter_outliers([(e, es) for e, es in zip(epochs, epoch_eff_steps) if es is not None])
    if es_pairs:
        plt.figure()
        xs = [p[0] for p in es_pairs]
        ys = [p[1] for p in es_pairs]
        plt.plot(xs, ys)
        plt.xlabel("epoch")
        plt.ylabel("effective_step")
        plt.title("Effective step (lr × grad_norm)")
        plt.savefig("effective_step_epoch.png")
        plt.tight_layout()




def _extract_routes_from_solution(sol: dict, sol_name: str | None = None):
    if "routes" in sol:  
        return sol["routes"], sol.get("total_distance")

    if not sol:
        raise ValueError("solution 文件内容为空")

    if sol_name is None:
        sol_name = next(iter(sol.keys()))
    
    lookup = {k.lower(): k for k in sol.keys()}
    key = lookup.get(sol_name.lower())
    if key is None:
        raise KeyError(f"未在解文件中找到实例 {sol_name}")

    entry = sol[key]
    if "final" not in entry or "routes" not in entry["final"]:
        raise ValueError("解文件格式不包含 final/routes 字段")
    final = entry["final"]
    return final["routes"], final.get("total_distance")




def plot_routes(instance_path: str, solution_path: str, n_customers: int | None = None, sol_name: str | None = None):
    inst = read_solomon(instance_path)

    with open(solution_path, encoding="utf-8") as f:
        sol = json.load(f)

    routes, total_dist = _extract_routes_from_solution(sol, sol_name)

    if n_customers is None:
        max_idx = max(cid for r in routes for cid in r) if routes else -1
        n_customers = max_idx + 1

    inst.customers = inst.customers[:n_customers]

    depot_x, depot_y = inst.depot.x, inst.depot.y
    cust_x = [c.x for c in inst.customers]
    cust_y = [c.y for c in inst.customers]

    plt.figure()
    plt.scatter(cust_x, cust_y, s=15, label="Customers")
    plt.scatter([depot_x], [depot_y], s=80, marker="*", label="Depot")

    for rid, r in enumerate(routes):
        rx = [depot_x] + [cust_x[i] for i in r] + [depot_x]
        ry = [depot_y] + [cust_y[i] for i in r] + [depot_y]
        plt.plot(rx, ry)
        if r:
            plt.text(cust_x[r[0]], cust_y[r[0]], f"v{rid}", fontsize=8)

    plt.xlabel("x coord")
    plt.ylabel("y coord")
    title = f"Routes: {solution_path}"
    if total_dist is not None:
        title += f" (total_dist={total_dist:.2f}, n={n_customers})"
    plt.title(title)
    plt.legend()
    plt.gca().set_aspect("equal", adjustable="box")
    plt.tight_layout()



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default="train_log.csv", help="训练日志 CSV 路径")
    ap.add_argument("--instance", help="Solomon 实例路径，例：solomon_data/c101.txt")
    ap.add_argument("--solution", help="解的 JSON 路径，例：neural.json 或 all_results.json")
    ap.add_argument("--solution_name", help="all_results.json 时指定实例名，不填则取首个")
    ap.add_argument("--n_customers", type=int, help="使用的客户数量（与训练/解码时一致）")
    args = ap.parse_args()

    plot_loss_and_distance(args.log)

    if args.instance and args.solution:
        plot_routes(args.instance, args.solution, n_customers=args.n_customers, sol_name=args.solution_name)
        plt.savefig("routes.png")

    plt.show()


if __name__ == "__main__":
    main()
