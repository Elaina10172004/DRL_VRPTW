
import argparse
import json
from pathlib import Path

from cost_utils import compute_cost, CostConfig
import config


def summarize_cost(distance: float, vehicles: int, alpha: float = 250.0) -> float:
    return float(distance) + alpha * float(vehicles)



def load_results(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for name, info in data.items():
        rec = info.get("final") if isinstance(info, dict) else None
        if not rec:
            rec = info

        vehicles = rec.get("vehicles") if isinstance(rec, dict) else None
        dist = None
        if isinstance(rec, dict):
            dist = rec.get("total_distance") or rec.get("dist")

        if vehicles is None or dist is None:
            continue
        
        cost = compute_cost(dist, [], vehicles_override=vehicles)
        rows.append(
            {
                "instance": name,
                "vehicles": vehicles,
                "distance": dist,
                "cost": cost,
            }
        )
    return rows



def save_xlsx(rows, out_path: Path):
    try:
        from openpyxl import Workbook
    except ImportError:
        return False

    wb = Workbook()
    ws = wb.active
    ws.title = "results"
    ws.append(["instance", "cost", "vehicles", "distance"])
    for r in rows:
        ws.append([r["instance"], r["cost"], r["vehicles"], r["distance"]])
    wb.save(out_path)
    return True



def save_csv(rows, out_path: Path):
    import csv

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["instance", "cost", "vehicles", "distance"])
        for r in rows:
            writer.writerow([r["instance"], r["cost"], r["vehicles"], r["distance"]])



def main():
    ap = argparse.ArgumentParser(description="导出 all_results/solve 汇总为 Excel/CSV")
    ap.add_argument("--json", type=Path, default=Path(getattr(config.export_defaults, "json", "all_results.json")), help="输入 JSON 路径")
    ap.add_argument(
        "--out",
        type=Path,
        default=Path(getattr(config.export_defaults, "out", "results.xlsx")),
        help="输出 Excel 路径（缺少 openpyxl 时回退 CSV）",
    )
    args = ap.parse_args()

    if not args.json.exists():
        raise SystemExit(f"输入文件不存在: {args.json}")

    rows = load_results(args.json)
    if not rows:
        raise SystemExit("未解析到记录（检查 final/vehicles/total_distance 字段）")

    if save_xlsx(rows, args.out):
        print(f"已写入 {args.out}（Excel）")
    else:
        csv_path = args.out.with_suffix(".csv")
        save_csv(rows, csv_path)
        print(f"缺少 openpyxl，已写入 {csv_path}（CSV）")


if __name__ == "__main__":
    main()
