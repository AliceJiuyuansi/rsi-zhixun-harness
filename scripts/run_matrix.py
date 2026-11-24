"""Run the controlled experiment matrix: stream x policy x seed.

Produces experiments/matrix-<ts>/matrix.csv and prints a markdown table.
Delta is always measured against the static policy on the SAME stream+seed,
which is the proxy for the official static baseline.
"""
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime

from common import EXP_DIR, list_streams, load_config

from src.harness.loop import run_experiment


def _fmt(x: float, nd: int = 4) -> str:
    if x != x:  # NaN
        return "n/a"
    return f"{x:.{nd}f}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--streams", default="all")
    ap.add_argument("--policies", default="static,periodic,trigger")
    ap.add_argument("--seeds", default="1")
    ap.add_argument("--swan-mode", default="auto",
                    choices=["auto", "online", "local", "offline", "disabled"])
    args = ap.parse_args()

    streams = list_streams() if args.streams == "all" else args.streams.split(",")
    policies = args.policies.split(",")
    # static must run first: every other delta is measured against it
    policies = sorted(dict.fromkeys(policies), key=lambda p: (p != "static", p))
    seeds = [int(s) for s in args.seeds.split(",")]
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    base_dir = EXP_DIR / f"matrix-{stamp}"
    base_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for stream in streams:
        cfg = load_config(stream)
        for seed in seeds:
            static_mean = None
            for policy in policies:
                name = f"{stream}-{policy}-s{seed}"
                summ = run_experiment(cfg, stream, policy, seed,
                                      str(base_dir / name), swan_mode=args.swan_mode)
                if policy == "static":
                    static_mean = summ["mean_perf_post"]
                delta = (summ["mean_perf_post"] - static_mean) if static_mean is not None else 0.0
                cost = summ["cost"]["weighted_cost"]
                unit = cost / max(delta, 1e-3) if policy != "static" else float("nan")
                results.append({
                    "stream": stream,
                    "policy": policy,
                    "seed": seed,
                    "ref_perf": summ["ref_perf"],
                    "mean_perf": summ["mean_perf_post"],
                    "final_perf": summ["final_perf"],
                    "delta_vs_static": delta,
                    "retrains": summ["retrains"],
                    "promotions": summ["promotions"],
                    "rollbacks": summ["rollbacks"],
                    "gpu_seconds": summ["cost"]["gpu_seconds"],
                    "labels": summ["cost"]["labels"],
                    "weighted_cost": cost,
                    "unit_gain_cost": unit,
                })
                print(f"[matrix] {name}: mean={summ['mean_perf_post']:.4f} "
                      f"delta={delta:+.4f} retrains={summ['retrains']}")

    csv_path = base_dir / "matrix.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    headers = ["stream", "policy", "ref_perf", "mean_perf", "delta_vs_static",
               "retrains", "rollbacks", "weighted_cost", "unit_gain_cost"]
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in results:
        lines.append("| " + " | ".join([
            r["stream"], r["policy"], _fmt(r["ref_perf"]), _fmt(r["mean_perf"]),
            _fmt(r["delta_vs_static"]), str(r["retrains"]), str(r["rollbacks"]),
            _fmt(r["weighted_cost"], 2),
            "&mdash;" if r["policy"] == "static" else _fmt(r["unit_gain_cost"], 3),
        ]) + " |")
    # aggregate by policy: mean delta, total cost, unit cost
    agg: dict[str, dict] = {}
    for r in results:
        a = agg.setdefault(r["policy"], {"n": 0, "delta": 0.0, "perf": 0.0, "cost": 0.0, "retrains": 0})
        a["n"] += 1
        a["delta"] += r["delta_vs_static"]
        a["perf"] += r["mean_perf"]
        a["cost"] += r["weighted_cost"]
        a["retrains"] += r["retrains"]
    agg_lines = ["| policy | runs | mean_perf | mean_delta_vs_static | total_cost | unit_gain_cost | retrains |",
                 "|---|---|---|---|---|---|---|"]
    for policy, a in agg.items():
        n = max(a["n"], 1)
        mean_delta = a["delta"] / n
        unit = "&mdash;" if policy == "static" else f"{a['cost'] / max(mean_delta, 1e-3):.2f}"
        agg_lines.append(f"| {policy} | {a['n']} | {a['perf']/n:.4f} | {mean_delta:+.4f} | "
                         f"{a['cost']:.2f} | {unit} | {a['retrains']} |")
    agg_md = "\n".join(agg_lines)

    md = "\n".join(lines)
    (base_dir / "matrix.md").write_text(md + "\n\n## Aggregate\n\n" + agg_md + "\n", encoding="utf-8")
    (base_dir / "matrix.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n" + md)
    print("\n## Aggregate\n\n" + agg_md)
    print(f"\nwrote {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
