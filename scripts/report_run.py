"""把一次实验的结果翻译成人话。

用法：
    python scripts/report_run.py                          # 自动找最新的一次 trigger 实验
    python scripts/report_run.py experiments/matrix-xxx/xjtu_3_4-trigger-s0
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import ROOT


def _latest_run() -> Path:
    runs = sorted(ROOT.glob("experiments/matrix-*/xjtu_3_4-trigger-s0"),
                  key=lambda p: p.stat().st_mtime)
    if not runs:
        raise SystemExit("没找到实验记录，先跑一次 scripts/run_stream.py")
    return runs[-1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run", nargs="?", default=None)
    args = ap.parse_args()
    run = Path(args.run) if args.run else _latest_run()
    if not run.is_absolute():
        run = ROOT / run

    s = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    rows = json.loads((run / "metrics.json").read_text(encoding="utf-8"))

    print("=" * 68)
    print(f"实验：{s['run']}")
    print("=" * 68)
    print(f"策略        : {s['policy']}（static=永不更新, periodic=定期, trigger=发现问题才更）")
    print(f"数据        : {s['stream']}  {s['data'].get('bearing', '')} "
          f"{s['data'].get('condition', '')}")
    print(f"任务        : {s['task']}   步数: {s['n_steps']}   训练用前 {s['warmup_steps']} 步")

    print("\n【成绩单】")
    if s["task"] == "regression":
        print(f"  平均误差 MAE : {s.get('mean_mae', float('nan')):.3f}  "
              f"（0.10 = 平均差 10% 寿命）")
        print(f"  最后一步误差 : {s.get('final_mae', float('nan')):.3f}")

    print("\n【它做了什么】")
    print(f"  重训次数 : {s['retrains']}")
    print(f"  晋升次数 : {s['promotions']}")
    print(f"  回滚次数 : {s['rollbacks']}")
    diag = s.get("diagnosis_counts", {})
    if diag:
        print("  诊断分布 : " + "，".join(f"{k}={v}" for k, v in diag.items()))

    print("\n【花了多少】")
    c = s["cost"]
    print(f"  GPU 秒数 : {c['gpu_seconds']:.2f}")
    print(f"  标注条数 : {c['labels']}")
    print(f"  加权成本 : {c['weighted_cost']:.2f}")
    print("  单位增益成本需要和静态基线对比，见 experiments/matrix-*/matrix.md")

    print("\n【关键步骤】（误差 = 最近 8 步的平均 MAE）")
    print(f"  {'步':>4}  {'误差':>7}  {'PSI':>7}  {'诊断':<9}  动作")
    shown = set()
    for r in rows:
        if r["retrain"] or r["rollback"] or r["step"] % 10 == 0:
            if r["step"] in shown:
                continue
            shown.add(r["step"])
            act = "重训" if r["retrain"] else ("回滚" if r["rollback"] else "-")
            print(f"  {r['step']:>4}  {-r['perf_rolling']:>7.3f}  {r['psi_max']:>7.2f}  "
                  f"{r['diagnosis']:<9}  {act}")

    print("\n【怎么看】")
    first, last = rows[0], rows[-1]
    print(f"  开始时误差 {-first['perf_rolling']:.3f}，结束时 {-last['perf_rolling']:.3f}。")
    if s["policy"] == "static":
        print("  这是对照组的极端：模型永远不更新，误差会一路涨。")
    elif s["retrains"] == 0:
        print("  这次没有触发更新，说明要么没漂移，要么阈值太高。")
    else:
        print(f"  系统共更新 {s['promotions']} 次，把误差压住了。")
        if s["rollbacks"]:
            print(f"  其中回滚 {s['rollbacks']} 次，说明保护机制被真实触发过。")
    print("\n还想看别的实验：python scripts/report_run.py <run目录>")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
