"""Export human-readable CSVs so the results can be opened in Excel.

Outputs (all UTF-8 with BOM, so Chinese Excel opens them cleanly):
  view/数据_<bearing>.csv      one row per snapshot: RUL + a few key features
  view/对比_<stream>.csv       one row per step: static vs trigger 的滚动 MAE

Usage:
    python scripts/export_view.py
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from common import ROOT

VIEW = ROOT / "view"
NPZ_DIR = ROOT / "data" / "processed" / "xjtu"

# 特征列在 36 维向量里的位置（见 src/data/xjtu.py）
COLS = {
    "水平RMS": ("h", "rms"),
    "垂直RMS": ("v", "rms"),
    "水平峭度": ("h", "kurtosis"),
    "垂直峭度": ("v", "kurtosis"),
}
TIME = ["mean", "std", "rms", "max_abs", "p2p", "skew", "kurtosis",
        "crest", "shape", "impulse", "clearance", "zcr"]
FREQ = ["band0", "band1", "band2", "band3", "centroid", "spread"]


def _idx(ch: str, name: str) -> int:
    off = 0 if ch == "h" else 18
    return off + (TIME + FREQ).index(name)


def export_data(npz_name: str = "40Hz10kN__Bearing3_4.npz", stride: int = 6) -> Path:
    d = np.load(NPZ_DIR / npz_name, allow_pickle=False)
    X, rul = np.asarray(d["X"])[::stride], np.asarray(d["rul"])[::stride]
    out = VIEW / f"数据_{Path(npz_name).stem.replace('__', '_')}.csv"
    with out.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["序号", "归一化剩余寿命(1=全新,0=失效)"]
                   + list(COLS))
        for i in range(len(X)):
            w.writerow([i, round(float(rul[i]), 4)]
                       + [round(float(X[i, _idx(*v)]), 5) for v in COLS.values()])
    return out


def export_compare(run_dir: str, stream: str = "xjtu_3_4") -> Path:
    base = ROOT / run_dir
    series = {}
    for policy in ("static", "trigger"):
        rows = json.loads((base / f"{stream}-{policy}-s0" / "metrics.json")
                          .read_text(encoding="utf-8"))
        series[policy] = {r["step"]: -r["perf_rolling"] for r in rows}  # MAE
    steps = sorted(series["static"])
    out = VIEW / f"对比_{stream}.csv"
    with out.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["步数", "静态模型平均误差MAE", "自进化模型平均误差MAE", "差距"])
        for s in steps:
            a, b = series["static"][s], series.get("trigger", {}).get(s, float("nan"))
            w.writerow([s, round(a, 4), round(b, 4), round(a - b, 4)])
    return out


def main() -> int:
    VIEW.mkdir(parents=True, exist_ok=True)
    p1 = export_data()
    p2 = export_compare("experiments/matrix-20261111-134705")
    print(f"wrote {p1}")
    print(f"wrote {p2}")
    print("用 Excel 直接打开这两个文件即可。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
