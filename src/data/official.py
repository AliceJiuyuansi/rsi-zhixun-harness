"""Official contest data packages (`data/raw/official_{a,b,c}/public/*.jsonl`).

Only the B profile (XJTU-SY RUL in minutes) is wired up so far. The loader is
deliberately dependency-free: jsonl in, `(X, y)` out, no pandas.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]

# 17 features per channel, order taken from TASK.md (B).
FEATURE_KEYS = [
    "rms", "peak", "peak_to_peak", "mean_abs", "kurtosis", "skewness",
    "crest_factor", "shape_factor", "impulse_factor", "margin_factor",
    "spectral_centroid", "spectral_rms", "spectral_kurtosis", "dominant_freq",
    "band1_energy_0_2k", "band2_energy_2k_6k", "band3_energy_6k_12k",
]
N_FEATURES = len(FEATURE_KEYS) * 2  # 34


def _row_vector(row: dict) -> list[float]:
    f = row["inputs"]["features"]
    h, v = f["horizontal"], f["vertical"]
    return [float(h[k]) for k in FEATURE_KEYS] + [float(v[k]) for k in FEATURE_KEYS]


def load_jsonl(path: str | Path) -> dict:
    """Read one official jsonl file -> arrays + traceability columns."""
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    X, y, unit, order = [], [], [], []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            X.append(_row_vector(row))
            y.append(float(row["answer"]))
            unit.append(str(row["id"]).split("_s")[0])
            order.append(int(row["meta"].get("source_index", len(order))))
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    unit = np.asarray(unit)
    order = np.asarray(order, dtype=np.int64)
    # keep each unit's snapshots in collection order
    perm = np.lexsort((order, unit))
    return {"X": X[perm], "y": y[perm], "unit": unit[perm], "path": str(path)}


def to_stream_arrays(data: dict, group: int = 20, max_steps: int | None = None):
    """Concatenate units into one time-ordered stream, `group` snapshots per step."""
    group = max(int(group), 1)
    X, y = data["X"], data["y"]
    n = (len(X) // group) * group
    X, y = X[:n], y[:n]
    t = (np.arange(n, dtype=np.int32) // group).astype(np.int32)
    if max_steps is not None:
        keep = t < int(max_steps)
        X, y, t = X[keep], y[keep], t[keep]
    return X, y, t
