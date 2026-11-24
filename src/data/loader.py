"""Stream construction from config: synthetic generator or real XJTU-SY cache."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from src.data.stream import Stream
from src.data.synthetic import SyntheticSpec, generate

ROOT = Path(__file__).resolve().parents[2]


def _load_xjtu(data_cfg: dict) -> tuple[Stream, dict]:
    p = Path(data_cfg["npz"])
    if not p.is_absolute():
        p = ROOT / p
    if not p.exists():
        raise FileNotFoundError(f"xjtu cache not found: {p} (run scripts/build_xjtu_cache.py)")
    d = np.load(p, allow_pickle=False)
    X = np.asarray(d["X"], dtype=np.float32)
    rul = np.asarray(d["rul"], dtype=np.float32)

    stride = max(int(data_cfg.get("stride", 1)), 1)
    X, rul = X[::stride], rul[::stride]

    group = max(int(data_cfg.get("group", 1)), 1)
    if group > 1:
        # Keep every snapshot as its own row; `group` consecutive snapshots share
        # one step, so a step batch has `group` samples and metrics are averaged
        # over them.
        n = (len(X) // group) * group
        X, rul = X[:n], rul[:n]

    n_rows = len(X)
    t = (np.arange(n_rows, dtype=np.int32) // group).astype(np.int32)
    n_steps = int(t[-1]) + 1 if n_rows else 0
    meta = {
        "bearing": str(d["bearing"]),
        "condition": str(d["condition"]),
        "raw_snapshots": int(len(np.asarray(d["X"]))),
        "stride": stride,
        "group": group,
        "n_steps": int(n_steps),
    }
    return Stream(X, rul, t), meta


def build_stream(cfg: dict, seed: int) -> tuple[Stream, dict]:
    """Return (stream, meta) for the configured data source."""
    data_cfg = dict(cfg.get("data", {}))
    kind = str(data_cfg.get("kind", "synthetic")).lower()
    if kind == "xjtu":
        return _load_xjtu(data_cfg)
    if kind != "synthetic":
        raise ValueError(f"unknown data kind: {kind}")
    data_cfg["seed"] = int(seed)
    data = generate(SyntheticSpec.from_cfg(data_cfg))
    return Stream(data["X"], data["y"], data["t"]), {"kind": "synthetic"}
