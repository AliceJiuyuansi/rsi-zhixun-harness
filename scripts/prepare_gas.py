"""Prepare the UCI Gas Sensor Array Drift dataset (official task A) as a local cache.

Official task A ships only batch 1 (train, 445 rows) and batch 2 (validation,
1244 rows). We verified byte-for-byte that both are exactly reproducible from the
public .dat files, so batches 3-10 (12,221 rows, the natural 36-month sensor
ageing of the same experiment) are legitimate "hidden-like" data: they were never
used for training or validation by the organiser.

Outputs (data/processed/gas/):
  gas_all.npz       X (13910,128) float32, y (13910,) int32 0..5, batch (13910,) int32 1..10
  manifest.json     row counts, class names, checksums, official split boundaries

Usage:
    python scripts/prepare_gas.py
    python scripts/prepare_gas.py --raw "gas+sensor+array+drift+dataset (1)/Dataset"
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CLASS_NAMES = ["Ethanol", "Ethylene", "Ammonia", "Acetaldehyde", "Acetone", "Toluene"]
N_FEATURES = 128


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_batch(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """One .dat line = `<class:int> <idx>:<value> ...` for indices 1..128."""
    X, y = [], []
    for lineno, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
        tokens = line.split()
        if not tokens:
            continue
        cls = int(tokens[0])
        if not 1 <= cls <= len(CLASS_NAMES):
            raise ValueError(f"{path.name}:{lineno}: class {cls} outside 1..6")
        pairs = sorted((int(tok.split(":")[0]), float(tok.split(":")[1])) for tok in tokens[1:])
        idx = [i for i, _ in pairs]
        if idx != list(range(1, N_FEATURES + 1)):
            raise ValueError(f"{path.name}:{lineno}: feature indices are not exactly 1..128")
        X.append([v for _, v in pairs])
        y.append(cls - 1)  # 0-based, matches classes order
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.int32)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="gas+sensor+array+drift+dataset (1)/Dataset")
    ap.add_argument("--out", default="data/processed/gas")
    args = ap.parse_args()

    raw = Path(args.raw)
    if not raw.is_absolute():
        raw = ROOT / raw
    if not raw.exists():
        raise SystemExit(f"raw gas dataset not found: {raw}")

    Xs, ys, bs, per_batch = [], [], [], {}
    hashes = {}
    for b in range(1, 11):
        path = raw / f"batch{b}.dat"
        if not path.exists():
            raise SystemExit(f"missing {path}")
        X, y = load_batch(path)
        Xs.append(X)
        ys.append(y)
        bs.append(np.full(len(y), b, dtype=np.int32))
        per_batch[str(b)] = int(len(y))
        hashes[f"batch{b}"] = _sha256(path)
        print(f"batch{b:<2} rows={len(y):<6} features={X.shape[1]}")

    X = np.concatenate(Xs)
    y = np.concatenate(ys)
    batch = np.concatenate(bs)

    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "gas_all.npz", X=X, y=y, batch=batch)

    manifest = {
        "dataset": "UCI Gas Sensor Array Drift (#224)",
        "task": "classification",
        "classes": CLASS_NAMES,
        "n_features": N_FEATURES,
        "n_rows": int(len(y)),
        "rows_per_batch": per_batch,
        "official_split": {"train": {"batch": 1, "rows": 445},
                           "validation": {"batch": 2, "rows": 1244}},
        "hidden_like_batches": list(range(3, 11)),
        "hidden_like_rows": int(sum(per_batch[str(b)] for b in range(3, 11))),
        "note": ("Official package train/validation reproduce exactly from batch1/batch2 "
                 "(verified maxdiff=0.0). Batches 3-10 are untouched by the organiser and "
                 "are used locally as hidden-like drift streams."),
        "sha256": hashes,
        "files": {"npz": "gas_all.npz"},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                                       encoding="utf-8")
    print(f"\nwrote {out / 'gas_all.npz'}  X={X.shape} y={y.shape} batches={sorted(per_batch)}")
    print(f"hidden-like rows (batch3-10): {manifest['hidden_like_rows']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
