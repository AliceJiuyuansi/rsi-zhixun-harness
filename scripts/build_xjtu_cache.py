"""Extract per-snapshot features from the raw XJTU-SY CSVs into npz caches.

Raw CSVs stay where they are (gitignored); output goes to data/processed/xjtu/.
Run:  python scripts/build_xjtu_cache.py            # all bearings, parallel
      python scripts/build_xjtu_cache.py --limit 200  # quick trial
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from common import ROOT  # noqa: F401  (sys.path setup)

from src.data.xjtu import bearing_features, discover_bearings

RAW_ROOT = ROOT / "XJTU-SY_Bearing_Datasets" / "Data" / "XJTU-SY_Bearing_Datasets" / "XJTU-SY_Bearing_Datasets"
OUT_ROOT = ROOT / "data" / "processed" / "xjtu"


def _one(args):
    key, bdir, out_dir, limit = args
    t0 = time.perf_counter()
    d = bearing_features(bdir, limit=limit)
    out = Path(out_dir) / (key.replace("/", "__") + ".npz")
    np.savez_compressed(
        out, X=d["X"], rul=d["rul"], file_id=d["file_id"],
        bearing=d["bearing"], condition=d["condition"],
    )
    return key, d["n"], time.perf_counter() - t0, str(out)


def main() -> int:
    import os
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=str(RAW_ROOT))
    ap.add_argument("--out", default=str(OUT_ROOT))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    bearings = discover_bearings(args.raw)
    if not bearings:
        print(f"no bearings found under {args.raw}")
        return 1
    print(f"found {len(bearings)} bearings, workers={args.workers}, limit={args.limit or 'all'}")

    t0 = time.perf_counter()
    todo = [(k, b, str(out_dir), args.limit or None) for k, b in bearings]
    manifest = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(_one, t): t[0] for t in todo}
        for i, fut in enumerate(as_completed(futures), 1):
            key, n, dt, path = fut.result()
            manifest.append({"key": key, "n": int(n), "seconds": round(dt, 2), "npz": path})
            print(f"[{i}/{len(todo)}] {key:22s} n={n:5d}  {dt:7.1f}s")
    manifest.sort(key=lambda r: r["key"])
    (out_dir / "manifest.json").write_text(
        json.dumps({"raw_root": str(args.raw), "features": 36, "bearings": manifest},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"done in {time.perf_counter() - t0:.1f}s -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
