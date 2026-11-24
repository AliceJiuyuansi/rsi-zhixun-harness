"""Run one policy on one drift stream.

Example:
    python scripts/run_stream.py --stream gradual --policy trigger --seed 1
"""
from __future__ import annotations

import argparse
import json

from common import load_config, new_outdir

from src.harness.loop import run_experiment


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stream", default="gradual")
    ap.add_argument("--policy", default="trigger", choices=["static", "periodic", "trigger"])
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--swan-mode", default="auto",
                    choices=["auto", "online", "local", "offline", "disabled"])
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()

    cfg = load_config(args.stream)
    outdir = args.outdir or new_outdir(f"{args.policy}-{args.stream}-s{args.seed}")
    summary = run_experiment(cfg, args.stream, args.policy, args.seed,
                             str(outdir), swan_mode=args.swan_mode)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
