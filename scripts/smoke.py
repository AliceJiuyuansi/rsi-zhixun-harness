"""Fast end-to-end sanity check. Must finish in a few seconds.

    python scripts/smoke.py
"""
from __future__ import annotations

import json
from datetime import datetime

from common import EXP_DIR, load_config

from src.harness.loop import run_experiment


def main() -> int:
    cfg = load_config("stable")
    cfg["data"]["n_steps"] = 40
    cfg["data"]["warmup_steps"] = 10
    cfg["policy"]["periodic_every"] = 10
    outdir = EXP_DIR / f"_smoke-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    summary = run_experiment(cfg, "stable", "static", 0, str(outdir), swan_mode="disabled")

    metrics = json.loads((outdir / "metrics.json").read_text(encoding="utf-8"))
    assert (outdir / "cost.json").exists(), "cost.json missing"
    assert (outdir / "summary.json").exists(), "summary.json missing"
    assert len(metrics) > 0, "no per-step metrics"
    assert summary["mean_perf_post"] == summary["mean_perf_post"], "perf is NaN"
    print(json.dumps({k: summary[k] for k in
                      ("run", "mean_perf_post", "retrains", "rollbacks", "swanlab_mode")},
                     ensure_ascii=False, indent=2))
    print(f"SMOKE OK -> {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
