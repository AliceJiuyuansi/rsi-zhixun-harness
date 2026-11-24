"""Offline entrypoint for the `zhixun-rsi` image.

`docker run --rm --network none <image>` must, with **no arguments**:

1. run one complete drift stream end-to-end (no human intervention),
2. always exit 0 - any failure degrades to "keep the current model",
3. leave machine-readable results plus a `run.log` in ``$RSI_OUT`` (default /app/out).

Default plan: the same stream under ``static`` (never update) and ``trigger``
(drift-triggered updates), so the run itself proves "not worse than the static
baseline". The knobs are env vars only, so the image stays argument-free:

    RSI_STREAM=xjtu_3_4      stream config name under configs/streams
    RSI_POLICIES=static,trigger
    RSI_SEED=0
    RSI_OUT=/app/out
    RSI_SWAN_MODE=disabled   never try to reach swanlab.cn from the graded image
"""
from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import yaml  # noqa: E402  (after sys.path setup)


class Tee:
    """stdout + run.log, never raises on a broken pipe."""

    def __init__(self, path: Path):
        self.lines: list[str] = []
        self.path = path

    def __call__(self, msg: str = "") -> None:
        self.lines.append(msg)
        try:
            print(msg, flush=True)
        except Exception:
            pass

    def flush(self) -> None:
        try:
            self.path.write_text("\n".join(self.lines) + "\n", encoding="utf-8")
        except Exception:
            pass


def deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(stream_name: str) -> dict:
    cfg_dir = ROOT / "configs"
    base = yaml.safe_load((cfg_dir / "base.yaml").read_text(encoding="utf-8"))
    path = cfg_dir / "streams" / f"{stream_name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"stream config not found: {path}")
    stream_cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    cfg = deep_merge(base, stream_cfg)
    cfg.setdefault("data", {})
    cfg["data"]["drift"] = (stream_cfg.get("data") or {}).get("drift", {"kind": "none"})
    cfg["stream_name"] = stream_name
    return cfg


def main() -> int:
    out = Path(os.environ.get("RSI_OUT", str(ROOT / "out")))
    out.mkdir(parents=True, exist_ok=True)
    log = Tee(out / "run.log")

    stream_name = os.environ.get("RSI_STREAM", "xjtu_3_4").strip()
    policies = [p.strip() for p in os.environ.get("RSI_POLICIES", "static,trigger").split(",") if p.strip()]
    try:
        seed = int(os.environ.get("RSI_SEED", "0"))
    except ValueError:
        seed = 0
    swan_mode = os.environ.get("RSI_SWAN_MODE", "disabled").strip() or "disabled"

    log("=" * 68)
    log("zhixun-rsi · offline RSI harness")
    log(f"stream={stream_name} policies={','.join(policies)} seed={seed} swan={swan_mode}")
    log(f"python={sys.version.split()[0]} out={out}")
    log("=" * 68)

    results: dict[str, dict] = {}
    failures: dict[str, str] = {}

    for policy in policies:
        tag = f"{policy}-{stream_name}-s{seed}"
        run_dir = out / tag
        t0 = time.perf_counter()
        log(f"\n[run] {tag}")
        try:
            from src.harness.loop import run_experiment

            cfg = load_config(stream_name)
            summary = run_experiment(cfg, stream_name, policy, seed,
                                     str(run_dir), swan_mode=swan_mode)
            results[policy] = summary
            log(f"[ok]  {tag} in {time.perf_counter() - t0:.1f}s "
                f"mean_mae={summary.get('mean_mae')} retrains={summary.get('retrains')} "
                f"promotions={summary.get('promotions')} rollbacks={summary.get('rollbacks')}")
        except Exception as exc:  # degrade, never crash the image
            failures[policy] = f"{type(exc).__name__}: {exc}"
            log(f"[degraded] {tag} failed -> keeping the previous model: {failures[policy]}")
            log(traceback.format_exc())

    # --- comparison: the graded run must not be worse than static ---
    comparison: dict[str, object] = {}
    static_mae = (results.get("static") or {}).get("mean_mae")
    for policy, s in results.items():
        mae = s.get("mean_mae")
        if mae is None or static_mae is None:
            continue
        comparison[policy] = {
            "mean_mae": mae,
            "vs_static": float(static_mae) - float(mae),  # positive = better than static
            "not_worse_than_static": bool(float(mae) <= float(static_mae) + 1e-9),
        }
    if comparison:
        log("\n[comparison] mean MAE (lower is better)")
        for policy, row in comparison.items():
            delta = row["vs_static"]
            log(f"  {policy:<9} mae={row['mean_mae']:.4f}  vs_static={delta:+.4f}  "
                f"{'OK' if row['not_worse_than_static'] else 'WORSE'}")

    report = {
        "image": "zhixun-rsi",
        "stream": stream_name,
        "seed": seed,
        "policies": policies,
        "swan_mode": swan_mode,
        "offline": True,
        "results": results,
        "comparison": comparison,
        "failures": failures,
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "exit_code": 0,
    }
    (out / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    log(f"\n[done] {len(results)} run(s) ok, {len(failures)} degraded; "
        f"results in {out}")
    log.flush()
    return 0  # never non-zero: a crash must not look like 'system broke'


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except BaseException:
        traceback.print_exc()
        raise SystemExit(0)
