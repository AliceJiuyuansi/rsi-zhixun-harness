"""Simulate the platform's 10-round continuation on task A, locally.

Platform constraints reproduced here:

- gradient training uses ONLY the public training split (batch 1, 445 rows);
- the public validation split (batch 2, 1244 rows) may be used for measurement
  and selection, but NOT merged into gradient training;
- batches 3-10 are never trained on and never used to pick a checkpoint -- they
  are only measured, because they are almost certainly the hidden test stream.

What the harness controls is therefore: when to continue training, for how many
epochs, at what learning rate, and whether to accept the result.

    python scripts/simulate_a_rounds.py --rounds 10
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.official_round import OfficialRound  # noqa: E402
from src.monitor.drift import psi as psi_fn  # noqa: E402
from src.policy.strategy import Signal, choose, diagnose  # noqa: E402

TEST_BATCHES = list(range(3, 11))


def feature_psi(ref: np.ndarray, cur: np.ndarray) -> float:
    scores = [psi_fn(ref[:, j], cur[:, j]) for j in range(ref.shape[1])]
    return float(np.max(scores))


def load_data():
    d = np.load(ROOT / "data/processed/gas/gas_all.npz")
    X, y, batch = d["X"].astype(float), d["y"], d["batch"]
    return X, y, batch


class Strategy:
    def __init__(self, name: str, rounds: int):
        self.name = name
        self.rounds = rounds
        self.last_update = -99

    def decide(self, round_index: int, psi: float, val_perf: float, target: float) -> dict:
        gap = max(target - val_perf, 0.0)
        if self.name == "static":
            return {"kind": "none", "reason": "static: never update"}
        if self.name == "periodic":
            due = round_index % 2 == 0
            if due:
                return {"kind": "finetune", "params": {"epochs": 5, "lr": 3e-3},
                        "reason": f"periodic: round {round_index} % 2 == 0"}
            return {"kind": "none", "reason": "not scheduled"}
        # trigger: reuse the documented drift-band table
        if round_index - self.last_update < 2:
            return {"kind": "none", "reason": "cooldown"}
        action = choose(Signal(psi=psi, gap=gap, pred_js=0.0))
        reason = (f"psi={psi:.3f} gap={gap:.3f} -> {diagnose(Signal(psi=psi, gap=gap))} "
                  f"-> {action.kind}")
        return {"kind": action.kind, "params": action.params, "reason": reason}


def run_strategy(name: str, rounds: int, X, y, batch, start_ckpt: Path, workdir: Path,
                 log_step=None) -> dict:
    tr, va = batch == 1, batch == 2
    Xtr, ytr = X[tr], y[tr]
    Xva, yva = X[va], y[va]
    test_idx = {b: (X[batch == b], y[batch == b]) for b in TEST_BATCHES}

    eng = OfficialRound()
    out = workdir / name
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True, exist_ok=True)

    champion = out / "round-0.npz"
    shutil.copyfile(start_ckpt, champion)
    champ_metrics = eng.evaluate_checkpoint(champion, Xva, yva)
    target = champ_metrics["macro_f1"]
    psi = feature_psi(Xtr, Xva)

    strategy = Strategy(name, rounds)
    total_epochs = 0
    total_seconds = 0.0
    history = []
    updates = promotions = 0

    for k in range(1, rounds + 1):
        decision = strategy.decide(k, psi=psi, val_perf=champ_metrics["macro_f1"], target=target)
        promoted = False
        cand_metrics = champ_metrics
        if decision["kind"] == "finetune" and k > 0:
            params = decision["params"]
            res = eng.continue_round(
                champion, Xtr, ytr, Xva, yva,
                out_dir=out / f"round-{k}", epochs=int(params.get("epochs", 3)),
                learning_rate=float(params.get("lr", 1.5e-3)), stage_id=f"{name}-round-{k}",
                hypothesis=decision["reason"], change_summary=f"{params}")
            updates += 1
            total_epochs += int(params.get("epochs", 3))
            total_seconds += res.seconds
            strategy.last_update = k
            cand_metrics = res.metrics_after
            if cand_metrics["macro_f1"] >= champ_metrics["macro_f1"] - 0.02:
                champion = res.checkpoint
                champ_metrics = cand_metrics
                promoted = True
                promotions += 1
                target = max(target, cand_metrics["macro_f1"])

        test_scores = {b: eng.evaluate_checkpoint(champion, Xb, yb)["macro_f1"]
                       for b, (Xb, yb) in test_idx.items()}
        history.append({"round": k, "action": decision["kind"], "reason": decision["reason"],
                        "val_macro_f1": champ_metrics["macro_f1"],
                        "test_mean_macro_f1": float(np.mean(list(test_scores.values()))),
                        "test_worst_macro_f1": float(np.min(list(test_scores.values()))),
                        "promoted": promoted, "per_batch": {str(b): v for b, v in test_scores.items()}})
        print(f"[{name:8}] round {k:2d} {decision['kind']:<9} "
              f"val={champ_metrics['macro_f1']:.4f} "
              f"test_mean={history[-1]['test_mean_macro_f1']:.4f} "
              f"test_worst={history[-1]['test_worst_macro_f1']:.4f}")
        if log_step is not None:
            log_step(history[-1], k)

    summary = {
        "strategy": name,
        "rounds": rounds,
        "updates": updates,
        "promotions": promotions,
        "total_epochs": total_epochs,
        "train_seconds": round(total_seconds, 3),
        "val_macro_f1": champ_metrics["macro_f1"],
        "test_mean_macro_f1": history[-1]["test_mean_macro_f1"],
        "test_worst_macro_f1": history[-1]["test_worst_macro_f1"],
        "per_batch": history[-1]["per_batch"],
        "history": history,
    }
    return summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--strategies", default="static,periodic,trigger")
    ap.add_argument("--start", default="data/raw/official_a/checkpoint.npz")
    ap.add_argument("--out", default="reports/sim_a_rounds.json")
    ap.add_argument("--swan-mode", default="disabled",
                    choices=["disabled", "online", "local", "offline"],
                    help="log every simulated round to SwanLab (online needs SWANLAB_API_KEY)")
    args = ap.parse_args()

    X, y, batch = load_data()
    start = Path(args.start)
    if not start.is_absolute():
        start = ROOT / start
    workdir = ROOT / "_state" / "sim"
    workdir.mkdir(parents=True, exist_ok=True)

    results = []
    for name in [s.strip() for s in args.strategies.split(",") if s.strip()]:
        logger = None
        if args.swan_mode != "disabled":
            import swanlab
            logger = swanlab.init(project="rsi-zhixun", workspace="xiangxinnidexin",
                                  name=f"gas-a-{name}-rounds", mode=args.swan_mode,
                                  config={"dataset": "official_a_batch1", "strategy": name,
                                          "rounds": args.rounds},
                                  tags=["official-A", "gas", name])

        def log_step(record: dict, step: int, _logger=logger) -> None:
            if _logger is None:
                return
            _logger.log({"val_macro_f1": record["val_macro_f1"],
                         "test_mean_macro_f1": record["test_mean_macro_f1"],
                         "test_worst_macro_f1": record["test_worst_macro_f1"]}, step=step)

        results.append(run_strategy(name, args.rounds, X, y, batch, start, workdir,
                                    log_step=log_step))
        if logger is not None:
            try:
                print(f"[swan] {name} run url: {getattr(logger, 'url', '')}")
                logger.finish()
            except Exception as exc:  # noqa: BLE001
                print(f"[swan] finish failed: {exc}")

    print("\n| strategy | updates | promoted | epochs | val macro-F1 | test mean | test worst |")
    print("|---|---|---|---|---|---|---|")
    for r in results:
        print(f"| {r['strategy']} | {r['updates']} | {r['promotions']} | {r['total_epochs']} | "
              f"{r['val_macro_f1']:.4f} | {r['test_mean_macro_f1']:.4f} | {r['test_worst_macro_f1']:.4f} |")

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
