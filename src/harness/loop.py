"""End-to-end experiment loop: stream -> predict -> monitor -> decide -> update -> rollback.

Task-agnostic: `data.task` selects classification (accuracy) or regression
(-MAE, used for normalized RUL). Every performance number is higher-is-better.

Every decision passes through `harness.guard.Gatekeeper` (kill-switch ->
whitelist -> budget -> amplitude cap) and every executed action is appended to
`<outdir>/audit.jsonl`. A failing step degrades to "keep the current champion"
instead of killing the run: unattended completion is a hard gate in the rules.
"""
from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import yaml

from src.data.loader import build_stream
from src.eval.cost import CostLedger
from src.eval.metrics import RollingMetrics, mae, step_perf
from src.harness.guard import Gatekeeper
from src.models.baseline import build_model, fit_model, score_model
from src.monitor.drift import DriftMonitor
from src.policy.policies import PolicyState, build_policy
from src.policy.whitelist import NONE_ACTION
from src.schema import GateDecision
from src.utils.audit import AuditLog
from src.utils.swan import start_run


def _finalize(obj):
    if isinstance(obj, dict):
        return {k: _finalize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_finalize(v) for v in obj]
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def run_experiment(cfg: dict, stream_name: str, policy_name: str, seed: int,
                   outdir: str, swan_mode: str = "auto") -> dict:
    data_cfg = dict(cfg.get("data", {}))
    task = str(data_cfg.get("task", "classification")).lower()
    stream, data_meta = build_stream(cfg, seed)

    warmup = int(data_cfg.get("warmup_steps", 24))
    warmup = max(2, min(warmup, len(stream.steps) - 1))
    label_lag = int(data_cfg.get("label_lag", 1))
    win = int(cfg.get("policy", {}).get("window", 40))
    eval_cfg = cfg.get("eval", {})
    rolling = RollingMetrics(window=int(eval_cfg.get("rolling_window", 10)))
    cost_cfg = cfg.get("cost", {})
    ledger = CostLedger(alpha_gpu=float(cost_cfg.get("alpha_gpu", 1.0)),
                        beta_label=float(cost_cfg.get("beta_label", 0.001)),
                        gamma_token=float(cost_cfg.get("gamma_token", 0.0)))
    model_cfg = dict(cfg.get("model", {}))
    policy = build_policy(dict(cfg.get("policy", {}), type=policy_name))
    monitor = DriftMonitor(psi_threshold=float(cfg["policy"].get("psi_threshold", 0.25)),
                           task=task)
    err_thr = float(cfg["policy"].get("error_threshold", 0.06))
    margin = float(cfg["policy"].get("promote_margin", 0.02))
    rb_thr = float(cfg["policy"].get("rollback_threshold", 0.10))
    rb_patience = int(cfg["policy"].get("rollback_patience", 3))

    run_name = f"{policy_name}-{stream_name}-s{seed}"
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)

    # --- controlled-RSI five-piece: audit log + gatekeeper -------------
    audit = AuditLog(out / "audit.jsonl", run=run_name)
    guard = Gatekeeper(cfg, audit=audit, root=out)
    audit.write("run_start", stream=stream_name, policy=policy_name, seed=int(seed),
                task=task, warmup_steps=warmup, window=win,
                guard=guard.as_dict())

    logger = start_run(
        project=cfg.get("project", "rsi-zhixun"),
        workspace=cfg.get("workspace", "xiangxinnidexin"),
        name=run_name,
        config=_finalize({"stream": stream_name, "policy": policy_name,
                          "seed": seed, "task": task, **cfg}),
        outdir=outdir,
        tags=[stream_name, policy_name, task],
        group=f"{stream_name}",
        mode=swan_mode,
    )

    # --- initial champion (static baseline of this run) ---
    Xw, yw = stream.warmup(warmup)
    res = fit_model(Xw, yw, model_cfg, seed, task=task)
    champion, champion_val = res.model, res.val_score
    champion_anchor = float("nan")
    prev_model, prev_val = None, float("nan")
    ledger.add_train(res.seconds, res.n_labels, is_retrain=False)

    monitor.set_reference(Xw)
    monitor.set_reference_predictions(champion.predict(Xw))
    ref_perf = float(champion_val)
    # What we currently believe is achievable. Drops to the challenger's own
    # validation score when the operating regime changes, so the controller does
    # not chase an unreachable pre-drift score forever.
    target_perf = ref_perf
    bad_steps = 0
    diag_counts: Counter = Counter()
    rows: list[dict] = []
    last_retrain_step = -1
    promotions = 0
    degraded_steps = 0

    try:
        for step in stream.steps[warmup:]:
            step = int(step)
            try:
                Xb, yb = stream.batch(step)
                t0 = time.perf_counter()
                pred = champion.predict(Xb)
                ledger.add_inference(time.perf_counter() - t0)

                perf = step_perf(task, yb, pred)
                rolling.update(perf)
                if np.isnan(champion_anchor):
                    champion_anchor = rolling.perf

                # Drift statistics use the whole recent window: a single step batch
                # (a handful of rows) is far too small for a 10-bin PSI.
                recent_X = stream.recent(step, win)
                t0 = time.perf_counter()
                recent_pred = champion.predict(recent_X)
                ledger.add_inference(time.perf_counter() - t0)
                feat = monitor.feature_score(recent_X)
                psi_max = feat["psi_max"]
                pred_js = monitor.pred_shift(recent_pred)
                error_gap = max(target_perf - rolling.perf, 0.0)
                diag = monitor.diagnose(psi_max, error_gap, pred_js, err_thr)
                diag_counts[diag["kind"]] += 1

                state = PolicyState(
                    step=step, retrains=ledger.retrains,
                    last_retrain_step=last_retrain_step,
                    psi_max=psi_max, error_gap=error_gap, diagnosis=diag,
                )
                proposal = policy.decide(state)
                decision = guard.review(proposal, step=step, retrains=ledger.retrains)
                # Belt and braces: the executor below only knows `retrain`. An
                # approved-but-unimplemented action must never be silently
                # executed as a retrain.
                if decision.approved and decision.action.kind != "retrain":
                    audit.write("reject_unimplemented", step=step,
                                action=str(decision.action.kind))
                    decision = GateDecision(False,
                                            f"executor has no implementation for "
                                            f"'{decision.action.kind}'", NONE_ACTION)
                did_retrain = promoted = rolled_back = False
                before_perf = float(rolling.perf)

                if decision.approved:
                    hist_X, hist_y = stream.history(step, label_lag, win)
                    min_hist = int(cfg["policy"].get("min_train_samples", 100))
                    enough = len(hist_X) >= min_hist
                    if task != "regression":
                        enough = enough and len(np.unique(hist_y)) >= 2
                    if enough:
                        last_retrain_step = step
                        did_retrain = True
                        # Champion vs challenger must be compared on the SAME holdout
                        # (the newest labeled window). Comparing scores measured on
                        # different regimes makes every challenger look worse.
                        split = max(int(len(hist_X) * 0.8), 1)
                        fit_X, fit_y = hist_X[:split], hist_y[:split]
                        hold_X, hold_y = hist_X[split:], hist_y[split:]
                        t0 = time.perf_counter()
                        cand_model = build_model(model_cfg, seed, task)
                        cand_model.fit(fit_X, fit_y)
                        ledger.add_train(time.perf_counter() - t0, len(fit_X))
                        cand_score, _ = score_model(task, cand_model, hold_X, hold_y)
                        champ_score, _ = score_model(task, champion, hold_X, hold_y)
                        if cand_score >= champ_score - margin:
                            # Refit the winner on the full window (more data, same regime).
                            t0 = time.perf_counter()
                            full_model = build_model(model_cfg, seed, task)
                            full_model.fit(hist_X, hist_y)
                            # Same labeled window, refit for data efficiency: compute
                            # counts, labels do not (they were acquired once).
                            ledger.add_train(time.perf_counter() - t0, 0, is_retrain=False)
                            prev_model, prev_val = champion, champion_val
                            champion, champion_val = full_model, cand_score
                            champion_anchor = rolling.perf
                            promoted = True
                            promotions += 1
                            target_perf = min(ref_perf, max(cand_score, target_perf - 0.10))
                            # The reference window follows the champion: once the model
                            # has adapted, the new regime is the new normal. This is what
                            # makes trigger-based updating cheaper than periodic retraining.
                            monitor.set_reference(hist_X)
                            monitor.set_reference_predictions(champion.predict(hist_X))
                        decision.reason += f"; challenger={cand_score:.3f} vs champion={champ_score:.3f}"
                        guard.commit(decision.action, step)
                        audit.write(
                            "action", step=step, action=str(decision.action.kind),
                            reason=decision.reason,
                            diagnosis=diag["kind"], psi_max=float(psi_max),
                            error_gap=float(error_gap), perf_before=before_perf,
                            challenger=float(cand_score), champion=float(champ_score),
                            promoted=bool(promoted),
                            gpu_seconds=float(ledger.total.gpu_seconds),
                            labels=int(ledger.total.labels),
                            tokens=int(ledger.total.tokens))
                    else:
                        decision.reason += "; not enough labeled history"

                # rollback: champion degrading relative to its own anchor
                if not promoted and rolling.perf < champion_anchor - rb_thr:
                    bad_steps += 1
                    if bad_steps >= rb_patience and prev_model is not None:
                        champion, champion_val = prev_model, prev_val
                        prev_model = None
                        ledger.rollbacks += 1
                        champion_anchor = rolling.perf
                        bad_steps = 0
                        rolled_back = True
                        audit.write("rollback", step=step,
                                    reason=f"rolling perf below anchor by {rb_thr}",
                                    perf_before=before_perf, perf_after=float(rolling.perf))
                else:
                    bad_steps = 0

                row = {
                    "step": step,
                    "perf": float(perf),
                    "perf_rolling": float(rolling.perf),
                    "ref_perf": ref_perf,
                    "error_gap": float(error_gap),
                    "psi_max": float(psi_max),
                    "psi_mean": float(feat["psi_mean"]),
                    "pred_js": float(pred_js),
                    "diagnosis": diag["kind"],
                    "diag_conf": float(diag["confidence"]),
                    "gate_approved": bool(decision.approved),
                    "gate_reason": decision.reason,
                    "action": str(decision.action.kind),
                    "retrain": bool(did_retrain),
                    "promoted": bool(promoted),
                    "rollback": bool(rolled_back),
                    "retrains": int(ledger.retrains),
                    "rollbacks": int(ledger.rollbacks),
                    "gpu_seconds_cum": float(ledger.total.gpu_seconds),
                    "labels_cum": int(ledger.total.labels),
                    "tokens_cum": int(ledger.total.tokens),
                    "unit_gain_cost": float(ledger.unit_gain_cost(rolling.perf - ref_perf)),
                    "kill_switch": guard.kill.status,
                }
                if task == "regression":
                    row["mae"] = float(mae(yb, pred))
                guard.note_step_ok()
            except Exception as exc:  # degrade: freeze the champion, keep the stream
                degraded_steps += 1
                guard.note_step_error(exc)
                audit.write("degraded", step=step, error=f"{type(exc).__name__}: {exc}")
                perf = float("nan")
                row = {
                    "step": step,
                    "perf": perf,
                    "perf_rolling": float(rolling.perf),
                    "ref_perf": ref_perf,
                    "error_gap": float("nan"),
                    "psi_max": float("nan"),
                    "psi_mean": float("nan"),
                    "pred_js": float("nan"),
                    "diagnosis": "error",
                    "diag_conf": 0.0,
                    "gate_approved": False,
                    "gate_reason": f"degraded: {type(exc).__name__}: {exc}",
                    "action": "none",
                    "retrain": False,
                    "promoted": False,
                    "rollback": False,
                    "retrains": int(ledger.retrains),
                    "rollbacks": int(ledger.rollbacks),
                    "gpu_seconds_cum": float(ledger.total.gpu_seconds),
                    "labels_cum": int(ledger.total.labels),
                    "tokens_cum": int(ledger.total.tokens),
                    "unit_gain_cost": float(ledger.unit_gain_cost(rolling.perf - ref_perf)),
                    "kill_switch": guard.kill.status,
                    "degraded": True,
                }
                if task == "regression":
                    row["mae"] = perf

            rows.append(row)
            log = {k: row[k] for k in (
                "perf", "perf_rolling", "ref_perf", "error_gap", "psi_max", "pred_js",
                "retrains", "rollbacks", "gpu_seconds_cum", "labels_cum",
                "unit_gain_cost")}
            if "mae" in row:
                log["mae"] = row["mae"]
            logger.log(log, step=step)

            # Wall-clock kill-switch means the grader's budget is spent: stop
            # early and report what was measured so far instead of overrunning.
            if guard.kill.tripped() and guard.kill.reason.startswith("wall-clock"):
                audit.write("kill_switch_end", step=step, reason=guard.kill.status)
                break
    finally:
        logger.finish()

    post = [r["perf_rolling"] for r in rows]
    perf_values = [p for p in post if not np.isnan(p)]
    mean_perf = float(np.mean(perf_values)) if perf_values else float("nan")
    final_perf = float(perf_values[-1]) if perf_values else float("nan")
    delta = mean_perf - ref_perf
    summary = {
        "run": run_name,
        "stream": stream_name,
        "policy": policy_name,
        "task": task,
        "seed": int(seed),
        "n_steps": int(len(stream.steps)),
        "warmup_steps": warmup,
        "data": data_meta,
        "ref_perf": ref_perf,
        "mean_perf_post": mean_perf,
        "final_perf": final_perf,
        "delta_vs_ref": delta,
        "retrains": int(ledger.retrains),
        "promotions": int(promotions),
        "rollbacks": int(ledger.rollbacks),
        "degraded_steps": int(degraded_steps),
        "diagnosis_counts": dict(diag_counts),
        "cost": ledger.as_dict(),
        "unit_gain_cost_vs_ref": float(ledger.unit_gain_cost(delta)),
        "guard": guard.as_dict(),
        "audit": audit.as_dict(),
        "swanlab_mode": logger.mode,
        "swanlab_url": logger.url,
    }
    if task == "regression":
        maes = [r["mae"] for r in rows if "mae" in r and not np.isnan(r["mae"])]
        summary["mean_mae"] = float(np.mean(maes)) if maes else float("nan")
        summary["final_mae"] = float(maes[-1]) if maes else float("nan")

    (out / "metrics.json").write_text(json.dumps(_finalize(rows), ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "cost.json").write_text(json.dumps(_finalize(ledger.as_dict()), ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "summary.json").write_text(json.dumps(_finalize(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "guard.json").write_text(json.dumps(_finalize(guard.as_dict()), ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "config.yaml").write_text(yaml.safe_dump(_finalize(cfg), allow_unicode=True, sort_keys=False), encoding="utf-8")
    if logger.url:
        (out / "swanlab_run.txt").write_text(logger.url, encoding="utf-8")

    audit.write("run_end", steps=int(len(rows)), mean_perf=mean_perf,
                retrains=int(ledger.retrains), promotions=int(promotions),
                rollbacks=int(ledger.rollbacks), degraded_steps=int(degraded_steps),
                kill_switch=guard.kill.status,
                weighted_cost=float(ledger.cost()),
                unit_gain_cost=float(summary["unit_gain_cost_vs_ref"]))
    return summary
