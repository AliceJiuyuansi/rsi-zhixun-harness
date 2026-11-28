"""Contest harness entry: one round = read history -> measure -> decide -> act.

Platform contract (SDK 1.0.4, protocol `contest.harness.workspace.v2`):

    train(rnd)    called once per round
    predict(sample, model_dir)   called later, one sample per stdin line

The v2 controller lives in `src/harness/round_controller.py`; this file is the
thin I/O shell: read the mount, call the model, pass the action through the
Gatekeeper, continue the parent checkpoint, and finish.

Ordering rules that are easy to get wrong:
  1. training data only comes from the platform mount (/data) - nothing local;
  2. the model must be the parent checkpoint *continued* (weights + Adam + scalers);
  3. every round needs at least one real model call, or `finish` is rejected;
  4. any failure degrades to "keep the parent model" and still finishes the round;
  5. state saved this round is what the next round sees - keep it small and honest.

Naming: variables are spelled out (`previous_state`, `candidate_dir`,
`round_result`) because this file is read more often than it is written.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

PLATFORM_APP_DIR = Path(os.environ.get("CONTEST_APP", "/app"))
if str(PLATFORM_APP_DIR) not in sys.path:
    sys.path.insert(0, str(PLATFORM_APP_DIR))

from src.harness.guard import Gatekeeper  # noqa: E402
from src.harness.late_budget import (  # noqa: E402
    execution_record,
    pace_c_budget,
    plan_c_budget,
    protect_c_attempt,
)
from src.harness.model_receipt import decide_with_receipt  # noqa: E402
from src.harness.round_controller import (  # noqa: E402
    acceptance,
    choose_budget,
    clip_regression_prediction,
    decision_record,
    feature_profile,
    js_divergence,
    metric_value,
    paired_acceptance,
    prediction_profile,
    psi_against,
    row_observation,
    score_gap,
    summarise_history,
)
from src.models.official_round import OfficialRound  # noqa: E402
from src.policy.whitelist import NONE_ACTION  # noqa: E402
from src.schema import Action, GateDecision  # noqa: E402
from src.utils.audit import AuditLog  # noqa: E402
from vendor.contest_baselines.data import flatten, read_dataset  # noqa: E402
from vendor.contest_baselines.model import Model  # noqa: E402

SUPPORTED_TASKS = ("classification", "regression")
HARNESS_ENTRY = os.environ.get("CONTEST_HARNESS", str(PLATFORM_APP_DIR / "harness.py"))
PREDICT_COMMAND = [sys.executable, HARNESS_ENTRY, "predict"]
WORKSPACE = Path(os.environ.get("CONTEST_WORK", "/workspace"))
STATE_DIR = Path(os.environ.get("CONTEST_STATE_DIR", "/state"))
EARLY_STOP_PATIENCE = 2
MAX_UPDATES = 10                 # lifetime cap inside one 10-round dataset run
# 版本号用于交付记录；预测策略仍随检查点保存。
RELEASE = os.environ.get("RSI_RELEASE", "0.17")
ULTRA_COMPACT_RELEASES = ("0.17",)
B_CALIBRATION_RELEASES = ("0.17",)
C_LATE_BUDGET_RELEASES = ("0.17",)
COMPACT_MODEL_RELEASES = ("0.15", "0.16") + ULTRA_COMPACT_RELEASES
RES_MLP_V2_RELEASES = ("0.14",) + COMPACT_MODEL_RELEASES
PAIRED_RELEASES = ("0.9", "0.9.1", "0.9.2") + RES_MLP_V2_RELEASES


def calibration_enabled(release, override=None):
    """显式0/1优先；未指定时采用对应版本默认值。"""
    if override is None:
        return release in ("0.7", "0.7.1", "0.8") + PAIRED_RELEASES
    return override.strip().lower() not in ("0", "false", "no", "off", "")


def calibration_mode(release, override=None):
    """C 校准器模式：显式 RSI_C_CAL_MODE 优先，否则按 RELEASE 选择。"""
    if override is not None and override.strip():
        mode = override.strip().lower()
        if mode in ("v8", "spline_v2", "res_mlp_v1", "res_mlp_v2"):
            return mode
    if release == "0.9.1":
        return "spline_v2"
    if release == "0.9.2":
        return "res_mlp_v1"
    if release in RES_MLP_V2_RELEASES:
        return "res_mlp_v2"
    return "v8"


C_CALIBRATION_ENABLED = calibration_enabled(RELEASE, os.environ.get("RSI_C_CALIBRATION"))
C_CALIBRATION_MODE = calibration_mode(RELEASE, os.environ.get("RSI_C_CAL_MODE"))
STATE_SCHEMA = 3


def log_to_stderr(message: str) -> None:
    """stdout belongs to the protocol; human logs go to stderr."""
    print(f"[submission] {message}", file=sys.stderr, flush=True)


def truncate_utf8(text: str, limit: int = 3900) -> str:
    """Keep the protocol's 4000-*byte* limit even for Chinese text."""
    raw = (text or "").encode("utf-8")
    if len(raw) <= limit:
        return text or ""
    return raw[:limit].decode("utf-8", "ignore")


def read_previous_state() -> dict:
    """Native state saved by the previous round (/state is read-only)."""
    for name in ("harness_state.json", "state.json"):
        try:
            payload = json.loads((STATE_DIR / name).read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(payload, dict):
            return payload
    return {}


def find_parent_checkpoint(rnd, root=None) -> Path:
    """Locate the parent checkpoint without assuming its exact file name."""
    declared_paths = list(rnd.parent.declared_paths())
    numpy_files = [path for path in declared_paths if path.endswith(".npz")]
    preferred = [path for path in numpy_files if path.endswith("checkpoint.npz")]
    candidates = preferred or numpy_files
    if not candidates:
        raise FileNotFoundError(f"no .npz checkpoint among declared parent files: {declared_paths}")
    return rnd.parent.path(candidates[0], root=root)


def _metric_state(previous: dict, before: dict, task: str) -> dict:
    """Best/last values in native units, plus the gap the policy reacts to."""
    name, value, higher = metric_value(task, before)
    if previous:
        best = float(previous.get("best", value))
        last = float(value)
        best = max(best, last) if higher else min(best, last)
        baseline = float(previous.get("baseline", best))
    else:
        best = last = baseline = float(value)
    return {
        "metric": name,
        "higher_is_better": higher,
        "before": float(value),
        "best": float(best),
        "last": float(last),
        "baseline": float(baseline),
        "gap": score_gap(best, last, higher),
    }


def _model_prompt(payload: dict, *, compact: bool = False, ultra_compact: bool = False) -> str:
    if ultra_compact:
        rows = payload.get("rows") or {}
        drift = payload.get("drift") or {}
        rb = payload.get("rule_budget") or {}
        tiny = {
            "d": payload.get("dataset"),
            "r": payload.get("round"),
            "st": payload.get("stage"),
            "m": round(float(payload.get("parent", 0.0)), 4),
            "best": round(float(payload.get("best", 0.0)), 4),
            "gap": round(float(payload.get("gap", 0.0)), 4),
            "psi": round(float(drift.get("psi_vs_previous", 0.0)), 3),
            "chg": int(bool(rows.get("stage_changed"))),
            "grow": int(bool(rows.get("train_grew"))),
            "stl": int(payload.get("stalled", 0)),
            "lg": round(float(payload.get("last_gain", 0.0)), 4),
            "act": rb.get("action"),
            "ep": int(rb.get("epochs", 0)),
            "lr": rb.get("lr", 0.0),
        }
        return (
            '/no_think\n仅输出一行JSON:{"decision":"train|hold","epochs":0-5,"lr":0.002}\n'
            + json.dumps(tiny, ensure_ascii=False, separators=(",", ":"))
        )
    if compact:
        state = {key: value for key, value in payload.items() if key != "history_evidence"}
        state["rule_budget"] = {key: payload["rule_budget"][key]
                                for key in ("action", "epochs", "lr", "rule")}
        return (
            '/no_think\n根据状态和规则预算，仅输出一行 JSON。保持父模型时输出 '
            '{"decision":"hold"}。训练时填写 decision="train"、epochs 和 lr。'
            '优先沿用规则值，有依据时可减少轮数。\n'
            + json.dumps(state, ensure_ascii=False, separators=(",", ":"))
        )
    return (
        "/no_think\n"
        "你是工业模型自进化系统的决策器。下面是一轮的真实状态（JSON）。\n"
        "请第一行直接输出唯一一行 JSON，不要思考过程，不要解释，不要 Markdown：\n"
        '{"decision":"train|hold","epochs":0-3,"lr":0.0005-0.003,"reason":"不超过60字"}\n'
        "规则：新阶段数据到达且有漂移时可 train；连续多轮无收益、或没有新数据且指标稳定时 hold；"
        "绝不建议超过规则预算的 epochs。\n"
        "状态：\n" + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    )


def train(rnd) -> None:
    round_started_at = time.time()
    task_contract = rnd.task or {}
    task = task_contract.get("task", "classification")
    if task not in SUPPORTED_TASKS:
        raise ValueError(f"unsupported task {task!r}")

    WORKSPACE.mkdir(parents=True, exist_ok=True)
    audit = AuditLog(WORKSPACE / f"audit-round-{rnd.round_index}.jsonl",
                     run=f"{rnd.dataset_key}-r{rnd.round_index}")
    previous_state = read_previous_state()
    platform_history = rnd.history()          # RPC: records of completed rounds
    history = summarise_history(platform_history, previous_state, task)

    # --- 1. the parent model (always a declared file, never /app) ---------
    parent_root_override = os.environ.get("CONTEST_PARENT_ROOT") or None  # local rehearsal only
    parent_checkpoint = find_parent_checkpoint(rnd, root=parent_root_override)
    parent_model = Model.load(str(parent_checkpoint))
    class_names = list(parent_model.meta["classes"])
    feature_paths = list(parent_model.meta["paths"])

    # --- 2. public data, schema-checked against the parent checkpoint -----
    training_dataset = read_dataset(rnd.data.train_path, task, class_names or None,
                                    expected_paths=feature_paths)
    use_training_panel = RELEASE == "0.17" and rnd.dataset_key == "A" and task == "classification"
    panel_ok = True
    if use_training_panel:
        from src.harness.a_training_panel import partition
        training_dataset, validation_dataset, panel_ok = partition(training_dataset)
    else:
        validation_dataset = read_dataset(rnd.data.validation_path, task, class_names or None,
                                          expected_paths=feature_paths)
    train_rows = int(len(training_dataset["ids"]))
    validation_rows = int(len(validation_dataset["ids"]))

    trainer = OfficialRound()
    validation_groups = (validation_dataset["groups"]
                         if rnd.dataset_key in ("B", "C") else None)
    validation_before = trainer.evaluate(parent_model, validation_dataset["x"],
                                         validation_dataset["y"], groups=validation_groups)
    metric_state = _metric_state(previous_state, validation_before, task)

    # --- 3. drift: against last stage, against round 1, and train vs val --
    previous_profile = previous_state.get("profile")
    baseline_profile = previous_state.get("baseline_profile") or previous_profile
    current_profile = feature_profile(training_dataset["x"])
    psi_prev = psi_against(previous_profile, training_dataset["x"])
    psi_baseline = psi_against(baseline_profile, training_dataset["x"])
    psi_holdout = psi_against(current_profile, validation_dataset["x"])

    previous_predictions = parent_model.predict(validation_dataset["x"])
    prediction_now = prediction_profile(
        previous_predictions, task, class_names,
        reference_edges=(previous_state.get("pred_profile") or {}).get("edges"))
    previous_pred_shares = (previous_state.get("pred_profile") or {}).get("shares")
    pred_js = js_divergence(previous_pred_shares, prediction_now.get("shares")
                            if isinstance(prediction_now, dict) else None)

    stage_id = str(getattr(rnd, "stage_id", "") or "")
    stage_changed = bool(stage_id) and stage_id != str(previous_state.get("stage_id", ""))
    if stage_changed:
        # Validation distributions differ across stages; an all-time best from
        # the previous stage is not comparable. Start the new stage's record at
        # the current parent, otherwise "best - current" looks like damage that
        # was never observed within this stage.
        metric_state["best"] = metric_state["before"]
        metric_state["gap"] = 0.0
    row_state = row_observation(previous_state, train_rows, validation_rows, int(rnd.round_index))
    data_grew = row_state["data_grew"]
    history["kill_switch"] = bool(previous_state.get("kill_switch"))

    # --- 4. rule budget, then let the model narrow it ---------------------
    use_v17 = RELEASE in ULTRA_COMPACT_RELEASES
    use_c_late = str(rnd.dataset_key) == "C" and RELEASE in C_LATE_BUDGET_RELEASES
    idle_rounds = int(previous_state.get("idle_rounds", 0))
    budget = choose_budget(
        dataset_key=str(rnd.dataset_key), task=task, round_index=int(rnd.round_index),
        psi_prev=psi_prev, psi_holdout=max(psi_holdout, psi_baseline if not previous_state else 0.0),
        data_grew=data_grew, stage_changed=stage_changed, gap=metric_state["gap"],
        history=history,
        recover_stalled_b=RELEASE in ("0.16", "0.17"),
        quiet_stall_a=use_v17)
    if use_c_late:
        budget = plan_c_budget(
            budget,
            idle_rounds=idle_rounds,
            kill_switch=bool(previous_state.get("kill_switch")),
            idle_threshold=3,
            min_calib_gain=0.005,
        )
    model_state = {
        "round": rnd.round_index, "stage": stage_id, "dataset": rnd.dataset_key,
        "task": task, "metric": metric_state["metric"],
        "parent": round(metric_state["before"], 6),
        "best": round(metric_state["best"], 6),
        "gap": round(metric_state["gap"], 6),
        "rows": {"train": train_rows, "validation": validation_rows,
                 "train_grew": data_grew, "stage_changed": stage_changed},
        "drift": {"psi_vs_previous": round(psi_prev, 4),
                  "psi_vs_baseline": round(psi_baseline, 4),
                  "psi_train_vs_validation": round(psi_holdout, 4),
                  "prediction_js": round(pred_js, 4)},
        "history": history["recent"],
        "history_evidence": history["evidence"],
        "last_gain": history["last_gain"], "stalled": history["stalled"],
        "rule_budget": budget,
    }
    compact_prompt = RELEASE in COMPACT_MODEL_RELEASES
    if use_v17:
        model_options = {"max_tokens": 64, "enable_thinking": False}
    elif compact_prompt:
        model_options = {"max_tokens": 256, "enable_thinking": False}
    else:
        model_options = {}
    decision, model_text, model_diagnostics = decide_with_receipt(
        rnd, _model_prompt(model_state, compact=compact_prompt, ultra_compact=use_v17),
        budget, history, int(rnd.round_index), **model_options)
    audit.write("model_receipt", step=int(rnd.round_index), **model_diagnostics)
    # The host may start with an older state. Recover its spent budget from the
    # platform's round records before assigning the next request.
    prior_spend = sum(int((r.get("configuration") or {}).get("epochs") or 0)
                      for r in (platform_history or []) if isinstance(r, dict))
    prior_spend = max(prior_spend, int(previous_state.get("requested_epochs_total", 0)))
    if use_c_late:
        decision = protect_c_attempt(decision, budget)
        decision = pace_c_budget(
            decision, spent=prior_spend, round_index=int(rnd.round_index),
            limit=12, reserve_epochs=3, release_round=8,
        )
    else:
        task_limit = {"A": 28, "B": 25, "C": 13}.get(str(rnd.dataset_key))
        if task_limit is not None and decision["epochs"] > max(0, task_limit - prior_spend):
            remaining = max(0, task_limit - prior_spend)
            decision.update(epochs=remaining, action="finetune" if remaining else "none",
                            rule="epoch_limit", reason="Remaining per-task epoch budget", source="budget")
    online = None
    if use_training_panel:
        from src.models.a_online import prepare, fit_round
        inventory, mask, online = prepare(training_dataset, previous_state, int(rnd.round_index),
                                          getattr(rnd, "frame", {}).get("run_id", ""), class_names)
        if not online["eligible"]:
            decision.update(action="none", epochs=0, rule="no_supported_new_labels", source="data")
        elif decision["epochs"]:
            decision["epochs"] = 1  # One closed-form fit, no Adam epoch.
    if use_training_panel and not panel_ok:
        decision.update(action="none", epochs=0, rule="no_independent_training_panel", source="data")
    log_to_stderr(f"round {rnd.round_index}: rows={train_rows} psi_prev={psi_prev:.3f} "
                  f"stage_changed={stage_changed} rule={budget['rule']} "
                  f"model={decision.get('model_action')} source={decision['source']} "
                  f"model_status={model_diagnostics['status']} budget={decision['epochs']}ep")

    # --- 5. controlled execution: kill-switch / whitelist / budget / bounds
    gate = Gatekeeper(
        cfg={"policy": {"max_retrains": MAX_UPDATES, "max_retrains_per_window": 0,
                        "amplitude_factor": 2.0, "max_consecutive_failures": 3}},
        audit=audit, root=WORKSPACE, executable=("none", "finetune"))
    old_kill = previous_state.get("kill_state") or {}
    gate.kill.consecutive_failures = int(old_kill.get("consecutive_failures", 0))
    gate.kill.total_failures = int(old_kill.get("total_failures", 0))
    if previous_state.get("kill_switch"):
        gate.kill.stopped = True
        gate.kill.reason = old_kill.get("reason") or "latched in a prior round"
    gate.limiter.last = dict(previous_state.get("bounds") or {})
    gate.budget.steps = [int(item) for item in (previous_state.get("budget_steps") or [])
                         if isinstance(item, (int, float))]
    if decision["epochs"] > 0:
        proposed = Action("finetune", {"epochs": int(decision["epochs"]),
                                       "lr": round(float(decision["lr"]), 6)},
                          est_cost=float(decision["epochs"]), est_gain=float(history["last_gain"]))
    else:
        proposed = NONE_ACTION
    gated = gate.review(GateDecision(True, decision["reason"], proposed),
                        step=int(rnd.round_index), retrains=int(previous_state.get("updates", 0)))
    if not gated.approved:
        log_to_stderr(f"gate rejected: {gated.reason}; keeping parent")
        decision.update(action="none", epochs=0, lr=0.0, rule="gate_reject",
                        source="gate",
                        reason=f"闸门拒绝：{gated.reason}；保持父模型")
    elif decision["epochs"] > 0:
        # the amplitude limiter may clip params; never let it *increase* epochs
        gated_epochs = int(gated.action.params.get("epochs", decision["epochs"]))
        decision["epochs"] = max(1, min(int(decision["epochs"]), gated_epochs))
        decision["lr"] = float(gated.action.params.get("lr", decision["lr"]))

    # --- 6. continue the parent, then decide whether to submit -----------
    candidate_dir = WORKSPACE / f"candidate-round-{rnd.round_index}"
    shutil.rmtree(candidate_dir, ignore_errors=True)
    round_result = None
    training_error = ""
    outcome = None
    calibrate_c = (C_CALIBRATION_ENABLED and rnd.dataset_key == "C"
                   and task == "regression" and len(feature_paths) == 24)
    calibrate_b = (RELEASE in B_CALIBRATION_RELEASES and rnd.dataset_key == "B"
                   and task == "regression")
    c_nasa_tol = 0.0015 if use_v17 else 0.0
    use_paired = RELEASE in PAIRED_RELEASES
    if int(decision["epochs"]) > 0:
        try:
            log_to_stderr(f"training: epochs={decision['epochs']} lr={decision['lr']} "
                          f"calibrate_c={calibrate_c} calibrate_b={calibrate_b} "
                          f"mode={C_CALIBRATION_MODE if calibrate_c else 'none'}")
            if use_training_panel:
                round_result = fit_round(parent_checkpoint, training_dataset, validation_dataset, mask,
                                         candidate_dir, stage_id, online)
            else:
                round_result = trainer.continue_round(
                parent_checkpoint,
                training_dataset["x"], training_dataset["y"],
                validation_dataset["x"], validation_dataset["y"],
                out_dir=candidate_dir,
                epochs=int(decision["epochs"]),
                learning_rate=float(decision["lr"]),
                patience=EARLY_STOP_PATIENCE,
                calibrate_c=calibrate_c,
                calibrate_b=calibrate_b,
                fast_c_calibration=use_v17,
                c_nasa_tol=c_nasa_tol,
                calibration_mode=C_CALIBRATION_MODE,
                selection_task=str(rnd.dataset_key) if use_paired else "",
                smooth_weights=True,
                train_groups=training_dataset.get("groups"),
                validation_groups=validation_groups,
                stage_id=stage_id or f"round-{rnd.round_index}",
                hypothesis=decision["reason"],
                change_summary=json.dumps({"rule": decision["rule"],
                                           "signals": decision["signals"]},
                                          ensure_ascii=False))
            decide_promotion = paired_acceptance if use_paired else acceptance
            outcome = decide_promotion(
                round_result.metrics_before, round_result.metrics_after,
                task, str(rnd.dataset_key), c_nasa_tol=c_nasa_tol,
            )
            gate.commit(gated.action, step=int(rnd.round_index))
            gate.note_step_ok()
            audit.write("round_result", step=int(rnd.round_index), rule=decision["rule"],
                        epochs=decision["epochs"], outcome=outcome,
                        metrics_after=round_result.metrics_after)
        except Exception as exc:  # noqa: BLE001 - degradation is part of the contract
            gate.note_step_error(exc)
            training_error = f"{type(exc).__name__}: {exc}"
            log_to_stderr(f"training failed, keeping the parent model: {training_error}")
            shutil.rmtree(candidate_dir, ignore_errors=True)
            outcome = None
    else:
        training_error = "no training this round (policy or gate chose none)"

    selected_artifact_id = rnd.parent.artifact_id
    promoted = bool(outcome and outcome.get("promote"))
    if promoted and round_result is not None:
        code_snapshot_dir = candidate_dir / "code"
        code_snapshot_dir.mkdir(exist_ok=True)
        for source_file in (Path(__file__), PLATFORM_APP_DIR / "src/policy/strategy.py",
                            PLATFORM_APP_DIR / "src/harness/round_controller.py",
                            PLATFORM_APP_DIR / "src/harness/late_budget.py",
                            PLATFORM_APP_DIR / "src/harness/model_receipt.py",
                            PLATFORM_APP_DIR / "src/models/official_round.py",
                            PLATFORM_APP_DIR / "vendor/contest_baselines/model.py",
                            PLATFORM_APP_DIR / "vendor/contest_baselines/data.py",
                            PLATFORM_APP_DIR / "vendor/contest_baselines/b_calibration.py",
                            PLATFORM_APP_DIR / "vendor/contest_baselines/c_calibration.py"):
            if source_file.exists():
                shutil.copyfile(source_file, code_snapshot_dir / source_file.name)
        if use_training_panel:
            shutil.copyfile(PLATFORM_APP_DIR / "src/models/a_online.py", code_snapshot_dir / "a_online.py")
            shutil.copyfile(PLATFORM_APP_DIR / "src/harness/a_training_panel.py", code_snapshot_dir / "a_training_panel.py")
        candidate_receipt = rnd.submit_candidate(
            candidate_dir,
            hypothesis=truncate_utf8(decision["reason"]),
            change_summary=(f"{decision['rule']}: epochs={decision['epochs']} lr={decision['lr']} "
                            f"before={round_result.metrics_before} after={round_result.metrics_after} "
                            f"gain={outcome['gain']}"),
            configuration={"rule": decision["rule"], "epochs": int(decision["epochs"]),
                           "lr": float(decision["lr"]), "psi_prev": round(psi_prev, 4),
                           "psi_holdout": round(psi_holdout, 4), "pred_js": round(pred_js, 4),
                           "history": history["recent"], "model_action": decision.get("model_action"),
                           "gain": outcome["gain"], "accepted": outcome["accepted"],
                           "calibrate_c": bool(round_result.config.get("calibrate_c")),
                           "promoted": True, "rows": train_rows},
            seed=int(parent_model.meta["seed"]),
            checkpoint_paths=["checkpoint.npz"],
            code_paths=[f"code/{path.name}" for path in code_snapshot_dir.iterdir()],
            predict_argv=PREDICT_COMMAND,
        )
        selected_artifact_id = candidate_receipt["artifact_id"]
    elif round_result is not None:
        log_to_stderr(f"candidate kept aside: before={round_result.metrics_before} "
                      f"after={round_result.metrics_after} outcome={outcome}")

    # --- 7. hand compact, honest state to the next round ------------------
    final_metrics = round_result.metrics_after if promoted else validation_before
    final_metric_state = _metric_state(previous_state, final_metrics, task)
    if promoted:
        selected_value = float(metric_value(task, round_result.metrics_after)[1])
        best_value = (max(metric_state["best"], selected_value) if metric_state["higher_is_better"]
                      else min(metric_state["best"], selected_value))
    else:
        selected_value = metric_state["before"]
        best_value = metric_state["best"]

    if promoted:
        stalled = 0
        last_gain = float(outcome["gain"])
        last_rejected = False
        promotions = int(previous_state.get("promotions", 0)) + 1
    elif int(decision["epochs"]) > 0:
        stalled = int(history["stalled"]) + 1
        last_gain = float(outcome["gain"]) if outcome else 0.0
        last_rejected = bool(outcome and not outcome.get("accepted"))
        promotions = int(previous_state.get("promotions", 0))
    else:
        stalled = int(history["stalled"])
        last_gain = float(history["last_gain"])
        last_rejected = bool(history.get("last_rejected", False))
        promotions = int(previous_state.get("promotions", 0))

    # predictions of whatever model will serve the next round
    if promoted and round_result is not None:
        try:
            next_model = Model.load(str(round_result.checkpoint))
            next_profile = prediction_profile(
                next_model.predict(validation_dataset["x"]), task, class_names,
                reference_edges=(prediction_now or {}).get("edges") if isinstance(prediction_now, dict) else None)
        except Exception:  # noqa: BLE001
            next_profile = prediction_now
    else:
        next_profile = prediction_now
    if next_profile:
        next_profile["round"] = int(rnd.round_index)
        next_profile["model"] = "candidate" if promoted else "parent"

    record = decision_record(
        round_index=int(rnd.round_index), stage_id=stage_id, dataset_key=str(rnd.dataset_key),
        action=decision, history=history,
        psi={"previous": round(psi_prev, 4), "baseline": round(psi_baseline, 4),
             "train_vs_validation": round(psi_holdout, 4), "prediction_js": round(pred_js, 4)},
        model_verdict=model_text, outcome=outcome, accepted=bool(outcome and outcome.get("accepted")),
        region="gate:" + ("approved" if gated.approved else gated.reason))
    records = list(previous_state.get("decisions") or [])[-9:] + [record]

    next_state = {
        "schema": STATE_SCHEMA,
        "dataset": str(rnd.dataset_key),
        "task": task,
        "metric": final_metric_state["metric"],
        "direction": "higher" if final_metric_state["higher_is_better"] else "lower",
        "round": int(rnd.round_index),
        "stage_id": stage_id or str(previous_state.get("stage_id", "")),
        "n_train": train_rows,
        "requested_epochs_total": prior_spend + int(decision["epochs"]),
        "n_validation": validation_rows,
        **row_state,
        "release": RELEASE,
        "prediction_policy": "nonnegative_unbounded",
        "model_receipt": model_diagnostics,
        "validation_selected": final_metrics,
        "baseline": round(float(previous_state.get("baseline", metric_state["before"])), 6),
        "best": round(float(best_value), 6),
        "last": round(float(selected_value), 6),
        "last_gain": round(float(last_gain), 6),
        "stalled": int(stalled),
        "idle_rounds": 0 if int(decision["epochs"]) > 0 else idle_rounds + 1,
        "last_rejected": bool(last_rejected),
        "updates": int(previous_state.get("updates", 0)) + (1 if int(decision["epochs"]) > 0 else 0),
        "promotions": int(promotions),
        "profile": current_profile,
        "baseline_profile": baseline_profile or current_profile,
        "pred_profile": next_profile or previous_state.get("pred_profile"),
        "bounds": gate.limiter.as_dict(),
        "budget_steps": list(gate.budget.steps),
        "kill_switch": bool(gate.kill.tripped()),
        "kill_state": gate.kill.as_dict(),
        "audit": audit.as_dict(),
        "audit_events": list(audit.events[-20:]),
        "decisions": records,
        "clock_seconds": round(float(previous_state.get("clock_seconds", 0.0))
                               + time.time() - round_started_at, 2),
    }
    if online is not None:
        next_state["a_fit_inventory"] = inventory
        next_state["a_online"] = online
    next_round_state_dir = WORKSPACE / f"state-round-{rnd.round_index}"
    shutil.rmtree(next_round_state_dir, ignore_errors=True)
    next_round_state_dir.mkdir(parents=True, exist_ok=True)
    (next_round_state_dir / "harness_state.json").write_text(
        json.dumps(next_state, ensure_ascii=False, indent=1, allow_nan=False), encoding="utf-8")
    (next_round_state_dir / "decisions.jsonl").write_text(
        "".join(json.dumps(item, ensure_ascii=False, allow_nan=False) + "\n" for item in records),
        encoding="utf-8")
    rnd.save_state(next_round_state_dir)

    # --- 8. finish with a reason a human can audit -----------------------
    after_text = (f"{final_metric_state['metric']} {outcome['before']}→{outcome['after']} "
                  f"gain={outcome['gain']:+.4f}" if outcome else
                  f"{final_metric_state['metric']} 保持 {metric_state['before']:.4f}")
    reason = (
        f"第{rnd.round_index}轮（{stage_id or 'unknown-stage'}）：{decision['reason']}。"
        f"历史：{history['evidence']}。"
        f"漂移：PSI(上轮)={psi_prev:.3f} PSI(基线)={psi_baseline:.3f} PSI(训/验)={psi_holdout:.3f} "
        f"预测JS={pred_js:.3f}。"
        f"模型裁决：{decision.get('model_action')}"
        f"（{(decision.get('model_reason') or model_text or '无文本，按规则执行')[:120]}）。"
        f"结果：{after_text}；{'提交新候选' if promoted else '保持父模型'}"
        f"{'；训练异常=' + training_error if training_error else ''}。"
    )
    log_to_stderr(reason[:600])
    rnd.finish(
        selected_artifact_id=selected_artifact_id,
        hypothesis=truncate_utf8(decision["reason"]),
        change_summary=(f"action={decision['action']} epochs={decision['epochs']} "
                        f"lr={decision['lr']} rule={decision['rule']} "
                        f"promoted={promoted} outcome={outcome}"),
        reason=truncate_utf8(reason),
        configuration={"rule": decision["rule"], "action": decision["action"],
                       "epochs": int(decision["epochs"]), "lr": float(decision["lr"]),
                       "psi_prev": round(psi_prev, 4), "psi_baseline": round(psi_baseline, 4),
                       "psi_holdout": round(psi_holdout, 4), "pred_js": round(pred_js, 4),
                       "rows": {"train": train_rows, "validation": validation_rows},
                       "stage_changed": stage_changed, "data_grew": data_grew,
                       "history": history["recent"], "model_action": decision.get("model_action"),
                       "source": decision["source"], "model_receipt": model_diagnostics,
                       "release": RELEASE, "prediction_policy": "nonnegative_unbounded",
                       **({"selection_audit": {
                           "validation_sha256": validation_dataset["sha256"],
                           "parent": {k: round_result.metrics_before[k] for k in ("macro_f1", "rmse", "nasa_log1p") if k in round_result.metrics_before},
                           "candidate": {k: round_result.metrics_after[k] for k in ("macro_f1", "rmse", "nasa_log1p") if k in round_result.metrics_after},
                           "smoothing": round_result.config["weight_smoothing"],
                           "component_cpu_seconds": round_result.config["component_cpu_seconds"],
                       }} if use_paired and round_result is not None else {}),
                       **({"a_online": online} if online is not None else {}),
                       "accepted": bool(outcome and outcome.get("accepted")),
                       "calibrate_c": bool(round_result and round_result.config.get("calibrate_c")),
                       "calibrate_b": bool(round_result and round_result.config.get("calibrate_b")),
                       "execution": execution_record(decision, round_result, promoted, training_error),
                       "promoted": promoted, "gain": None if not outcome else outcome["gain"],
                       "gate": gated.approved},
        seed=int(parent_model.meta["seed"]),
        history_ids=list(rnd.history_ids)[:50],
        model_call_ids=list(rnd.call_ids),
    )


# --------------------------------------------------------------- predict
_LOADED_MODELS: dict[str, Model] = {}


def load_model_once(checkpoint_path: Path) -> Model:
    key = str(checkpoint_path)
    if key not in _LOADED_MODELS:
        _LOADED_MODELS[key] = Model.load(str(checkpoint_path))
    return _LOADED_MODELS[key]


def find_artifact_checkpoint(model_dir) -> Path:
    """The checkpoint the platform mounted for prediction.

    `CONTEST_ARTIFACT_PATH` is authoritative; the directory scan is the fallback
    for a local run where that variable is not set.
    """
    from_env = os.environ.get("CONTEST_ARTIFACT_PATH")
    if from_env and Path(from_env).is_file():
        return Path(from_env)
    numpy_files = sorted(Path(model_dir).rglob("*.npz"))
    if not numpy_files:
        raise FileNotFoundError(f"no .npz found under {model_dir}")
    preferred = [path for path in numpy_files if path.name == "checkpoint.npz"]
    return (preferred or numpy_files)[0]


def predict(sample, model_dir):
    """Answer one sample. The platform reads one JSON line back before sending the next."""
    model = load_model_once(find_artifact_checkpoint(model_dir))

    _, feature_values = flatten(sample.inputs)
    if len(feature_values) != len(model.meta["paths"]):
        raise ValueError("input schema mismatch: feature count does not match checkpoint")

    output = model.predict(np.asarray([feature_values], dtype=float))
    if model.meta["task"] == "classification":
        class_scores = np.asarray(output)[0]
        best_index = int(np.argmax(class_scores))
        return [{"label": model.meta["classes"][best_index],
                 "score": float(class_scores[best_index])}]
    return {"value": clip_regression_prediction(
        float(np.asarray(output).reshape(-1)[0]), len(model.meta["paths"]))}
