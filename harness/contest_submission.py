"""Contest harness entry: one round = read parent -> continue -> submit -> finish.

Platform contract (SDK 1.0.4, protocol `contest.harness.workspace.v2`):

    train(rnd)    called once per round
    predict(sample, model_dir)   called later, one sample per stdin line

Ordering rules that are easy to get wrong:
  1. training data only comes from the platform mount (/data) - nothing local;
  2. the model must be the parent checkpoint *continued* (weights + Adam + scalers);
  3. every round needs at least one real model call, or `finish` is rejected;
  4. any failure degrades to "keep the parent model" and still finishes the round.

Naming: variables are spelled out (`previous_run`, `candidate_dir`,
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

from src.models.official_round import OfficialRound  # noqa: E402
from src.policy.strategy import Signal, choose, diagnose, explain  # noqa: E402
from vendor.contest_baselines.data import flatten, read_dataset  # noqa: E402
from vendor.contest_baselines.model import Model  # noqa: E402

SUPPORTED_TASKS = ("classification", "regression")
HARNESS_ENTRY = os.environ.get("CONTEST_HARNESS", str(PLATFORM_APP_DIR / "harness.py"))
PREDICT_COMMAND = [sys.executable, HARNESS_ENTRY, "predict"]

# A candidate is only accepted when it is at least this close to the parent.
ACCEPT_MARGIN = 0.02
FIRST_ROUND_NEW_EPOCHS = 5      # round 1: make a real attempt on the shipped baseline
FIRST_ROUND_NEW_LEARNING_RATE = 2e-3
FRESH_DATA_EPOCHS = 3           # later rounds, when the platform reveals new data
FRESH_DATA_LEARNING_RATE = 1.5e-3
QUIET_ROUND_EPOCHS = 1          # drift quiet: barely touch the weights
QUIET_ROUND_LEARNING_RATE = 1e-3
EARLY_STOP_PATIENCE = 2         # stop after 2 epochs with no new best validation score


def log_to_stderr(message: str) -> None:
    """stdout belongs to the protocol; human logs go to stderr."""
    print(f"[submission] {message}", file=sys.stderr, flush=True)


def read_previous_state() -> dict:
    """Native state saved by the previous round (/state is read-only)."""
    try:
        return json.loads((Path("/state") / "harness_state.json").read_text(encoding="utf-8"))
    except Exception:
        return {}


def feature_drift_psi(reference_rows: np.ndarray, current_rows: np.ndarray, bins: int = 10) -> float:
    """Largest PSI across features: how far the current inputs moved from the reference.

    PSI < 0.10 stable, 0.10-0.25 mild, > 0.25 clear drift (industry rule of thumb).
    """
    reference_rows = np.asarray(reference_rows, dtype=float)
    current_rows = np.asarray(current_rows, dtype=float)
    if len(reference_rows) < 10 or len(current_rows) < 10:
        return 0.0
    per_feature_scores = []
    for column in range(reference_rows.shape[1]):
        bin_edges = np.unique(np.quantile(reference_rows[:, column], np.linspace(0, 1, bins + 1)))
        if bin_edges.size < 3:
            continue
        reference_share = np.histogram(reference_rows[:, column], bins=bin_edges)[0].astype(float)
        current_share = np.histogram(current_rows[:, column], bins=bin_edges)[0].astype(float)
        reference_share = np.clip(reference_share / max(reference_share.sum(), 1), 1e-6, None)
        current_share = np.clip(current_share / max(current_share.sum(), 1), 1e-6, None)
        reference_share /= reference_share.sum()
        current_share /= current_share.sum()
        per_feature_scores.append(
            float(np.sum((current_share - reference_share) * np.log(current_share / reference_share))))
    return float(max(per_feature_scores)) if per_feature_scores else 0.0


def decide_training_budget(round_index: int, previous_run: dict, training_rows: int,
                           drift_psi: float) -> tuple[dict, str]:
    """How many epochs at what learning rate this round, plus the reason.

    Three cases, cheapest first:
      - kill-switch latched     -> do not touch the model
      - round 1                 -> a real attempt, so later rounds start from something
      - new public data arrived -> spend more (the platform revealed a new stage)
      - otherwise               -> drift-band table decides
    """
    if previous_run.get("kill_switch"):
        return {"epochs": 0}, "kill-switch was latched by an earlier round"

    if round_index <= 1:
        return ({"epochs": FIRST_ROUND_NEW_EPOCHS, "lr": FIRST_ROUND_NEW_LEARNING_RATE},
                "round 1: continue the shipped baseline with a real training attempt")

    best_validation_score = float(previous_run.get("best_val", 0.0))
    last_validation_score = float(previous_run.get("val", best_validation_score or 0.0))
    score_gap = max(best_validation_score - last_validation_score, 0.0)
    data_grew = training_rows > int(previous_run.get("n_train", 0))

    action = choose(Signal(psi=drift_psi, gap=score_gap,
                           retrains=int(previous_run.get("updates", 0))))
    if action.kind == "none":
        epochs = FRESH_DATA_EPOCHS if data_grew else QUIET_ROUND_EPOCHS
        rate = FRESH_DATA_LEARNING_RATE if data_grew else QUIET_ROUND_LEARNING_RATE
        return ({"epochs": epochs, "lr": rate},
                "no action needed ("
                + explain(action, Signal(psi=drift_psi, gap=score_gap))
                + "); cheap calibration only")

    epochs = int(action.params.get("epochs", FRESH_DATA_EPOCHS))
    rate = float(action.params.get("lr", FRESH_DATA_LEARNING_RATE))
    return ({"epochs": epochs, "lr": rate},
            explain(action, Signal(psi=drift_psi, gap=score_gap)))


def find_parent_checkpoint(rnd, root=None) -> Path:
    """Locate the parent checkpoint without assuming its exact file name.

    The platform declares `checkpoint_paths` in the parent artifact; today it is
    `checkpoint.npz`, but hard-coding that would break the moment the organiser
    nests it (`baseline/checkpoint.npz`). We therefore look at what was declared.
    """
    declared_paths = list(rnd.parent.declared_paths())
    numpy_files = [path for path in declared_paths if path.endswith(".npz")]
    preferred = [path for path in numpy_files if path.endswith("checkpoint.npz")]
    candidates = preferred or numpy_files
    if not candidates:
        raise FileNotFoundError(f"no .npz checkpoint among declared parent files: {declared_paths}")
    return rnd.parent.path(candidates[0], root=root)


def train(rnd) -> None:
    round_started_at = time.time()
    task_contract = rnd.task or {}
    task = task_contract.get("task", "classification")
    if task not in SUPPORTED_TASKS:
        raise ValueError(f"unsupported task {task!r}")

    # --- 1. the parent model (always a declared file, never /app) ---------
    parent_root_override = os.environ.get("CONTEST_PARENT_ROOT") or None  # local rehearsal only
    parent_checkpoint = find_parent_checkpoint(rnd, root=parent_root_override)
    parent_model = Model.load(str(parent_checkpoint))
    class_names = list(parent_model.meta["classes"])
    feature_paths = list(parent_model.meta["paths"])

    # --- 2. public data, schema-checked against the parent checkpoint -----
    training_dataset = read_dataset(rnd.data.train_path, task, class_names or None,
                                    expected_paths=feature_paths)
    validation_dataset = read_dataset(rnd.data.validation_path, task, class_names or None,
                                      expected_paths=feature_paths)
    drift_psi = feature_drift_psi(training_dataset["x"], validation_dataset["x"])
    previous_run = read_previous_state()

    trainer = OfficialRound()
    validation_before = trainer.evaluate(parent_model, validation_dataset["x"],
                                         validation_dataset["y"])
    budget, budget_reason = decide_training_budget(
        rnd.round_index, previous_run, len(training_dataset["ids"]), drift_psi)
    log_to_stderr(
        f"round {rnd.round_index}: rows={len(training_dataset['ids'])} psi={drift_psi:.3f} "
        f"before={validation_before} budget={budget} why={budget_reason}")

    # --- 3. the model call the platform requires (never skipped) ----------
    state_summary = json.dumps({
        "dataset": rnd.dataset_key,
        "round": rnd.round_index,
        "rows": {"train": len(training_dataset["ids"]),
                 "validation": len(validation_dataset["ids"])},
        "previous_best": previous_run.get("best_val"),
        "validation_before": validation_before,
        "drift": {"psi_max": round(drift_psi, 4),
                  "diagnosis": diagnose(Signal(psi=drift_psi, gap=0.0))},
        "proposed_budget": budget,
    }, ensure_ascii=False)
    model_receipt = rnd.model(
        [{"role": "user",
          "content": "工业模型本轮状态如下（JSON）。请用一句话判断：继续训练、保持父模型，还是回滚。\n"
                     + state_summary}],
        max_tokens=min(256, rnd.max_output_tokens), temperature=0)
    model_verdict = (model_receipt.text or "").strip()[:400]
    log_to_stderr(f"model call {model_receipt.call_id}: {model_verdict[:120]}")

    # --- 4. continue the parent, then decide whether to submit -----------
    workspace_dir = Path(os.environ.get("CONTEST_WORK", "/workspace"))
    candidate_dir = workspace_dir / f"candidate-round-{rnd.round_index}"
    shutil.rmtree(candidate_dir, ignore_errors=True)  # continue_round creates it itself

    round_result = None
    training_error = ""
    candidate_is_better = False
    try:
        if int(budget.get("epochs", 0)) > 0:
            round_result = trainer.continue_round(
                parent_checkpoint,
                training_dataset["x"], training_dataset["y"],
                validation_dataset["x"], validation_dataset["y"],
                out_dir=candidate_dir,
                epochs=int(budget["epochs"]),
                learning_rate=float(budget.get("lr", FRESH_DATA_LEARNING_RATE)),
                patience=EARLY_STOP_PATIENCE,
                stage_id=f"round-{rnd.round_index}",
                hypothesis=budget_reason,
                change_summary=json.dumps(budget, ensure_ascii=False))
            score_before = _primary_score(round_result.metrics_before)
            score_after = _primary_score(round_result.metrics_after)
            candidate_is_better = score_after >= score_before - ACCEPT_MARGIN
        else:
            training_error = "no training this round (kill-switch or do-nothing decision)"
    except Exception as exc:  # noqa: BLE001 - degradation is part of the contract
        training_error = f"{type(exc).__name__}: {exc}"
        log_to_stderr(f"training failed, keeping the parent model: {training_error}")
        shutil.rmtree(candidate_dir, ignore_errors=True)

    selected_artifact_id = rnd.parent.artifact_id
    if round_result is not None and candidate_is_better:
        code_snapshot_dir = candidate_dir / "code"
        code_snapshot_dir.mkdir(exist_ok=True)
        for source_file in (Path(__file__), PLATFORM_APP_DIR / "src/policy/strategy.py"):
            if source_file.exists():
                shutil.copyfile(source_file, code_snapshot_dir / source_file.name)
        candidate_receipt = rnd.submit_candidate(
            candidate_dir,
            hypothesis=budget_reason[:4000],
            change_summary=(f"continue {budget['epochs']} epochs at lr={budget.get('lr')} "
                            f"from parent; before={round_result.metrics_before} "
                            f"after={round_result.metrics_after}")[:4000],
            configuration=dict(budget, psi=drift_psi, model_verdict=model_verdict[:200]),
            seed=int(parent_model.meta["seed"]),
            checkpoint_paths=["checkpoint.npz"],
            code_paths=[f"code/{path.name}" for path in code_snapshot_dir.iterdir()],
            predict_argv=PREDICT_COMMAND,
        )
        selected_artifact_id = candidate_receipt["artifact_id"]
    elif round_result is not None:
        log_to_stderr(f"candidate not accepted: before={round_result.metrics_before} "
                      f"after={round_result.metrics_after}")

    # --- 5. hand state to the next round, then finish ---------------------
    next_round_state_dir = workspace_dir / f"state-round-{rnd.round_index}"
    next_round_state_dir.mkdir(parents=True, exist_ok=True)
    final_score = _primary_score(round_result.metrics_after if round_result
                                 else validation_before)
    (next_round_state_dir / "harness_state.json").write_text(json.dumps({
        "n_train": len(training_dataset["ids"]),
        "val": final_score,
        "best_val": max(float(previous_run.get("best_val", 0.0)), final_score),
        "updates": int(previous_run.get("updates", 0)) + (1 if round_result else 0),
        "last_round": rnd.round_index,
        "clock_seconds": round(float(previous_run.get("clock_seconds", 0.0))
                               + time.time() - round_started_at, 2),
        "kill_switch": bool(previous_run.get("kill_switch")) or os.environ.get("RSI_STOP", "") == "1",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    rnd.save_state(next_round_state_dir)

    rnd.finish(
        selected_artifact_id=selected_artifact_id,
        hypothesis=budget_reason[:4000],
        change_summary=(f"round {rnd.round_index}: {json.dumps(budget, ensure_ascii=False)}; "
                        f"accepted={candidate_is_better}; error={training_error or 'none'}")[:4000],
        reason=(model_verdict or "no model text")[:4000],
        configuration=dict(budget, accepted=candidate_is_better, psi=round(drift_psi, 4)),
        seed=int(parent_model.meta["seed"]),
        history_ids=[item.get("id", "") for item in rnd.history()][:50],
        model_call_ids=list(rnd.call_ids),
    )


def _primary_score(metrics: dict) -> float:
    """One number to compare rounds: macro-F1 for classification, -RMSE otherwise."""
    if "macro_f1" in metrics:
        return float(metrics["macro_f1"])
    if "rmse" in metrics:
        return -float(metrics["rmse"])
    return float(metrics.get("accuracy", 0.0))


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
    return {"value": float(max(0.0, float(np.asarray(output).reshape(-1)[0])))}
