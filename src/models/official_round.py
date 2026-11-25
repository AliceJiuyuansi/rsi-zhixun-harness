"""One continuation round of the organiser's CPU baseline (tasks A / B / C).

Why this module exists: on the platform the graded artifact is a **checkpoint
produced by continuing the parent checkpoint**, not a model we invent. So the
harness needs exactly one primitive:

    continue_round(parent_ckpt, train, val, cfg) -> new checkpoint + metrics

Everything else (drift monitoring, gating, rollback, bookkeeping) decides *when*
and *how* to call it, and *whether to accept* the result. Keeping the training
mechanics here means the controller never touches numpy math or checkpoint
formats.

Contract: docs/04_工程架构与规范.md §4.2.
The vendored copy of the official package lives in `vendor/contest_baselines/`
(synced by `scripts/sync_official_baselines.py`).
"""
from __future__ import annotations

import json
import time
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from src.harness.round_controller import paired_acceptance
from vendor.contest_baselines.model import PARAMETERS, Model, checkpoint_digest

REPORT_FORMAT = "contest.cpu-baseline-report.v1"


def balanced_sample_weights(groups, exponent: float) -> np.ndarray | None:
    """Mean-one inverse frequency weights from training groups alone."""
    if groups is None or exponent == 0:
        return None
    if not 0 < exponent <= 1:
        raise ValueError("group weighting exponent must be between zero and one")
    counts = Counter(group for group in groups if group is not None and group != "")
    if len(counts) < 2:
        return None
    by_group = {group: count ** -exponent for group, count in counts.items()}
    known_mean = sum(counts[group] * weight for group, weight in by_group.items()) / sum(counts.values())
    weights = np.array([by_group.get(group, known_mean) for group in groups], dtype=float)
    return weights / weights.mean()


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Macro-F1 over the labels seen in truth or prediction (official formula)."""
    labels = sorted(set(np.asarray(y_true).tolist()) | set(np.asarray(y_pred).tolist()))
    scores = []
    for label in labels:
        tp = int(np.sum((y_pred == label) & (y_true == label)))
        fp = int(np.sum((y_pred == label) & (y_true != label)))
        fn = int(np.sum((y_pred != label) & (y_true == label)))
        denom = 2 * tp + fp + fn
        scores.append((2 * tp / denom) if denom else 0.0)
    return float(np.mean(scores)) if scores else float("nan")


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    err = np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float)
    return {"mae": float(np.mean(np.abs(err))), "rmse": float(np.sqrt(np.mean(err ** 2)))}


def nasa_metrics(y_true, y_pred) -> dict:
    """实际服务输出的逐样本 NASA 求和；对数总分保留极端尾部诊断。"""
    truth = np.asarray(y_true, dtype=float).reshape(-1)
    pred = np.asarray(y_pred, dtype=float).reshape(-1)
    if truth.shape != pred.shape or not truth.size or not np.all(np.isfinite(truth)) or not np.all(np.isfinite(pred)):
        raise ValueError("NASA requires equally sized nonempty finite arrays")
    error = pred - truth
    exponent = np.where(error < 0, -error / 13.0, error / 10.0)
    # log(expm1(x))，避免大指数溢出；x=0 的贡献为零。
    log_terms = np.full_like(exponent, -np.inf)
    positive = exponent > 0
    values = exponent[positive]
    log_terms[positive] = values + np.log(-np.expm1(-values))
    if not np.any(positive):
        return {"nasa": 0.0, "nasa_log1p": 0.0, "nasa_saturated": False}
    peak = float(np.max(log_terms))
    log_sum = peak + float(np.log(np.exp(log_terms - peak).sum()))
    limit = float(np.log(np.finfo(float).max))
    saturated = log_sum >= limit
    return {"nasa": float(np.finfo(float).max) if saturated else float(np.exp(log_sum)),
            "nasa_log1p": float(np.logaddexp(0.0, log_sum)), "nasa_saturated": saturated}


def grouped_regression_metrics(y_true, y_pred, groups) -> dict:
    """观察性设备报告；不改变父子选择与公开聚合指标。"""
    truth, pred = np.asarray(y_true).reshape(-1), np.asarray(y_pred).reshape(-1)
    if len(groups) != len(truth):
        raise ValueError("validation groups must align with targets")
    known = np.asarray([g if isinstance(g, str) and g.strip() else "" for g in groups])
    by_group = {}
    for group in sorted(set(known) - {""}):
        mask = known == group
        by_group[group] = {"rows": int(mask.sum()), **regression_metrics(truth[mask], pred[mask])}
    return {"by_group": by_group, "group_rows": int(np.sum(known != "")),
            "ungrouped_rows": int(np.sum(known == ""))}


def _primary_score(metrics: dict) -> float:
    """One number to compare epochs: macro-F1 when present, else -RMSE."""
    if "macro_f1" in metrics:
        return float(metrics["macro_f1"])
    if "rmse" in metrics:
        return -float(metrics["rmse"])
    return float(metrics.get("accuracy", 0.0))


def _refit_c(model: Model, X_train, y_train, mode: str = "v8") -> None:
    from vendor.contest_baselines.c_calibration import fit

    # Fit residuals on the uncalibrated training predictions, never on the holdout.
    model.meta.pop("c_calibration", None)
    model.meta["c_calibration"] = fit(X_train, y_train, model.predict(X_train), mode=mode)


def _fit_b_state(X_train, y_train, train_groups) -> dict | None:
    """Fit train-only sign-consistent log-RUL calibration for B-bearing if groups allow."""
    from vendor.contest_baselines import b_calibration

    try:
        return b_calibration.fit(X_train, y_train, train_groups)
    except Exception:  # noqa: BLE001 - optional calibration falls back cleanly to base MLP
        return None


@dataclass
class RoundResult:
    out_dir: Path
    checkpoint: Path
    checkpoint_sha: str
    parent_sha: str | None
    metrics_before: dict
    metrics_after: dict
    history: list[dict] = field(default_factory=list)
    seconds: float = 0.0
    epochs_before: int = 0
    epochs_after: int = 0
    steps_after: int = 0
    config: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "out_dir": str(self.out_dir),
            "checkpoint": str(self.checkpoint),
            "checkpoint_sha256": self.checkpoint_sha,
            "parent_checkpoint_sha256": self.parent_sha,
            "validation_before": self.metrics_before,
            "validation_after": self.metrics_after,
            "epochs_before": self.epochs_before,
            "epochs_after": self.epochs_after,
            "steps_after": self.steps_after,
            "training_seconds": self.seconds,
            "configuration": self.config,
        }


class OfficialRound:
    """Thin, safe wrapper around the organiser's NumPy MLP checkpoint."""

    def __init__(self, package_dir: str | Path | None = None):
        if package_dir is not None:  # allow an explicit copy (tests, other profiles)
            import sys
            sys.path.insert(0, str(Path(package_dir).resolve().parent))
            globals()["Model"] = __import__(f"{Path(package_dir).name}.model", fromlist=["Model"]).Model
            globals()["checkpoint_digest"] = __import__(f"{Path(package_dir).name}.model",
                                                          fromlist=["checkpoint_digest"]).checkpoint_digest

    # --- metrics -------------------------------------------------------
    def evaluate(self, model: "Model", X: np.ndarray, y: np.ndarray, groups=None) -> dict:
        pred = model.predict(np.asarray(X, dtype=float))
        if model.meta["task"] == "classification":
            labels = pred.argmax(axis=1)
            return {"accuracy": float(np.mean(labels == y)), "macro_f1": macro_f1(y, labels)}
        metrics = regression_metrics(y, pred)
        if len(model.meta["paths"]) == 24:
            metrics.update(nasa_metrics(y, pred))
        if groups is not None:
            metrics.update(grouped_regression_metrics(y, pred, groups))
        return metrics

    # --- the only update primitive -------------------------------------
    def continue_round(self, parent_checkpoint: str | Path, X_train, y_train,
                       X_val, y_val, *, out_dir: str | Path, epochs: int = 3,
                       learning_rate: float | None = None, batch_size: int | None = None,
                       patience: int = 0, calibrate_c: bool = False, validation_groups=None,
                       stage_id: str = "round", hypothesis: str = "",
                       change_summary: str = "", smooth_weights: bool = False,
                       selection_task: str = "", calibration_mode: str = "v8",
                       train_groups=None, group_alpha: float = 0.0,
                       weight_decay: float = 0.0, label_smoothing: float = 0.0,
                       calibrate_b: bool = False, fast_c_calibration: bool = False,
                       c_nasa_tol: float = 0.0) -> RoundResult:
        """Continue the parent for up to `epochs`, keeping the best-scoring weights.

        `patience > 0` enables early stopping: after that many epochs without a
        new best validation score the round ends. This is what stops "train
        longer" from quietly returning an over-fitted model - measured on the
        gas task, fixed 5-epoch continuation lowered pseudo-hidden macro-F1 from
        0.399 to 0.339 while the validation score went up.

        `smooth_weights` tries a blend of the best epoch (85%) and parent (15%).
        It replaces the best epoch only on a strict same-holdout improvement.
        """
        parent = Path(parent_checkpoint)
        out = Path(out_dir)
        if out.exists():
            raise ValueError(f"output must be a new directory: {out}")
        if int(epochs) < 1 or int(epochs) > 10000:
            raise ValueError("epochs must be in 1..10000")

        model = Model.load(str(parent))          # restores weights + Adam + scalers
        X_train = np.asarray(X_train, dtype=float)
        X_val = np.asarray(X_val, dtype=float)
        if X_train.ndim != 2 or X_train.shape[1] != len(model.meta["paths"]):
            raise ValueError("train features do not match the checkpoint schema")
        if X_val.shape[1] != len(model.meta["paths"]):
            raise ValueError("validation features do not match the checkpoint schema")

        if selection_task not in ("", "A", "B", "C"):
            raise ValueError("unknown snapshot selection task")
        cpu = {key: 0.0 for key in ("fit", "calibration", "validation", "smoothed_calibration", "smoothed_validation")}

        @contextmanager
        def measured(key):
            start = time.process_time()
            try:
                yield
            finally:
                cpu[key] += time.process_time() - start

        with measured("validation"):
            before = self.evaluate(model, X_val, y_val, groups=validation_groups)

        def eligible(metrics):
            return not selection_task or paired_acceptance(
                before, metrics, model.meta["task"], selection_task, c_nasa_tol=c_nasa_tol
            )["promote"]

        def improves(ok, score):
            # Lexicographic order: an eligible snapshot always beats an ineligible
            # one. If neither can deploy, retain the best for diagnosis, not serving.
            return (ok and not best_ok) or (ok == best_ok and score > best_score + 1e-12)

        epochs_before = int(model.meta["epochs"])
        steps_before = int(model.meta["steps"])
        if learning_rate is not None:
            model.meta["learning_rate"] = float(learning_rate)
        if batch_size is not None:
            model.meta["batch_size"] = int(batch_size)
        model.validate()

        started = time.perf_counter()
        out.mkdir(parents=True, exist_ok=False)  # every round owns a fresh directory

        history: list[dict] = []
        best_score = unconstrained_best = float("-inf")
        best_ok = False
        best_validation = before
        best_snapshot: Path | None = None
        epochs_without_progress = 0
        epochs_run = 0
        parent_weights = {k: model.params[k].copy() for k in PARAMETERS} if smooth_weights else None
        total_epochs = int(epochs)
        best_raw_rmse = float("inf")
        b_cal_state = None
        if calibrate_b:
            with measured("calibration"):
                b_cal_state = _fit_b_state(X_train, np.asarray(y_train, dtype=float), train_groups)
        weights = balanced_sample_weights(train_groups, group_alpha)
        fit_options = {"sample_weight": weights} if weights is not None else {}
        if weight_decay:
            fit_options["weight_decay"] = weight_decay
        if label_smoothing:
            fit_options["label_smoothing"] = label_smoothing

        if calibrate_c and fast_c_calibration:
            # Fast single-refit path for C-engine: train base MLP epochs cheaply,
            # pick best raw-RMSE epoch, then fit res_mlp_v2 ONCE (2.5-3.5x speedup).
            model.meta.pop("c_calibration", None)
            best_raw_snap: Path | None = None
            for ep_idx in range(total_epochs):
                with measured("fit"):
                    loss = model.fit(X_train, np.asarray(y_train), 1, **fit_options)[0]
                epochs_run += 1
                with measured("validation"):
                    raw_val = regression_metrics(y_val, model.predict(X_val))
                raw_rmse = float(raw_val["rmse"])
                history.append({"epoch": int(model.meta["epochs"]), "step": int(model.meta["steps"]),
                                "loss": float(loss), "validation": raw_val, "phase": "raw_search"})
                if raw_rmse < best_raw_rmse - 1e-12:
                    best_raw_rmse = raw_rmse
                    epochs_without_progress = 0
                    snap = out / f".raw-{epochs_run}.npz"
                    model.save(snap)
                    if best_raw_snap is not None:
                        best_raw_snap.unlink(missing_ok=True)
                    best_raw_snap = snap
                else:
                    epochs_without_progress += 1
                if patience > 0 and epochs_without_progress >= patience:
                    break
            if best_raw_snap is not None:
                best_model = Model.load(str(best_raw_snap))
                best_raw_snap.unlink(missing_ok=True)
            else:
                best_model = model
            try:
                with measured("calibration"):
                    _refit_c(best_model, X_train, y_train, mode=calibration_mode)
            except Exception:  # noqa: BLE001
                best_model.meta.pop("c_calibration", None)
            with measured("validation"):
                validation = self.evaluate(best_model, X_val, y_val, groups=validation_groups)
            best_ok = eligible(validation)
            best_score = _primary_score(validation)
            best_validation = validation
            best_snapshot = out / ".snapshot-fast.npz"
            best_model.save(best_snapshot)
            history.append({"epoch": int(best_model.meta["epochs"]), "step": int(best_model.meta["steps"]),
                            "loss": float(history[-1]["loss"]) if history else 0.0,
                            "validation": validation, "phase": "calibrated",
                            **({"eligible": best_ok} if selection_task else {})})
        else:
            for ep_idx in range(total_epochs):
                if calibrate_b:
                    model.meta.pop("b_calibration", None)
                with measured("fit"):
                    loss = model.fit(X_train, np.asarray(y_train), 1, **fit_options)[0]
                epochs_run += 1
                if calibrate_b and b_cal_state is not None:
                    model.meta["b_calibration"] = b_cal_state
                if calibrate_c:
                    should_refit = True
                    if calibration_mode != "v8" and total_epochs > 1:
                        # Skip expensive non-v8 refit on clearly degraded middle epochs.
                        model.meta.pop("c_calibration", None)
                        raw_rmse = regression_metrics(y_val, model.predict(X_val))["rmse"]
                        if ep_idx == 0 or ep_idx == total_epochs - 1 or raw_rmse <= best_raw_rmse * 1.005:
                            best_raw_rmse = min(best_raw_rmse, raw_rmse)
                        else:
                            should_refit = False
                    if should_refit:
                        try:
                            with measured("calibration"):
                                _refit_c(model, X_train, y_train, mode=calibration_mode)
                        except Exception:  # noqa: BLE001 - calibration is optional; keep training
                            model.meta.pop("c_calibration", None)
                with measured("validation"):
                    validation = self.evaluate(model, X_val, y_val, groups=validation_groups)
                ok = eligible(validation)
                history.append({"epoch": int(model.meta["epochs"]), "step": int(model.meta["steps"]),
                                "loss": float(loss), "validation": validation,
                                **({"eligible": ok} if selection_task else {})})

                score = _primary_score(validation)
                if improves(ok, score):
                    # Snapshot the weights + Adam state that actually scored best, so
                    # "train longer" can never quietly hand back a worse model.
                    best_score = score
                    best_validation = validation
                    snapshot = out / f".snapshot-{epochs_run}.npz"
                    model.save(snapshot)
                    if best_snapshot is not None:
                        best_snapshot.unlink(missing_ok=True)
                    best_snapshot = snapshot
                    best_ok = ok
                # Keep the original primary-score patience and optimizer trajectory;
                # only change which observed snapshot can be returned.
                if score > unconstrained_best + 1e-12:
                    unconstrained_best = score
                    epochs_without_progress = 0
                else:
                    epochs_without_progress += 1

                if patience > 0 and epochs_without_progress >= patience:
                    break

        smoothing = "not_attempted"
        if best_snapshot is not None and parent_weights is not None:
            # Start from the best epoch, including its unchanged Adam state and counters.
            smoothed = Model.load(str(best_snapshot))
            for key in PARAMETERS:
                smoothed.params[key] = 0.85 * smoothed.params[key] + 0.15 * parent_weights[key]
            skip_smooth = False
            if calibrate_c and fast_c_calibration:
                smoothed.meta.pop("c_calibration", None)
                with measured("smoothed_validation"):
                    smooth_raw_rmse = float(regression_metrics(y_val, smoothed.predict(X_val))["rmse"])
                if smooth_raw_rmse > best_raw_rmse * 1.001:
                    skip_smooth = True
                    smoothing = "skipped_raw_worse"
            if not skip_smooth:
                try:
                    if calibrate_b and b_cal_state is not None:
                        smoothed.meta["b_calibration"] = b_cal_state
                    if calibrate_c:
                        with measured("smoothed_calibration"):
                            _refit_c(smoothed, X_train, y_train, mode=calibration_mode)
                    with measured("smoothed_validation"):
                        validation = self.evaluate(smoothed, X_val, y_val, groups=validation_groups)
                except (ValueError, ArithmeticError, np.linalg.LinAlgError):
                    # A bad optional fit must not discard an already valid trained snapshot.
                    smoothing = "invalid_candidate"
                else:
                    smoothing = "kept_original"
                    if improves(eligible(validation), _primary_score(validation)):
                        snapshot = out / ".snapshot-smoothed.npz"
                        smoothed.save(snapshot)
                        snapshot.replace(best_snapshot)
                        best_validation = validation
                        smoothing = "selected"

        seconds = time.perf_counter() - started
        checkpoint = out / "checkpoint.npz"
        if best_snapshot is not None:
            best_snapshot.replace(checkpoint)
        else:  # no epochs ran (epochs=0 is rejected earlier, but stay safe)
            model.save(checkpoint)
        after = best_validation  # the saved checkpoint is the best-scoring snapshot
        selected_model = Model.load(str(checkpoint))
        result = RoundResult(
            out_dir=out, checkpoint=checkpoint,
            checkpoint_sha=checkpoint_digest(checkpoint),
            parent_sha=checkpoint_digest(parent),
            metrics_before=before, metrics_after=after, history=history,
            seconds=seconds, epochs_before=epochs_before,
            epochs_after=int(selected_model.meta["epochs"]), steps_after=int(selected_model.meta["steps"]),
            config={"epochs_requested": int(epochs), "epochs_run": epochs_run,
                    "early_stopped": bool(epochs_run < int(epochs)),
                    "weight_smoothing": smoothing, "selection_task": selection_task,
                    "component_cpu_seconds": cpu,
                    "executed_steps": int(model.meta["steps"]) - steps_before,
                    "executed_final_epochs": int(model.meta["epochs"]),
                    "executed_final_steps": int(model.meta["steps"]),
                    "learning_rate": float(model.meta["learning_rate"]),
                    "batch_size": int(model.meta["batch_size"]),
                    "calibrate_c": bool(calibrate_c),
                    "calibrate_b": bool(calibrate_b and b_cal_state is not None),
                    "fast_c_calibration": bool(calibrate_c and fast_c_calibration),
                    "calibration_mode": calibration_mode if calibrate_c else "",
                    "stage_id": stage_id, "hypothesis": hypothesis,
                    "change_summary": change_summary},
        )
        report = {"format": REPORT_FORMAT, "profile": model.meta.get("profile"),
                  "task": model.meta["task"], "purpose": "local_harness",
                  "device": "cpu", "stage_id": stage_id, "seed": model.meta["seed"],
                  "parameter_count": model.meta["parameter_count"],
                  "input_features": len(model.meta["paths"]),
                  "train_rows": int(len(X_train)), "validation_rows": int(len(X_val)),
                  "parent_checkpoint_sha256": result.parent_sha,
                  "checkpoint_sha256": result.checkpoint_sha,
                  "initial_epochs": result.epochs_before, "final_epochs": result.epochs_after,
                  "final_steps": result.steps_after, "training_seconds": seconds,
                  "executed_final_epochs": result.config["executed_final_epochs"],
                  "executed_final_steps": result.config["executed_final_steps"],
                  "learning_rate": result.config["learning_rate"],
                  "batch_size": result.config["batch_size"],
                  "calibrate_c": bool(calibrate_c),
                  "weight_smoothing": smoothing, "selection_task": selection_task,
                  "component_cpu_seconds": cpu,
                  "validation_before": before, "validation_after": after,
                  "hypothesis": hypothesis, "change_summary": change_summary}
        (out / "training.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        with (out / "history.jsonl").open("w", encoding="utf-8") as fh:
            for record in history:
                fh.write(json.dumps({"stage_id": stage_id, "seed": model.meta["seed"], **record},
                                    allow_nan=False) + "\n")
        return result

    # --- inference -----------------------------------------------------
    def predict(self, checkpoint: str | Path, X) -> np.ndarray:
        model = Model.load(str(checkpoint))
        return model.predict(np.asarray(X, dtype=float))

    def evaluate_checkpoint(self, checkpoint: str | Path, X, y) -> dict:
        model = Model.load(str(checkpoint))
        return self.evaluate(model, np.asarray(X, dtype=float), np.asarray(y))
