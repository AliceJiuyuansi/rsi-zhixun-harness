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
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from vendor.contest_baselines.model import Model, checkpoint_digest

REPORT_FORMAT = "contest.cpu-baseline-report.v1"


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


def _primary_score(metrics: dict) -> float:
    """One number to compare epochs: macro-F1 when present, else -RMSE."""
    if "macro_f1" in metrics:
        return float(metrics["macro_f1"])
    if "rmse" in metrics:
        return -float(metrics["rmse"])
    return float(metrics.get("accuracy", 0.0))


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
    def evaluate(self, model: "Model", X: np.ndarray, y: np.ndarray) -> dict:
        pred = model.predict(np.asarray(X, dtype=float))
        if model.meta["task"] == "classification":
            labels = pred.argmax(axis=1)
            return {"accuracy": float(np.mean(labels == y)), "macro_f1": macro_f1(y, labels)}
        return regression_metrics(y, pred)

    # --- the only update primitive -------------------------------------
    def continue_round(self, parent_checkpoint: str | Path, X_train, y_train,
                       X_val, y_val, *, out_dir: str | Path, epochs: int = 3,
                       learning_rate: float | None = None, batch_size: int | None = None,
                       patience: int = 0, calibrate_c: bool = False,
                       stage_id: str = "round", hypothesis: str = "",
                       change_summary: str = "") -> RoundResult:
        """Continue the parent for up to `epochs`, keeping the best-scoring weights.

        `patience > 0` enables early stopping: after that many epochs without a
        new best validation score the round ends. This is what stops "train
        longer" from quietly returning an over-fitted model - measured on the
        gas task, fixed 5-epoch continuation lowered pseudo-hidden macro-F1 from
        0.399 to 0.339 while the validation score went up.
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

        before = self.evaluate(model, X_val, y_val)
        if learning_rate is not None:
            model.meta["learning_rate"] = float(learning_rate)
        if batch_size is not None:
            model.meta["batch_size"] = int(batch_size)
        model.validate()

        started = time.perf_counter()
        out.mkdir(parents=True, exist_ok=False)  # every round owns a fresh directory

        history: list[dict] = []
        best_score = float("-inf")
        best_validation = before
        best_snapshot: Path | None = None
        epochs_without_progress = 0
        epochs_run = 0
        for _ in range(int(epochs)):
            loss = model.fit(X_train, np.asarray(y_train), 1)[0]
            epochs_run += 1
            if calibrate_c:
                from vendor.contest_baselines.c_calibration import fit
                # Refit around this epoch's base prediction, never stack corrections.
                model.meta.pop("c_calibration", None)
                try:
                    model.meta["c_calibration"] = fit(X_train, y_train, model.predict(X_train))
                except Exception:  # noqa: BLE001 - calibration is optional; keep training
                    model.meta.pop("c_calibration", None)
            validation = self.evaluate(model, X_val, y_val)
            history.append({"epoch": int(model.meta["epochs"]), "step": int(model.meta["steps"]),
                            "loss": float(loss), "validation": validation})

            score = _primary_score(validation)
            if score > best_score + 1e-12:
                # Snapshot the weights + Adam state that actually scored best, so
                # "train longer" can never quietly hand back a worse model.
                best_score = score
                best_validation = validation
                snapshot = out / f".snapshot-{epochs_run}.npz"
                model.save(snapshot)
                if best_snapshot is not None:
                    best_snapshot.unlink(missing_ok=True)
                best_snapshot = snapshot
                epochs_without_progress = 0
            else:
                epochs_without_progress += 1

            if patience > 0 and epochs_without_progress >= patience:
                break

        seconds = time.perf_counter() - started
        checkpoint = out / "checkpoint.npz"
        if best_snapshot is not None:
            best_snapshot.replace(checkpoint)
        else:  # no epochs ran (epochs=0 is rejected earlier, but stay safe)
            model.save(checkpoint)
        after = best_validation  # the saved checkpoint is the best-scoring snapshot
        result = RoundResult(
            out_dir=out, checkpoint=checkpoint,
            checkpoint_sha=checkpoint_digest(checkpoint),
            parent_sha=checkpoint_digest(parent),
            metrics_before=before, metrics_after=after, history=history,
            seconds=seconds, epochs_before=int(model.meta["epochs"]) - epochs_run,
            epochs_after=int(model.meta["epochs"]), steps_after=int(model.meta["steps"]),
            config={"epochs_requested": int(epochs), "epochs_run": epochs_run,
                    "early_stopped": bool(epochs_run < int(epochs)),
                    "learning_rate": float(model.meta["learning_rate"]),
                    "batch_size": int(model.meta["batch_size"]),
                    "calibrate_c": bool(calibrate_c),
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
                  "learning_rate": result.config["learning_rate"],
                  "batch_size": result.config["batch_size"],
                  "calibrate_c": bool(calibrate_c),
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
