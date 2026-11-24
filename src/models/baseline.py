"""Model factory and time-ordered fit/validate.

CPU-only by design: the official task states the data is small and ships
pre-extracted features, so a tree ensemble is enough for the vertical slice.

Two tasks:
- classification -> HistGradientBoostingClassifier, score = accuracy
- regression     -> HistGradientBoostingRegressor,  score = -MAE (higher better)
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import (HistGradientBoostingClassifier,
                              HistGradientBoostingRegressor)

from src.eval.metrics import accuracy, mae


@dataclass
class TrainResult:
    model: object
    val_score: float      # higher-is-better: accuracy or -MAE
    val_mae: float        # only meaningful for regression, else NaN
    seconds: float
    n_labels: int

    @property
    def val_acc(self) -> float:  # historical alias
        return self.val_score


def build_model(cfg: dict, seed: int = 0, task: str = "classification"):
    kind = cfg.get("kind", "hgb")
    common = dict(
        max_iter=cfg.get("max_iter", 60),
        learning_rate=cfg.get("learning_rate", 0.12),
        max_leaf_nodes=cfg.get("max_leaf_nodes", 15),
        min_samples_leaf=cfg.get("min_samples_leaf", 20),
        l2_regularization=cfg.get("l2_regularization", 0.1),
        random_state=cfg.get("random_state", 0) + seed,
    )
    if task == "regression":
        if kind not in ("hgb", "hgb_regressor"):
            raise ValueError(f"unsupported regression model kind: {kind}")
        return HistGradientBoostingRegressor(
            loss=cfg.get("loss", "squared_error"), **common)
    if kind != "hgb":
        raise ValueError(f"unsupported classification model kind: {kind}")
    return HistGradientBoostingClassifier(**common)


def score_model(task: str, model, X, y) -> tuple[float, float]:
    pred = model.predict(X)
    if task == "regression":
        m = mae(y, pred)
        return -m, m
    return accuracy(y, pred), float("nan")


def fit_model(X: np.ndarray, y: np.ndarray, cfg: dict, seed: int = 0,
              task: str = "classification") -> TrainResult:
    """Fit on the first 80% (time order) for validation, then refit on all.

    The reported val_score is from the holdout tail, so it reflects the *newest*
    regime the model has seen - which is what a challenger should be judged on.
    """
    t0 = time.perf_counter()
    n = len(X)
    split = max(int(n * 0.8), 1)
    model = build_model(cfg, seed, task)
    if n >= 50 and split < n:
        model.fit(X[:split], y[:split])
        val_score, val_mae = score_model(task, model, X[split:], y[split:])
        model = build_model(cfg, seed, task)
        model.fit(X, y)
    else:
        model.fit(X, y)
        val_score, val_mae = score_model(task, model, X, y)
    return TrainResult(model=model, val_score=val_score, val_mae=val_mae,
                       seconds=time.perf_counter() - t0, n_labels=n)
