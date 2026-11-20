"""Per-step and rolling metrics.

Two task types share one interface: every performance value is
higher-is-better. Classification uses accuracy, regression uses negated MAE.
"""
from __future__ import annotations

from collections import deque

import numpy as np
from sklearn.metrics import f1_score


def accuracy(y_true, y_pred) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.size == 0:
        return float("nan")
    return float((y_true == y_pred).mean())


def macro_f1(y_true, y_pred) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.size == 0:
        return float("nan")
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def mae(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.size == 0:
        return float("nan")
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def step_perf(task: str, y_true, y_pred) -> float:
    """Higher-is-better performance for one batch."""
    if task == "regression":
        return -mae(y_true, y_pred)
    return accuracy(y_true, y_pred)


class RollingMetrics:
    def __init__(self, window: int = 10, ewma_alpha: float = 0.3):
        self.window = window
        self.buf: deque[float] = deque(maxlen=window)
        self.ewma = float("nan")
        self.alpha = ewma_alpha

    def update(self, perf: float) -> None:
        self.buf.append(perf)
        self.ewma = perf if np.isnan(self.ewma) else (1 - self.alpha) * self.ewma + self.alpha * perf

    @property
    def perf(self) -> float:
        return float(np.mean(self.buf)) if self.buf else float("nan")

    # historical alias
    acc = perf
