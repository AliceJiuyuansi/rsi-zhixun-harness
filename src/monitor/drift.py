"""Two-channel drift monitoring: feature distribution (PSI) + error residual.

Diagnosis is a heuristic and is logged with its evidence; it is intentionally
transparent rather than "accurate" - the goal is an auditable decision.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.distance import jensenshannon


def psi(ref: np.ndarray, cur: np.ndarray, bins: int = 10, eps: float = 1e-6) -> float:
    ref = np.asarray(ref, dtype=float)
    cur = np.asarray(cur, dtype=float)
    if ref.size < 10 or cur.size < 10:
        return 0.0
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if edges.size < 3:
        return 0.0
    ref_pct = np.histogram(ref, bins=edges)[0].astype(float)
    cur_pct = np.histogram(cur, bins=edges)[0].astype(float)
    ref_pct = np.clip(ref_pct / max(ref_pct.sum(), 1), eps, None)
    cur_pct = np.clip(cur_pct / max(cur_pct.sum(), 1), eps, None)
    ref_pct /= ref_pct.sum()
    cur_pct /= cur_pct.sum()
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


class DriftMonitor:
    def __init__(self, psi_threshold: float = 0.25, js_threshold: float = 0.05,
                 task: str = "classification"):
        self.psi_threshold = psi_threshold
        self.js_threshold = js_threshold
        self.task = task
        self.ref_X: np.ndarray | None = None
        self.ref_pred_dist: np.ndarray | None = None
        self.n_classes = 0
        self._pred_edges: np.ndarray | None = None

    def set_reference(self, X: np.ndarray, y_pred=None) -> None:
        self.ref_X = np.asarray(X, dtype=float)
        if y_pred is not None:
            self.set_reference_predictions(y_pred)

    def _pred_hist(self, y_pred) -> np.ndarray:
        counts = np.histogram(y_pred, bins=self._pred_edges)[0].astype(float)
        total = max(counts.sum(), 1.0)
        return counts / total

    def set_reference_predictions(self, y_pred) -> None:
        y_pred = np.asarray(y_pred)
        if self.task == "regression":
            y_pred = y_pred.astype(float)
            lo, hi = float(y_pred.min()), float(y_pred.max())
            pad = max((hi - lo) * 0.05, 1e-6)
            self._pred_edges = np.linspace(lo - pad, hi + pad, 11)
            self.ref_pred_dist = self._pred_hist(y_pred)
            return
        if self.n_classes == 0:
            self.n_classes = int(y_pred.max()) + 1
            self.ref_pred_dist = np.zeros(self.n_classes)
        counts = np.bincount(y_pred, minlength=self.n_classes).astype(float)
        self.ref_pred_dist = counts / max(counts.sum(), 1)

    def feature_score(self, X: np.ndarray) -> dict:
        if self.ref_X is None:
            return {"psi_max": 0.0, "psi_mean": 0.0, "per_feature": []}
        scores = [psi(self.ref_X[:, j], X[:, j]) for j in range(self.ref_X.shape[1])]
        return {"psi_max": float(np.max(scores)), "psi_mean": float(np.mean(scores)),
                "per_feature": [round(float(s), 4) for s in scores]}

    def pred_shift(self, y_pred) -> float:
        if self.ref_pred_dist is None:
            return 0.0
        if self.task == "regression":
            dist = self._pred_hist(np.asarray(y_pred, dtype=float))
        else:
            counts = np.bincount(np.asarray(y_pred), minlength=self.n_classes).astype(float)
            dist = counts / max(counts.sum(), 1)
        return float(jensenshannon(self.ref_pred_dist, dist, base=2))

    def diagnose(self, psi_max: float, error_gap: float, pred_js: float,
                 err_threshold: float = 0.06) -> dict:
        feats = psi_max >= self.psi_threshold
        err = error_gap >= err_threshold
        pred = pred_js >= self.js_threshold
        if not feats and not err:
            kind, conf, reason = "none", 0.9, "no feature shift, no error gap"
        elif feats and not err:
            kind, conf, reason = "covariate", 0.7, "feature shift but model still holds"
        elif not feats and err:
            kind, conf, reason = "concept", 0.75, "error gap without feature shift"
        elif pred and error_gap < 1.5 * err_threshold:
            kind, conf, reason = "label", 0.55, "prediction prior shift with modest error gap"
        else:
            # Feature movement dominates; the relationship may still be learnable.
            kind, conf, reason = "covariate", 0.55, "feature shift dominates the error gap"
        return {"kind": kind, "confidence": conf, "reason": reason,
                "evidence": {"psi_max": psi_max, "error_gap": error_gap, "pred_js": pred_js}}
