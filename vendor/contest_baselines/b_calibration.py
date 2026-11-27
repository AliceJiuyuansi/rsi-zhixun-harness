"""Train-only sign-consistent log-RUL calibration for B-bearing (XJTU-SY).

Design goals (all strictly train-only, pure NumPy, deterministic, per-sample):
1. Suppress operating-condition scale & duration imbalance (two ultra-long bearings
   can account for >70% of training rows and drown out short/medium bearings) via
   inverse-bearing-count weighting in log1p(RUL) space.
2. Select vibration amplitude/energy features whose degradation correlation sign
   is consistent across training bearings (>=80% sign agreement), and enforce
   matching coefficient signs via convex projected L2 ridge so unseen bearings
   never suffer inverted correlation flips.
3. Combine with the continued parent MLP via geometric log1p blending so base
   MLP updates always flow through, with exact single-row == batch equivalence
   (<1e-12) and safe fallback to the base MLP if calibration state is absent.
"""
from __future__ import annotations

import numpy as np

VERSION = "sign_log_v1"


def _log_transform(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    return np.sign(arr) * np.log1p(np.abs(arr))


def fit(
    x,
    y,
    groups,
    *,
    alpha: float = 0.80,
    bw: float = 1.0,
    lam: float = 300.0,
    agree_thresh: float = 0.80,
    min_abs_corr: float = 0.25,
    clip_hi: float = 8.5,
    min_group_rows: int = 15,
) -> dict:
    """Fit a deterministic sign-consistent log-RUL calibrator on training data only."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    if x.ndim != 2 or len(x) != len(y) or len(x) < 32 or x.shape[1] < 2:
        raise ValueError("insufficient training data for B calibration")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)) or np.any(y < 0.0):
        raise ValueError("B calibration requires finite features and nonnegative targets")
    if groups is None or len(groups) != len(x):
        raise ValueError("B calibration requires aligned training bearing groups")

    g_arr = np.asarray(["" if g is None else str(g).strip() for g in groups], dtype=object)
    valid_groups = sorted(
        g for g in set(g_arr.tolist())
        if g and int(np.sum(g_arr == g)) >= min_group_rows and float(np.std(y[g_arr == g])) > 1e-8
    )
    if len(valid_groups) < 2:
        raise ValueError("B calibration requires at least two valid training bearings")

    feat = _log_transform(x)
    n_feat = feat.shape[1]

    # Per-bearing Pearson correlation with RUL (purely from training bearings).
    corrs = []
    for g in valid_groups:
        mask = g_arr == g
        fg = feat[mask]
        yg = y[mask]
        y_center = yg - yg.mean()
        y_norm = float(np.linalg.norm(y_center))
        if y_norm <= 1e-12:
            continue
        f_center = fg - fg.mean(axis=0, keepdims=True)
        f_norm = np.linalg.norm(f_center, axis=0)
        row = np.zeros(n_feat, dtype=float)
        nonzero = f_norm > 1e-10
        row[nonzero] = (f_center[:, nonzero].T @ y_center) / (f_norm[nonzero] * y_norm)
        corrs.append(row)
    if len(corrs) < 2:
        raise ValueError("B calibration could not compute cross-bearing correlations")

    corr_mat = np.asarray(corrs, dtype=float)
    mean_c = corr_mat.mean(axis=0)
    mean_sign = np.sign(mean_c)
    agree = (np.sign(corr_mat) == mean_sign).mean(axis=0)
    keep = (agree >= float(agree_thresh)) & (np.abs(mean_c) >= float(min_abs_corr)) & (mean_sign != 0.0)
    if int(keep.sum()) < min(3, n_feat):
        score = agree * np.abs(mean_c)
        top_k = min(6, n_feat)
        top_idx = np.argsort(-score)[:top_k]
        keep = np.zeros(n_feat, dtype=bool)
        keep[top_idx] = True

    indices = np.where(keep)[0].astype(int)
    signs = mean_sign[indices]
    f_sel = feat[:, indices]
    mu = f_sel.mean(axis=0)
    sd = f_sel.std(axis=0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    z = np.clip((f_sel - mu) / sd, -5.0, 5.0)

    # Bearing-balanced sample weights so long bearings do not drown out short bearings.
    counts = {g: int(np.sum(g_arr == g)) for g in set(g_arr.tolist()) if g}
    known_w = [float(counts[g]) ** (-float(bw)) for g in counts]
    fallback_w = float(np.mean(known_w)) if known_w else 1.0
    weights = np.array(
        [float(counts[g]) ** (-float(bw)) if g in counts else fallback_w for g in g_arr.tolist()],
        dtype=float,
    )
    weights /= float(weights.mean())

    target = np.log1p(np.maximum(0.0, y))
    design = np.column_stack([np.ones(len(z), dtype=float), z])
    w_design = design * weights[:, None]
    n_rows = float(len(z))
    hess = (w_design.T @ design) / n_rows
    rhs = (w_design.T @ target) / n_rows
    reg_diag = np.full(design.shape[1], float(lam) / n_rows, dtype=float)
    reg_diag[0] = 1e-8
    hess[np.arange(design.shape[1]), np.arange(design.shape[1])] += reg_diag

    # Convex sign-projected gradient descent (exact global optimum, <3ms).
    lip = float(np.linalg.eigvalsh(hess)[-1]) + 1e-8
    coef = np.zeros(design.shape[1], dtype=float)
    coef[0] = float((weights @ target) / weights.sum())
    neg_mask = signs < 0.0
    pos_mask = signs > 0.0
    for _ in range(300):
        coef = coef - (hess @ coef - rhs) / lip
        w_feat = coef[1:]
        w_feat[neg_mask] = np.minimum(w_feat[neg_mask], 0.0)
        w_feat[pos_mask] = np.maximum(w_feat[pos_mask], 0.0)
        coef[1:] = w_feat

    if not np.all(np.isfinite(coef)) or not np.all(np.isfinite(mu)) or not np.all(np.isfinite(sd)):
        raise ValueError("B calibration produced nonfinite parameters")

    return {
        "version": VERSION,
        "indices": [int(i) for i in indices.tolist()],
        "mu": [float(v) for v in mu.tolist()],
        "sd": [float(v) for v in sd.tolist()],
        "coef": [float(v) for v in coef.tolist()],
        "alpha": float(alpha),
        "clip_hi": float(clip_hi),
        "n_groups": len(valid_groups),
    }


def predict(x, parent_prediction, state: dict) -> np.ndarray:
    """Apply sign-consistent log-RUL calibration with exact single-row == batch output."""
    if not isinstance(state, dict) or state.get("version") != VERSION:
        raise ValueError(f"unsupported B calibration state version: {state.get('version') if isinstance(state, dict) else None}")
    x_arr = np.asarray(x, dtype=float)
    base = np.maximum(0.0, np.asarray(parent_prediction, dtype=float).reshape(-1))
    if x_arr.ndim != 2 or len(x_arr) != len(base):
        raise ValueError("B calibration prediction shape mismatch")
    indices = [int(i) for i in state["indices"]]
    mu = np.asarray(state["mu"], dtype=float)
    sd = np.asarray(state["sd"], dtype=float)
    coef = np.asarray(state["coef"], dtype=float)
    alpha = float(state["alpha"])
    clip_hi = float(state["clip_hi"])
    if not (0.0 <= alpha <= 1.0) or len(indices) != len(mu) or len(coef) != len(indices) + 1:
        raise ValueError("malformed B calibration state")
    feat = _log_transform(x_arr)[:, indices]
    z = np.clip((feat - mu) / sd, -5.0, 5.0)
    log_cal = np.clip(coef[0] + z @ coef[1:], 0.0, clip_hi)
    log_base = np.log1p(base)
    if "downward_alpha" in state:
        downward_alpha = float(state["downward_alpha"])
        if not alpha <= downward_alpha <= 1.0:
            raise ValueError("invalid downward calibration weight")
        alpha = np.where(log_cal < log_base, downward_alpha, alpha)
    blended = np.expm1(np.clip((1.0 - alpha) * log_base + alpha * log_cal, 0.0, max(clip_hi, 12.0)))
    if not np.all(np.isfinite(blended)):
        raise ValueError("nonfinite B calibration output")
    return np.maximum(0.0, blended)
