"""C-MAPSS residual calibration: spline + regime shift + endpoint corrector.

v13 merges two lines of work:

* v12 (Gemini): parent-prediction x operating-regime interaction features and an
  asymmetric correction clip that allows +20 cycles upward (fixing the
  under-estimation tail) but only -10 downward.
* v12.1.1: NASA-safe bias (no one-sided down-weighting), per-regime shift search
  on a symmetric grid, an endpoint corrector, and one final refit per round.

Legacy behaviour (``bias="legacy"``, ``interaction=False``) is kept byte-for-byte
for old checkpoints and the existing exact-value tests.
"""
import numpy as np

ENDPOINT_GRID = tuple(range(-8, 9, 2))
ENDPOINT_RMSE_TOLERANCE = 0.005   # candidate may not lose >0.5% RMSE
ENDPOINT_MIN_NASA_GAIN = 0.005    # require >=0.5% NASA improvement to enable
SHIFT_RMSE_TOLERANCE = 0.005      # v12 regime-shift budget


def _normalized(x, conditions, moments):
    sensors = np.zeros_like(x[:, :21], dtype=float)
    known = np.zeros(len(x), dtype=bool)
    for condition in np.unique(conditions[np.isfinite(conditions)]):
        pair = moments.get(str(int(condition)))
        if pair is None:
            continue
        select = conditions == condition
        sensors[select] = (x[select, :21] - pair[0]) / pair[1]
        known[select] = True
    return np.clip(sensors, -6, 6), known


def _parent_norm(parent, conditions, parent_moments):
    """Normalise the parent prediction inside each operating condition."""
    parent = np.asarray(parent, dtype=float)
    out = np.zeros(len(parent), dtype=float)
    for condition in np.unique(conditions[np.isfinite(conditions)]):
        pair = (parent_moments or {}).get(str(int(condition)))
        if pair is None:
            continue
        mask = conditions == condition
        mean, scale = float(pair[0]), max(float(pair[1]), 1e-5)
        out[mask] = (parent[mask] - mean) / scale
    return np.clip(out, -6, 6)


def _design(z, knots, *, interaction=False, p_norm=None, parent_knots=None,
            cond_order=None, conditions=None):
    columns = [np.ones(len(z)), z] + [np.maximum(z - q, 0) for q in knots]
    if interaction:
        p = np.zeros(len(z)) if p_norm is None else np.asarray(p_norm, dtype=float)
        columns.append(p.reshape(-1, 1))
        for q in (parent_knots or ()):
            columns.append(np.maximum(p - q, 0).reshape(-1, 1))
        for condition in (cond_order or ()):
            mask = (np.asarray(conditions) == condition).astype(float)
            columns.append((mask * p).reshape(-1, 1))
        columns.append(z * p.reshape(-1, 1))
    return np.column_stack(columns)


def _corrected(parent, correction, shift, lower=-10.0, upper=10.0):
    # Bound the *sum* once, then enforce nonnegative RUL (including during fit).
    return np.maximum(0.0, parent + np.clip(correction + shift, lower, upper))


def _metrics(error):
    """RMSE and log of the NASA sum, without squaring or exponentiating a large tail."""
    scale = np.max(np.abs(error))
    if scale == 0:
        return 0.0, -np.inf
    rmse = scale * np.sqrt(np.mean((error / scale) ** 2))
    exponent = np.where(error < 0, -error / 13.0, error / 10.0)
    peak = np.max(exponent)
    nasa_scaled = np.sum(np.exp(exponent - peak) * -np.expm1(-exponent))
    log_nasa = peak + np.log(nasa_scaled) if nasa_scaled else -np.inf
    return rmse, log_nasa


def _best_shift(parent, correction, y, grid=None):
    """Legacy objective: RMSE + a small NASA term (kept for old checkpoints)."""
    if len(y) < 64:
        return 0.0
    rmse0, log_nasa0 = _metrics(_corrected(parent, correction, 0.0) - y)
    if rmse0 == 0.0 or log_nasa0 == -np.inf:
        return 0.0
    log_denom = max(log_nasa0, 0.0)
    best_obj = 1.0 + 0.05 * np.exp(log_nasa0 - log_denom)
    best_shift, best_rmse, best_log_nasa = 0.0, rmse0, log_nasa0
    candidates = np.linspace(-3.5, 0.5, 17) if grid is None else np.asarray(grid)
    for delta in candidates:
        if delta == 0:
            continue
        rmse, log_nasa = _metrics(_corrected(parent, correction, delta) - y)
        obj = rmse / rmse0 + 0.05 * np.exp(log_nasa - log_denom)
        if obj < best_obj:
            best_obj = obj
            best_shift, best_rmse, best_log_nasa = float(delta), rmse, log_nasa
    if best_rmse <= 1.0015 * rmse0 and best_log_nasa - log_nasa0 < np.log(0.98):
        return best_shift
    return 0.0


def _best_shift_risk(parent, correction, y, grid, lower, upper):
    """v12-style regime shift: minimise log-NASA under a 0.5% RMSE budget."""
    if len(y) < 64:
        return 0.0
    rmse0, log0 = _metrics(_corrected(parent, correction, 0.0, lower, upper) - y)
    if rmse0 == 0.0 or log0 == -np.inf:
        return 0.0
    best_log, best_shift = log0, 0.0
    for delta in grid:
        if delta == 0:
            continue
        rmse, log_nasa = _metrics(_corrected(parent, correction, delta, lower, upper) - y)
        if rmse > rmse0 * (1.0 + SHIFT_RMSE_TOLERANCE):
            continue
        if log_nasa < best_log - 1e-12:
            best_log, best_shift = log_nasa, float(delta)
    if best_log <= log0 + np.log(0.995):
        return best_shift
    return 0.0


def _endpoint_offsets(prediction, endpoint):
    p20 = float(endpoint["p20"])
    p80 = float(endpoint["p80"])
    if not np.isfinite(p20) or not np.isfinite(p80) or p80 <= p20:
        return np.zeros(len(prediction))
    t = np.clip((np.asarray(prediction, dtype=float) - p20) / (p80 - p20), 0.0, 1.0)
    return float(endpoint["low"]) + (float(endpoint["high"]) - float(endpoint["low"])) * t


def _fit_endpoint(prediction, y):
    prediction = np.asarray(prediction, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    p20, p80 = (float(v) for v in np.quantile(prediction, (0.2, 0.8)))
    endpoint = {"p20": p20, "p80": p80, "low": 0.0, "high": 0.0, "enabled": False}
    if len(y) < 64 or p80 <= p20:
        return endpoint
    base_rmse, base_log = _metrics(prediction - y)
    if base_rmse == 0.0 or base_log == -np.inf:
        return endpoint
    best = (base_log, 0.0, 0.0, base_rmse)
    for low in ENDPOINT_GRID:
        for high in ENDPOINT_GRID:
            candidate = prediction + _endpoint_offsets(
                prediction, {"p20": p20, "p80": p80, "low": low, "high": high})
            rmse, log = _metrics(candidate - y)
            if rmse > base_rmse * (1.0 + ENDPOINT_RMSE_TOLERANCE):
                continue
            if log < best[0] - 1e-12:
                best = (log, float(low), float(high), rmse)
    log, low, high, _ = best
    improved = log <= base_log + np.log(1.0 - ENDPOINT_MIN_NASA_GAIN)
    endpoint.update(low=low if improved else 0.0, high=high if improved else 0.0,
                    enabled=bool(improved and (low or high)))
    return endpoint


def fit(x, y, parent_prediction, endpoint: bool = False, bias: str = "legacy",
        interaction: bool = False):
    """Fit spline (+ optional parent interaction), regime shift and endpoint.

    ``bias="nasa"``: symmetric residual weights, shift grid [-8, +8], asymmetric
    clip [-10, +20].  ``interaction=True``: append parent-prediction x regime
    features (v12).  Both default off for old checkpoints and exact-value tests.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    parent_prediction = np.asarray(parent_prediction, dtype=float).reshape(-1)
    if (x.ndim != 2 or x.shape[1] < 22 or not len(x)
            or len(x) != len(y) or len(x) != len(parent_prediction)):
        raise ValueError("invalid calibration data shape")
    if not all(np.all(np.isfinite(a)) for a in (x, y, parent_prediction)):
        raise ValueError("calibration data must be finite")

    conditions = np.rint(x[:, 21])
    moments = {}
    parent_moments = {}
    for condition in np.unique(conditions):
        mask = conditions == condition
        sensors = x[mask, :21]
        mean, scale = sensors.mean(axis=0), sensors.std(axis=0)
        if not (np.isfinite(mean).all() and np.isfinite(scale).all()):
            raise ValueError("calibration moments are not finite")
        scale[scale < 1e-5] = 1.0
        moments[str(int(condition))] = [mean.tolist(), scale.tolist()]
        parent_mean = float(parent_prediction[mask].mean())
        parent_scale = float(parent_prediction[mask].std()) or 1.0
        parent_moments[str(int(condition))] = [parent_mean, parent_scale]

    z, _ = _normalized(x, conditions, moments)
    knots = np.quantile(z, (0.1, 0.3, 0.5, 0.7, 0.9), axis=0)
    p_norm = _parent_norm(parent_prediction, conditions, parent_moments) if interaction else None
    parent_knots = np.quantile(p_norm, (0.2, 0.5, 0.8)).tolist() if interaction else []
    cond_order = [float(c) for c in np.unique(conditions)] if interaction else []
    design = _design(z, knots, interaction=interaction, p_norm=p_norm,
                     parent_knots=parent_knots, cond_order=cond_order,
                     conditions=conditions)

    target = y - parent_prediction
    if bias == "nasa":
        weights = np.ones(len(target))
        lower, upper = -10.0, 20.0
        shift_grid = np.linspace(-8.0, 8.0, 33)
    else:
        weights = np.where(target < 0.0, 1.35, 1.0)
        lower, upper = -10.0, 10.0
        shift_grid = None
    weighted_design = design * weights[:, None]
    penalty = np.eye(design.shape[1]) * (0.001 * weights.sum())
    normal = weighted_design.T @ design + penalty
    moment = weighted_design.T @ target
    try:
        coef = np.linalg.solve(normal, moment)
    except np.linalg.LinAlgError:
        coef = np.linalg.lstsq(normal, moment, rcond=None)[0]
    if not np.all(np.isfinite(coef)):
        raise ValueError("calibration coefficients are not finite")
    correction = design @ coef

    shifts = {}
    for condition in np.unique(conditions):
        mask = conditions == condition
        if bias == "nasa":
            shifts[str(int(condition))] = _best_shift_risk(
                parent_prediction[mask], correction[mask], y[mask], shift_grid, lower, upper)
        else:
            shifts[str(int(condition))] = _best_shift(
                parent_prediction[mask], correction[mask], y[mask], grid=shift_grid)

    shift_vector = np.array([shifts.get(str(int(c)), 0.0) for c in conditions])
    fitted = _corrected(parent_prediction, correction, shift_vector, lower, upper)
    endpoint_state = _fit_endpoint(fitted, y) if endpoint else None
    return {"moments": moments, "knots": knots.tolist(), "coef": coef.tolist(),
            "shifts": shifts, "endpoint": endpoint_state,
            "interaction": bool(interaction), "parent_moments": parent_moments,
            "parent_knots": parent_knots, "cond_order": cond_order,
            "lower_limit": lower, "upper_limit": upper}


def predict(x, parent_prediction, state):
    x = np.asarray(x, dtype=float)
    parent_prediction = np.asarray(parent_prediction, dtype=float).reshape(-1)
    conditions = np.rint(x[:, 21])
    z, known = _normalized(x, conditions, state["moments"])
    interaction = bool(state.get("interaction", False))
    p_norm = None
    if interaction:
        p_norm = _parent_norm(parent_prediction, conditions, state.get("parent_moments"))
    correction = _design(
        z, np.asarray(state["knots"]), interaction=interaction, p_norm=p_norm,
        parent_knots=state.get("parent_knots", []),
        cond_order=state.get("cond_order", []), conditions=conditions) @ np.asarray(state["coef"])
    shifts = state.get("shifts", {})
    shift = np.array([shifts.get(str(int(c)), 0.0) for c in conditions[known]])
    lower = float(state.get("lower_limit", -10.0))
    upper = float(state.get("upper_limit", 10.0))
    result = np.maximum(0.0, parent_prediction)
    result[known] = _corrected(parent_prediction[known], correction[known], shift, lower, upper)
    endpoint = state.get("endpoint")
    if endpoint and endpoint.get("enabled"):
        result = np.maximum(0.0, result + _endpoint_offsets(result, endpoint))
    return result
