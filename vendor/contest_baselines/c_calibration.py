"""Train-only operating-condition residual and NASA risk shift for C-MAPSS."""
import numpy as np


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


def _design(z, knots):
    return np.column_stack([np.ones(len(z)), z] + [np.maximum(z-q, 0) for q in knots])


def _corrected(parent, correction, shift):
    # Bound the *sum* once, then enforce nonnegative RUL (including during fit).
    return np.maximum(0.0, parent + np.clip(correction + shift, -10.0, 10.0))


def _metrics(error):
    """RMSE and log of the NASA sum, without squaring or exponentiating a large tail."""
    scale = np.max(np.abs(error))
    if scale == 0:
        return 0.0, -np.inf
    rmse = scale * np.sqrt(np.mean((error / scale) ** 2))
    exponent = np.where(error < 0, -error / 13.0, error / 10.0)
    peak = np.max(exponent)
    # expm1(t) = exp(t) * (1 - exp(-t)); the latter factor is accurate near zero.
    nasa_scaled = np.sum(np.exp(exponent - peak) * -np.expm1(-exponent))
    log_nasa = peak + np.log(nasa_scaled) if nasa_scaled else -np.inf
    return rmse, log_nasa


def _best_shift(parent, correction, y):
    if len(y) < 64:
        return 0.0
    rmse0, log_nasa0 = _metrics(_corrected(parent, correction, 0.0) - y)
    if rmse0 == 0.0 or log_nasa0 == -np.inf:
        return 0.0
    log_denom = max(log_nasa0, 0.0)  # NASA / max(NASA0, 1)
    best_obj = 1.0 + 0.05 * np.exp(log_nasa0 - log_denom)
    best_shift, best_rmse, best_log_nasa = 0.0, rmse0, log_nasa0
    for delta in np.linspace(-3.5, 0.5, 17):
        if delta == 0:
            continue
        rmse, log_nasa = _metrics(_corrected(parent, correction, delta) - y)
        obj = rmse / rmse0 + 0.05 * np.exp(log_nasa - log_denom)
        if obj < best_obj:
            best_obj = obj
            best_shift, best_rmse, best_log_nasa = float(delta), rmse, log_nasa
    # Gate the minimizer itself, not a second-best point that passes both tests.
    if best_rmse <= 1.0015 * rmse0 and best_log_nasa - log_nasa0 < np.log(0.98):
        return best_shift
    return 0.0


def fit(x, y, parent_prediction):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    parent_prediction = np.asarray(parent_prediction, dtype=float).reshape(-1)
    if (x.ndim != 2 or x.shape[1] < 22 or not len(x)
            or len(x) != len(y) or len(x) != len(parent_prediction)):
        raise ValueError("invalid calibration data shape")
    if not all(np.all(np.isfinite(a)) for a in (x, y, parent_prediction)):
        raise ValueError("calibration data must be finite")

    moments = {}
    conditions = np.rint(x[:, 21])
    for condition in np.unique(conditions):
        sensors = x[conditions == condition, :21]
        mean, scale = sensors.mean(axis=0), sensors.std(axis=0)
        if not (np.isfinite(mean).all() and np.isfinite(scale).all()):
            raise ValueError("calibration moments are not finite")
        scale[scale < 1e-5] = 1.0
        moments[str(int(condition))] = [mean.tolist(), scale.tolist()]
    z, _ = _normalized(x, conditions, moments)
    knots = np.quantile(z, (0.1, 0.3, 0.5, 0.7, 0.9), axis=0)
    design = _design(z, knots)
    target = y - parent_prediction
    weights = np.where(target < 0.0, 1.35, 1.0)
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
        shifts[str(int(condition))] = _best_shift(parent_prediction[mask], correction[mask], y[mask])
    return {"moments": moments, "knots": knots.tolist(), "coef": coef.tolist(), "shifts": shifts}


def predict(x, parent_prediction, state):
    x = np.asarray(x, dtype=float)
    parent_prediction = np.asarray(parent_prediction, dtype=float).reshape(-1)
    conditions = np.rint(x[:, 21])
    z, known = _normalized(x, conditions, state["moments"])
    correction = _design(z, np.asarray(state["knots"])) @ np.asarray(state["coef"])
    shifts = state.get("shifts", {})  # Older checkpoints only contain the spline.
    shift = np.array([shifts.get(str(int(c)), 0.0) for c in conditions[known]])
    result = np.maximum(0.0, parent_prediction)
    result[known] = _corrected(parent_prediction[known], correction[known], shift)
    return result
