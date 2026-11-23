"""Train-only operating-condition residual for public C-MAPSS inputs."""
import numpy as np


def _conditions(x):
    return np.rint(x[:, 21]).astype(int)


def _normalized(x, moments):
    conditions = _conditions(x)
    sensors = np.zeros_like(x[:, :21], dtype=float)
    known = np.zeros(len(x), dtype=bool)
    for condition in np.unique(conditions):
        pair = moments.get(str(condition))
        if pair is None:
            continue
        select = conditions == condition
        sensors[select] = (x[select, :21] - pair[0]) / pair[1]
        known[select] = True
    return np.clip(sensors, -6, 6), known


def _design(z, knots):
    return np.column_stack([np.ones(len(z)), z] + [np.maximum(z-q, 0) for q in knots])


def fit(x, y, parent_prediction):
    moments = {}
    conditions = _conditions(x)
    for condition in np.unique(conditions):
        sensors = x[conditions == condition, :21]
        scale = sensors.std(axis=0)
        scale[scale < 1e-5] = 1.0
        moments[str(condition)] = [sensors.mean(axis=0).tolist(), scale.tolist()]
    z, _ = _normalized(x, moments)
    knots = np.quantile(z, (0.1, 0.3, 0.5, 0.7, 0.9), axis=0)
    design = _design(z, knots)
    penalty = np.eye(design.shape[1]) * (0.001 * len(y))
    target = y - parent_prediction
    normal = design.T @ design + penalty
    moment = design.T @ target
    try:
        coef = np.linalg.solve(normal, moment)
    except np.linalg.LinAlgError:
        coef = np.linalg.lstsq(normal, moment, rcond=None)[0]
    if not np.all(np.isfinite(coef)):
        raise ValueError("calibration coefficients are not finite")
    return {"moments": moments, "knots": knots.tolist(), "coef": coef.tolist()}


def predict(x, parent_prediction, state):
    z, known = _normalized(x, state["moments"])
    correction = _design(z, np.asarray(state["knots"])) @ np.asarray(state["coef"])
    return np.maximum(0, parent_prediction + np.where(known, np.clip(correction, -10, 10), 0))
