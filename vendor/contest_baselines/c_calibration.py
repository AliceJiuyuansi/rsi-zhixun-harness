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


def paper_weights(y, parent_prediction, threshold_cycles=10.0, weight_cap_cycles=100.0):
    """论文Eq.10的难样本权重；截顶与均值归一化是本项目的受控适配。"""
    if not np.isfinite(threshold_cycles) or threshold_cycles <= 0:
        raise ValueError("threshold_cycles must be finite and positive")
    if not np.isfinite(weight_cap_cycles) or weight_cap_cycles < threshold_cycles:
        raise ValueError("weight cap must be finite and >= threshold")
    truth = np.asarray(y, dtype=float).reshape(-1)
    parent = np.asarray(parent_prediction, dtype=float).reshape(-1)
    if truth.shape != parent.shape or not truth.size:
        raise ValueError("training truth and predictions must align")
    error = np.abs(truth - parent)
    if not np.all(np.isfinite(error)):
        raise ValueError("training errors must be finite")
    raw = np.where(error < threshold_cycles, error / 2.0, error)
    raw = np.minimum(raw, weight_cap_cycles)
    mean = float(np.mean(raw))
    return raw / mean if mean > 0 else np.ones_like(raw)


def nasa_log_objective(coef, design, truth, parent, limit, ridge_strength):
    """固定n时log(mean(exp(a)))与NASA排序相同；另加L2控制校准参数。"""
    correction = design @ coef
    shifted = parent + np.clip(correction, -limit, limit)
    prediction = np.maximum(0, shifted)
    error = prediction - truth
    exponent = np.where(error < 0, -error / 13.0, error / 10.0)
    peak = float(exponent.max())
    exp_shifted = np.exp(exponent - peak)
    softmax = exp_shifted / exp_shifted.sum()
    value = peak + float(np.log(exp_shifted.mean())) + .5 * ridge_strength * float(coef @ coef)
    derivative = np.where(error < 0, -1/13.0, np.where(error > 0, 1/10.0, 0.0))
    active = (np.abs(correction) < limit) & (shifted > 0)
    gradient = design.T @ (softmax * derivative * active) + ridge_strength * coef
    return value, gradient


def _fit_nasa_log(design, truth, parent, limit, ridge_strength, steps):
    coef = np.zeros(design.shape[1])
    value, gradient = nasa_log_objective(coef, design, truth, parent, limit, ridge_strength)
    initial = value
    accepted_steps = 0
    for _ in range(steps):
        norm2 = float(gradient @ gradient)
        if norm2 < 1e-16:
            break
        step_size = 1.0
        for _ in range(12):
            proposed = coef - step_size * gradient
            candidate, candidate_gradient = nasa_log_objective(proposed, design, truth, parent, limit, ridge_strength)
            if np.isfinite(candidate) and candidate <= value - 1e-4 * step_size * norm2:
                coef, value, gradient = proposed, candidate, candidate_gradient
                accepted_steps += 1
                break
            step_size *= .5
        else:
            break
    return coef, {"objective": "logmeanexp_NASA_exponent_plus_L2", "initial": initial,
                  "final": value, "accepted_steps": accepted_steps, "max_steps": steps}


def fit(x, y, parent_prediction, *, method="ridge", threshold_cycles=10.0,
        weight_cap_cycles=100.0, ridge_strength=0.001, correction_limit_cycles=10.0,
        optimization_steps=40):
    """每轮只用训练数据重新拟合；权重固定于该轮未校准MLP的训练残差。"""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    parent_prediction = np.asarray(parent_prediction, dtype=float).reshape(-1)
    if x.ndim != 2 or x.shape != (len(y), 24) or y.shape != parent_prediction.shape or not len(y):
        raise ValueError("expected aligned nonempty 24-feature training arrays")
    if not all(np.all(np.isfinite(v)) for v in (x, y, parent_prediction)):
        raise ValueError("training arrays must be finite")
    if method not in ("ridge", "paper_dw", "nasa_log"):
        raise ValueError("unknown calibration method")
    if not np.isfinite(ridge_strength) or ridge_strength <= 0:
        raise ValueError("ridge strength must be finite and positive")
    if not np.isfinite(correction_limit_cycles) or not 0 < correction_limit_cycles <= 40:
        raise ValueError("correction limit must stay within the predeclared 40-cycle bound")
    if not isinstance(optimization_steps, int) or not 1 <= optimization_steps <= 100:
        raise ValueError("optimization_steps must be in 1..100")
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
    penalty = np.eye(design.shape[1]) * (ridge_strength * len(y))
    target = y - parent_prediction
    weights = (paper_weights(y, parent_prediction, threshold_cycles, weight_cap_cycles)
               if method == "paper_dw" else np.ones(len(y)))
    if method == "paper_dw":
        # 等价于 Phi.T W Phi；原地加权避免再分配一个全量设计矩阵。
        root_weight = np.sqrt(weights)
        design *= root_weight[:, None]
        target = target * root_weight
    optimization = {}
    if method == "nasa_log":
        coef, optimization = _fit_nasa_log(design, y, parent_prediction,
                                           correction_limit_cycles, ridge_strength, optimization_steps)
    else:
        normal = design.T @ design + penalty
        moment = design.T @ target
        try:
            coef = np.linalg.solve(normal, moment)
        except np.linalg.LinAlgError:
            coef = np.linalg.lstsq(normal, moment, rcond=None)[0]
    if not np.all(np.isfinite(coef)):
        raise ValueError("calibration coefficients are not finite")
    return {"moments": moments, "knots": knots.tolist(), "coef": coef.tolist(),
            "schema": 2, "method": method, "correction_limit_cycles": float(correction_limit_cycles),
            "optimization": optimization,
            "fit": {"train_rows": len(y), "threshold_cycles": float(threshold_cycles),
                    "weight_cap_cycles": float(weight_cap_cycles), "ridge_strength": float(ridge_strength),
                    "weight_min": float(weights.min()), "weight_max": float(weights.max()),
                    "effective_rows": float(weights.sum() ** 2 / np.sum(weights ** 2))}}


def predict(x, parent_prediction, state):
    z, known = _normalized(x, state["moments"])
    correction = _design(z, np.asarray(state["knots"])) @ np.asarray(state["coef"])
    limit = float(state.get("correction_limit_cycles", 10.0))
    if not np.isfinite(limit) or not 0 < limit <= 40:
        raise ValueError("invalid persisted calibration correction limit")
    return np.maximum(0, parent_prediction + np.where(known, np.clip(correction, -limit, limit), 0))
