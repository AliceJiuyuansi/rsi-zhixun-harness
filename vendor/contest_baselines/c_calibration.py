"""Train-only operating-condition residual and NASA risk calibration for C-MAPSS.

Supported calibration modes (selected at training time; inference always dispatches
on ``state.get("version")`` stored inside the checkpoint so serving never depends
on host environment variables):

- ``"v8"`` (legacy default, no ``version`` key):
  Weighted piecewise-linear spline with symmetric +-10 clip and [-3.5, 0.5] shift.
- ``"spline_v2"`` (Version A - deterministic analytical):
  Per-condition normalized sensors + parent prediction spline & per-condition
  interactions, asymmetric clip [-80, +40], bidirectional per-condition NASA shift.
- ``"res_mlp_v1"`` (Version B - lightweight residual MLP):
  Condition-aware 2-hidden-layer NumPy MLP (32, 16) trained with Tempered
  Entropic-NASA + MSE composite loss, overestimation guard, asymmetric clip
  [-120, +42], and per-condition NASA risk shift.
"""
import numpy as np

SUPPORTED_MODES = ("v8", "spline_v2", "res_mlp_v1")


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
    return np.column_stack([np.ones(len(z)), z] + [np.maximum(z - q, 0) for q in knots])


def _corrected(parent, correction, shift, neg_clip=10.0, pos_clip=10.0):
    # Bound the *sum* once, then enforce nonnegative RUL (including during fit).
    return np.maximum(0.0, parent + np.clip(correction + shift, -float(neg_clip), float(pos_clip)))


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


def _best_shift_asym(parent, correction, y, neg_clip, pos_clip,
                     *, nasa_alpha=0.15, shift_lo=-12.0, shift_hi=12.0, rmse_tol=1.005):
    if len(y) < 64:
        return 0.0
    rmse0, log_nasa0 = _metrics(_corrected(parent, correction, 0.0, neg_clip, pos_clip) - y)
    if rmse0 == 0.0 or log_nasa0 == -np.inf:
        return 0.0
    log_denom = max(log_nasa0, 0.0)
    best_obj = 1.0 + nasa_alpha * np.exp(log_nasa0 - log_denom)
    best_shift, best_rmse, best_log_nasa = 0.0, rmse0, log_nasa0
    for delta in np.linspace(shift_lo, shift_hi, 49):
        if delta == 0.0:
            continue
        rmse, log_nasa = _metrics(_corrected(parent, correction, delta, neg_clip, pos_clip) - y)
        obj = rmse / rmse0 + nasa_alpha * np.exp(min(log_nasa - log_denom, 20.0))
        if obj < best_obj:
            best_obj = obj
            best_shift, best_rmse, best_log_nasa = float(delta), rmse, log_nasa
    if best_rmse <= rmse_tol * rmse0 and best_log_nasa - log_nasa0 < np.log(0.98):
        return best_shift
    return 0.0


def _validate_inputs(x, y, parent_prediction):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    parent_prediction = np.asarray(parent_prediction, dtype=float).reshape(-1)
    if (x.ndim != 2 or x.shape[1] < 22 or not len(x)
            or len(x) != len(y) or len(x) != len(parent_prediction)):
        raise ValueError("invalid calibration data shape")
    if not all(np.all(np.isfinite(a)) for a in (x, y, parent_prediction)):
        raise ValueError("calibration data must be finite")
    return x, y, parent_prediction


def _fit_moments(x, conditions):
    moments = {}
    for condition in np.unique(conditions):
        sensors = x[conditions == condition, :21]
        mean, scale = sensors.mean(axis=0), sensors.std(axis=0)
        if not (np.isfinite(mean).all() and np.isfinite(scale).all()):
            raise ValueError("calibration moments are not finite")
        scale[scale < 1e-5] = 1.0
        moments[str(int(condition))] = [mean.tolist(), scale.tolist()]
    return moments


def _fit_v8(x, y, parent_prediction):
    conditions = np.rint(x[:, 21])
    moments = _fit_moments(x, conditions)
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


def _spline_v2_design(z_ext, cond_int, cond_keys, knots):
    cols = [np.ones(len(z_ext)), z_ext] + [np.maximum(z_ext - q, 0.0) for q in knots]
    p_col = z_ext[:, -1]
    for ck in cond_keys:
        ind = (cond_int == int(ck)).astype(float)
        cols.append(ind)
        cols.append(ind * p_col)
        cols.append(ind * np.maximum(p_col, 0.0))
    return np.column_stack(cols)


def _fit_spline_v2(x, y, parent_prediction):
    neg_clip, pos_clip = 80.0, 40.0
    over_weight, ridge = 1.5, 0.002
    conditions = np.rint(x[:, 21])
    moments = _fit_moments(x, conditions)
    cond_keys = sorted(moments.keys(), key=int)
    p_mu = float(np.mean(parent_prediction))
    p_sd = max(float(np.std(parent_prediction)), 1.0)

    z, _ = _normalized(x, conditions, moments)
    p_norm = np.clip((parent_prediction - p_mu) / p_sd, -4.0, 6.0)
    z_ext = np.column_stack([z, p_norm])
    knots = np.quantile(z_ext, (0.1, 0.3, 0.5, 0.7, 0.9), axis=0)
    cond_int = conditions.astype(int)
    design = _spline_v2_design(z_ext, cond_int, cond_keys, knots)

    target = y - parent_prediction
    weights = np.where(target < 0.0, over_weight, 1.0)
    weighted_design = design * weights[:, None]
    penalty = np.eye(design.shape[1]) * (ridge * weights.sum())
    penalty[0, 0] = 1e-6
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
    for ck in cond_keys:
        mask = cond_int == int(ck)
        shifts[ck] = _best_shift_asym(
            parent_prediction[mask], correction[mask], y[mask], neg_clip, pos_clip
        )
    return {
        "version": "spline_v2",
        "moments": moments,
        "cond_keys": cond_keys,
        "p_mu": p_mu,
        "p_sd": p_sd,
        "knots": knots.tolist(),
        "coef": coef.tolist(),
        "shifts": shifts,
        "neg_clip": neg_clip,
        "pos_clip": pos_clip,
    }


def _fit_res_mlp_v1(x, y, parent_prediction):
    hidden = (32, 16)
    epochs, lr, wd, bs = 10, 7e-3, 8e-4, 4096
    gamma, nasa_w, nasa_scale, t_cap, over_boost = 0.55, 0.50, 2.3, 22.0, 1.65
    neg_clip, pos_clip = 120.0, 42.0

    conditions = np.rint(x[:, 21])
    moments = _fit_moments(x, conditions)
    cond_keys = sorted(int(k) for k in moments)
    cond_int = conditions.astype(int)
    active = [
        i for i in range(21)
        if any(moments[str(c)][1][i] != 1.0 or float(np.std(x[:, i])) > 1e-5 for c in cond_keys)
    ]
    if not active:
        active = list(range(21))

    p_mu = float(np.mean(parent_prediction))
    p_sd = max(float(np.std(parent_prediction)), 1.0)
    y_sd = max(float(np.std(y)), 1.0)

    z, _ = _normalized(x, conditions, moments)
    ck_map = {c: idx for idx, c in enumerate(cond_keys)}
    coh = np.zeros((len(x), len(cond_keys)), dtype=float)
    for c in cond_keys:
        coh[cond_int == c, ck_map[c]] = 1.0
    p_norm = np.clip((parent_prediction - p_mu) / p_sd, -5.0, 6.0)[:, None]
    x_feat = np.column_stack([z[:, active], coh, p_norm]).astype(np.float32)
    y32 = y.astype(np.float32)
    p32 = parent_prediction.astype(np.float32)

    rng = np.random.default_rng(42)
    dims = [x_feat.shape[1], *hidden, 1]
    params = {}
    for i in range(len(dims) - 1):
        params[f"w{i}"] = rng.normal(0.0, np.sqrt(1.0 / dims[i]), (dims[i], dims[i + 1])).astype(np.float32)
        params[f"b{i}"] = np.zeros(dims[i + 1], dtype=np.float32)
    last = len(dims) - 2
    params[f"w{last}"] *= np.float32(0.05)
    m1 = {k: np.zeros_like(v) for k, v in params.items()}
    m2 = {k: np.zeros_like(v) for k, v in params.items()}
    n_layers = len(dims) - 1
    step = 0
    ys32 = np.float32(y_sd)

    for ep in range(epochs):
        lr_now = np.float32(lr * (0.08 + 0.92 * 0.5 * (1.0 + np.cos(np.pi * ep / max(epochs, 1)))))
        order = rng.permutation(len(x_feat))
        for start in range(0, len(order), bs):
            idx = order[start:start + bs]
            xb, yb, pb = x_feat[idx], y32[idx], p32[idx]
            acts = [xb]
            for i in range(n_layers - 1):
                acts.append(np.tanh(acts[-1] @ params[f"w{i}"] + params[f"b{i}"]))
            out = (acts[-1] @ params[f"w{last}"] + params[f"b{last}"])[:, 0]
            err = (pb + out * ys32) - yb
            d_mse = np.float32(2.0) * err / (ys32 * ys32)
            t = np.where(err < 0.0, -err / np.float32(13.0), err / np.float32(10.0))
            dt = np.where(err < 0.0, np.float32(-1.0 / 13.0), np.float32(over_boost / 10.0))
            gt = np.float32(gamma) * np.minimum(t, np.float32(t_cap))
            ex = np.exp(gt - gt.max())
            w_soft = ex / (ex.sum() + np.float32(1e-12)) * np.float32(len(idx))
            d_nasa = w_soft * dt / ys32
            d_pred = np.float32(1.0 - nasa_w) * d_mse + np.float32(nasa_w * nasa_scale) * d_nasa
            delta = (d_pred * ys32 / np.float32(len(idx)))[:, None]
            grads = {}
            for i in range(n_layers - 1, -1, -1):
                grads[f"w{i}"] = acts[i].T @ delta + np.float32(wd) * params[f"w{i}"]
                grads[f"b{i}"] = delta.sum(axis=0)
                if i > 0:
                    delta = (delta @ params[f"w{i}"].T) * (np.float32(1.0) - acts[i] ** 2)
            norm = float(np.sqrt(sum(float(np.sum(g * g)) for g in grads.values())))
            if not np.isfinite(norm):
                raise ValueError("nonfinite residual calibration gradient")
            clip = np.float32(min(1.0, 5.0 / max(norm, 1e-12)))
            step += 1
            b1c = np.float32(1.0 - 0.9 ** step)
            b2c = np.float32(1.0 - 0.999 ** step)
            for k in params:
                g = grads[k] * clip
                m1[k] = np.float32(0.9) * m1[k] + np.float32(0.1) * g
                m2[k] = np.float32(0.999) * m2[k] + np.float32(0.001) * g * g
                params[k] -= lr_now * (m1[k] / b1c) / (np.sqrt(m2[k] / b2c) + np.float32(1e-8))

    if not all(np.all(np.isfinite(v)) for v in params.values()):
        raise ValueError("calibration parameters are not finite")

    p64 = {k: np.round(v.astype(float), 7) for k, v in params.items()}
    h = x_feat.astype(float)
    for i in range(n_layers - 1):
        h = np.tanh(h @ p64[f"w{i}"] + p64[f"b{i}"])
    corr_all = (h @ p64[f"w{last}"] + p64[f"b{last}"])[:, 0] * y_sd
    shifts = {}
    for c in cond_keys:
        mask = cond_int == c
        shifts[str(c)] = _best_shift_asym(
            parent_prediction[mask], corr_all[mask], y[mask], neg_clip, pos_clip,
            shift_lo=-8.0, shift_hi=8.0,
        )
    return {
        "version": "res_mlp_v1",
        "moments": moments,
        "cond_keys": cond_keys,
        "active": active,
        "p_mu": p_mu,
        "p_sd": p_sd,
        "y_sd": y_sd,
        "n_layers": n_layers,
        "params": {k: v.tolist() for k, v in p64.items()},
        "shifts": shifts,
        "neg_clip": neg_clip,
        "pos_clip": pos_clip,
    }


def fit(x, y, parent_prediction, mode="v8"):
    x, y, parent_prediction = _validate_inputs(x, y, parent_prediction)
    if mode == "v8":
        return _fit_v8(x, y, parent_prediction)
    if mode == "spline_v2":
        return _fit_spline_v2(x, y, parent_prediction)
    if mode == "res_mlp_v1":
        return _fit_res_mlp_v1(x, y, parent_prediction)
    raise ValueError(f"unsupported calibration mode: {mode}")


def _predict_v8(x, parent_prediction, state):
    conditions = np.rint(x[:, 21])
    z, known = _normalized(x, conditions, state["moments"])
    correction = _design(z, np.asarray(state["knots"])) @ np.asarray(state["coef"])
    shifts = state.get("shifts", {})  # Older checkpoints only contain the spline.
    shift = np.array([shifts.get(str(int(c)), 0.0) for c in conditions[known]])
    result = np.maximum(0.0, parent_prediction)
    result[known] = _corrected(parent_prediction[known], correction[known], shift)
    return result


def _predict_spline_v2(x, parent_prediction, state):
    conditions = np.rint(x[:, 21])
    z, known = _normalized(x, conditions, state["moments"])
    p_norm = np.clip((parent_prediction - float(state["p_mu"])) / float(state["p_sd"]), -4.0, 6.0)
    z_ext = np.column_stack([z, p_norm])
    cond_int = np.where(np.isfinite(conditions), conditions, 0.0).astype(int)
    knots = np.asarray(state["knots"], dtype=float)
    design = _spline_v2_design(z_ext, cond_int, state["cond_keys"], knots)
    correction = design @ np.asarray(state["coef"], dtype=float)
    shifts = state.get("shifts", {})
    shift = np.array([shifts.get(str(int(c)), 0.0) for c in cond_int[known]])
    result = np.maximum(0.0, parent_prediction)
    result[known] = _corrected(
        parent_prediction[known], correction[known], shift,
        neg_clip=state["neg_clip"], pos_clip=state["pos_clip"],
    )
    return result


def _predict_res_mlp_v1(x, parent_prediction, state):
    conditions = np.rint(x[:, 21])
    z, known = _normalized(x, conditions, state["moments"])
    cond_keys = [int(c) for c in state["cond_keys"]]
    ck_map = {c: idx for idx, c in enumerate(cond_keys)}
    cond_int = np.where(np.isfinite(conditions), conditions, 0.0).astype(int)
    coh = np.zeros((len(x), len(cond_keys)), dtype=float)
    for c in cond_keys:
        coh[cond_int == c, ck_map[c]] = 1.0
    p_norm = np.clip((parent_prediction - float(state["p_mu"])) / float(state["p_sd"]), -5.0, 6.0)[:, None]
    h = np.column_stack([z[:, state["active"]], coh, p_norm])
    params = state["params"]
    n_layers = int(state["n_layers"])
    for i in range(n_layers - 1):
        h = np.tanh(h @ np.asarray(params[f"w{i}"], dtype=float) + np.asarray(params[f"b{i}"], dtype=float))
    last = n_layers - 1
    correction = (
        h @ np.asarray(params[f"w{last}"], dtype=float) + np.asarray(params[f"b{last}"], dtype=float)
    )[:, 0] * float(state["y_sd"])
    shifts = state.get("shifts", {})
    shift = np.array([shifts.get(str(int(c)), 0.0) for c in cond_int[known]])
    result = np.maximum(0.0, parent_prediction)
    result[known] = _corrected(
        parent_prediction[known], correction[known], shift,
        neg_clip=state["neg_clip"], pos_clip=state["pos_clip"],
    )
    return result


def predict(x, parent_prediction, state):
    x = np.asarray(x, dtype=float)
    parent_prediction = np.asarray(parent_prediction, dtype=float).reshape(-1)
    version = state.get("version")
    if version is None or version == "v8":
        return _predict_v8(x, parent_prediction, state)
    if version == "spline_v2":
        return _predict_spline_v2(x, parent_prediction, state)
    if version == "res_mlp_v1":
        return _predict_res_mlp_v1(x, parent_prediction, state)
    raise ValueError(f"unsupported calibration state version: {version}")
