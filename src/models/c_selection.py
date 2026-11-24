"""C父子必须同一验证投影比较；NASA使用稳定对数总分，而非最大值。"""
import math


def _log_score(metrics):
    if "nasa" not in metrics:
        return None
    try:
        score = float(metrics["nasa"])
        if not math.isfinite(score) or score < 0:
            return None
        if "nasa_log1p" in metrics:
            value = float(metrics["nasa_log1p"])
            return value if math.isfinite(value) and value >= 0 else None
        return None if metrics.get("nasa_saturated") else math.log1p(score)
    except (TypeError, ValueError):
        return None


def nasa_noninferior(before, after):
    parent, candidate = _log_score(before), _log_score(after)
    passed = parent is not None and candidate is not None and candidate <= parent + 1e-9
    return {"passed": passed, "reason": "pass" if passed else "reject_nasa",
            "parent_log1p": parent, "candidate_log1p": candidate,
            "comparison": "same-validation summed NASA; numerical log tolerance 1e-9"}


def c_candidate_decision(before, after, reference_rmse=None, rmse_tolerance=.01, min_nasa_gain=.10):
    """同集NASA优先，RMSE受持久参考约束；不把逐轮容差相乘。"""
    nasa_gate = nasa_noninferior(before, after)
    try:
        parent_rmse, candidate_rmse = float(before["rmse"]), float(after["rmse"])
        reference = parent_rmse if reference_rmse is None else float(reference_rmse)
        valid = all(math.isfinite(v) and v >= 0 for v in (parent_rmse, candidate_rmse, reference))
        valid = valid and 0 <= rmse_tolerance <= .01 and 0 <= min_nasa_gain <= 1
    except (KeyError, TypeError, ValueError):
        parent_rmse = candidate_rmse = reference = 0.
        valid = False
    parent_log, candidate_log = _log_score(before), _log_score(after)
    nasa_gain = 0.
    if parent_log is not None and candidate_log is not None and parent_log > 0:
        if candidate_log == 0:
            nasa_gain = 1.
        else:
            log_parent_score = parent_log + math.log(-math.expm1(-parent_log))
            log_candidate_score = candidate_log + math.log(-math.expm1(-candidate_log))
            nasa_gain = -math.expm1(min(700., log_candidate_score-log_parent_score))
    rmse_gain = (parent_rmse-candidate_rmse)/max(parent_rmse, 1e-9)
    cap = reference*(1+rmse_tolerance)
    accepted = bool(valid and nasa_gate["passed"] and candidate_rmse <= cap+1e-9)
    promote = accepted and (rmse_gain >= .002 or nasa_gain >= min_nasa_gain)
    return {"accepted": accepted, "promote": bool(promote), "nasa_gate": nasa_gate,
            "nasa_gain": nasa_gain, "reference_rmse": reference, "rmse_cap": cap,
            "reason": "pass" if promote else "reject_c_risk_or_minimum_gain"}
