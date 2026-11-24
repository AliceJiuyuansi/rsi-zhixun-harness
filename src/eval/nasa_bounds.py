"""二进制指数分桶的NASA区间，仅用于粗筛研究；不替代正式精算。"""
import numpy as np


def _log1p_bucket_sum(exponents, counts):
    positive = exponents > 0
    if not np.any(positive):
        return 0.0
    values = exponents[positive]
    terms = np.log(counts[positive]) + values + np.log(-np.expm1(-values))
    peak = float(terms.max())
    return float(np.logaddexp(0., peak + np.log(np.exp(terms-peak).sum())))


def binary_nasa_bounds(y_true, y_pred, subdivisions=8):
    """每个2倍指数范围再分m桶；返回log(1+NASA)的上下界。"""
    if not isinstance(subdivisions, int) or not 1 <= subdivisions <= 1024:
        raise ValueError("subdivisions must be an integer in 1..1024")
    truth, pred = np.asarray(y_true, dtype=float).reshape(-1), np.asarray(y_pred, dtype=float).reshape(-1)
    if truth.shape != pred.shape or not len(truth) or not np.all(np.isfinite(truth)) or not np.all(np.isfinite(pred)):
        raise ValueError("expected aligned finite nonempty arrays")
    error = pred-truth
    exponent = np.where(error < 0, -error/13., error/10.)
    exponent = exponent[exponent > 0]  # 已知零误差的贡献严格为0。
    if not len(exponent):
        return {"lower_log1p": 0., "upper_log1p": 0., "occupied_buckets": 0,
                "rows": len(truth), "subdivisions": subdivisions}
    step = np.log(2.)/subdivisions
    indices = np.floor(exponent/step)
    if not np.all(np.isfinite(indices)) or indices.max() >= np.iinfo(np.int64).max-1:
        raise ValueError("exponent too large for bucket indices; use exact log-sum-exp")
    indices = indices.astype(np.int64)
    if indices.max() < 65536:
        counts = np.bincount(indices)
        buckets = np.flatnonzero(counts)
        counts = counts[buckets]
    else:
        buckets, counts = np.unique(indices, return_counts=True)
    lower = np.maximum(0., np.nextafter(buckets*step, -np.inf))
    upper = np.nextafter((buckets+1)*step, np.inf)
    return {"lower_log1p": max(0., _log1p_bucket_sum(lower, counts)-1e-12),
            "upper_log1p": _log1p_bucket_sum(upper, counts)+1e-12,
            "occupied_buckets": len(buckets), "rows": len(truth), "subdivisions": subdivisions}


def compare_bounds(candidate, parent):
    if candidate["lower_log1p"] > parent["upper_log1p"]:
        return "definitely_worse"
    if candidate["upper_log1p"] <= parent["lower_log1p"]:
        return "definitely_noninferior"
    return "need_exact"
