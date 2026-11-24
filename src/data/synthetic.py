"""Synthetic industrial sensor stream with controllable drift.

This is the proxy dataset used until the official data package is available.
It mimics: 12 sensors, 4 classes (normal + 3 fault types), time-ordered batches,
and five drift patterns (none / gradual / sudden / recurring / concept / label shift).

The generator is deterministic given `seed` and `means_seed`.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


DEFAULTS = {
    "n_features": 12,
    "n_classes": 4,
    "n_steps": 160,
    "batch_size": 64,
    "sigma": 1.0,
    "class_scale": 1.9,
    "seed": 7,
    "means_seed": 2026,
    "priors": [0.40, 0.20, 0.20, 0.20],
}


@dataclass
class SyntheticSpec:
    n_features: int = 12
    n_classes: int = 4
    n_steps: int = 160
    batch_size: int = 64
    sigma: float = 1.0
    class_scale: float = 1.9
    seed: int = 7
    means_seed: int = 2026
    priors: list[float] = field(default_factory=lambda: [0.40, 0.20, 0.20, 0.20])
    drift: dict = field(default_factory=dict)

    @classmethod
    def from_cfg(cls, cfg: dict) -> "SyntheticSpec":
        spec = cls()
        for k in ("n_features", "n_classes", "n_steps", "batch_size",
                  "sigma", "class_scale", "seed", "means_seed"):
            if k in cfg:
                setattr(spec, k, cfg[k])
        if "priors" in cfg:
            spec.priors = list(cfg["priors"])
        spec.drift = dict(cfg.get("drift") or {})
        return spec


def _class_means(d: int, k: int, scale: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    m = rng.normal(size=(k, d))
    m -= m.mean(axis=0, keepdims=True)
    m /= np.linalg.norm(m, axis=1, keepdims=True)
    return m * scale


def _unit_vector(d: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed + 991)
    v = rng.normal(size=d)
    return v / np.linalg.norm(v)


def _sensor_bias(d: int, seed: int, frac: float = 0.4) -> np.ndarray:
    """Per-sensor calibration bias: a subset of channels drifts, with random signs.

    A common-mode shift in all features barely hurts a tree model; real sensor
    drift is per-channel, so this is both more realistic and more adversarial.
    """
    rng = np.random.default_rng(seed + 17)
    k = max(2, int(round(d * frac)))
    idx = rng.choice(d, size=k, replace=False)
    bias = np.zeros(d)
    bias[idx] = rng.choice([-1.0, 1.0], size=k)
    return bias


def _drift_delta(t: int, cfg: dict, d: int, seed: int) -> np.ndarray:
    kind = cfg.get("kind", "none")
    u = _sensor_bias(d, seed)
    if kind == "covariate_gradual":
        start, end = cfg.get("start", 40), cfg.get("end", 120)
        a = np.clip((t - start) / max(end - start, 1), 0.0, 1.0)
        return a * cfg.get("magnitude", 1.8) * u
    if kind == "covariate_sudden":
        return (cfg.get("magnitude", 2.0) * u) if t >= cfg.get("start", 60) else np.zeros(d)
    if kind == "recurring":
        start = cfg.get("start", 40)
        if t < start:
            return np.zeros(d)
        phase = ((t - start) // cfg.get("period", 30)) % 2
        return cfg.get("magnitude", 1.8) * u * phase
    return np.zeros(d)


def _priors_at(t: int, cfg: dict, base: list[float]) -> list[float]:
    if cfg.get("kind") == "label_shift" and t >= cfg.get("start", 60):
        p = cfg.get("priors_after")
        if p:
            return [float(x) for x in p]
    if cfg.get("kind") == "label_shift":
        p = cfg.get("priors_before")
        if p:
            return [float(x) for x in p]
    return list(base)


def generate(spec: SyntheticSpec) -> dict:
    """Return X (N,d), y (N,), t (N,), and metadata."""
    rng = np.random.default_rng(spec.seed)
    d, k = spec.n_features, spec.n_classes
    means = _class_means(d, k, spec.class_scale, spec.means_seed)
    drift = dict(spec.drift or {})

    X_all, y_all, t_all = [], [], []
    for t in range(spec.n_steps):
        priors = _priors_at(t, drift, spec.priors)
        y = rng.choice(k, size=spec.batch_size, p=priors)
        delta = _drift_delta(t, drift, d, spec.seed)
        X = means[y] + delta + rng.normal(0.0, spec.sigma, size=(spec.batch_size, d))

        # Concept drift: relabel after the change point, keep X marginal identical.
        if drift.get("kind") == "concept" and t >= drift.get("start", 60):
            a, b = drift.get("swap", [1, 2])
            old = y.copy()
            y[old == a] = b
            y[old == b] = a

        X_all.append(X)
        y_all.append(y)
        t_all.append(np.full(spec.batch_size, t, dtype=np.int64))

    return {
        "X": np.concatenate(X_all),
        "y": np.concatenate(y_all),
        "t": np.concatenate(t_all),
        "meta": {
            "kind": drift.get("kind", "none"),
            "n_steps": spec.n_steps,
            "batch_size": spec.batch_size,
            "n_features": d,
            "n_classes": k,
            "seed": spec.seed,
            "means_seed": spec.means_seed,
        },
    }
