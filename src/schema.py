"""Shared data contracts for the RSI harness.

Field semantics are defined in docs/04_工程架构与规范.md section 3.
Add fields freely; do not change the meaning of an existing one without
updating the document and notifying the team.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

import numpy as np


@dataclass
class StreamSlice:
    t: int
    X: np.ndarray
    y: Optional[np.ndarray]
    meta: dict = field(default_factory=dict)


@dataclass
class Metrics:
    perf: float
    perf_delta: float
    latency_p95: float
    n_samples: int


@dataclass
class DriftSignal:
    t: int
    feature_score: float      # max PSI across features, 0..inf
    residual_score: float     # accuracy gap vs reference, >=0 when degrading
    is_drift: bool
    evidence: dict = field(default_factory=dict)


@dataclass
class Diagnosis:
    kind: Literal["covariate", "concept", "label", "none"]
    confidence: float
    reason: str
    suggested: list[str] = field(default_factory=list)


@dataclass
class Action:
    kind: Literal["none", "retrain", "finetune", "swap_model", "reweight"]
    params: dict = field(default_factory=dict)
    est_cost: float = 0.0
    est_gain: float = 0.0


@dataclass
class GateDecision:
    approved: bool
    reason: str
    action: Action


@dataclass
class CostRecord:
    gpu_seconds: float = 0.0
    labels: int = 0
    tokens: int = 0
    wall_seconds: float = 0.0
