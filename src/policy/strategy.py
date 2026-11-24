"""Drift-band strategy: turn monitoring signals into ONE whitelisted action.

This is the "decision" half of the RSI loop. It is deliberately a small, readable
table instead of a black box, because the competition scores *why* you update
(专家分：漂移检测 8 / 自主决策 10), not only the final metric.

Signals (all from monitor/drift.py, computed over the recent `policy.window`):

    psi      max feature PSI vs the reference window   -> "inputs moved"
    gap      target_perf - rolling_perf                -> "we are losing score"
    pred_js  Jensen-Shannon of the prediction histogram-> "answers moved"
    kind     covariate / concept / label / none        -> "which knob to turn"

Decision table (classification, macro-F1 units):

    | psi        | gap      | reading                          | action          |
    |------------|----------|----------------------------------|-----------------|
    | < 0.10     | < 0.02   | nothing to do                    | none            |
    | < 0.10     | >= 0.02  | inputs steady, mapping broke      | finetune 4 ep   |
    |            |          | (prediction histogram moved too)  | reweight 2x     |
    | 0.10-0.25  | any      | mild sensor drift                | finetune 3 ep   |
    | >= 0.25    | < 0.03   | inputs moved, model still holds   | none (watch)    |
    | >= 0.25    | >= 0.03  | clear covariate drift + damage    | finetune 5 ep   |

Why "finetune" and never "retrain from scratch": the platform's parent
checkpoint must be *continued* (weights + Adam moments + normalisation scalers).
A from-scratch model cannot be declared as its continuation, so the action space
inside one round is epochs / learning rate / sample weighting, not architecture.

Cost matters: every action spends wall-clock and epochs, which is the
"resource efficiency" score (5 pts). A band that spends 5 epochs to recover 0.01
macro-F1 is a loss, hence `MIN_GAIN` and the "watch, do nothing" branch.
"""
from __future__ import annotations

from dataclasses import dataclass

from src.schema import Action

# --- thresholds (documented, tunable) ---------------------------------
PSI_STABLE = 0.10          # PSI < 0.10: distributions look the same
PSI_MILD = 0.25            # 0.10-0.25: mild drift; >= 0.25: clear drift
GAP_WATCH = 0.02           # error gap we accept without acting (macro-F1 units)
GAP_ACT = 0.03             # gap that justifies the most expensive update
MIN_GAIN = 0.01            # estimated recoverable score must beat this

# --- action presets (all inside the whitelist's parameter ranges) ------
MEDIUM = {"epochs": 3, "lr": 1.5e-3}   # mild drift, cheap touch-up
STRONG = {"epochs": 5, "lr": 2e-3}     # confirmed drift + damage
CONCEPT = {"epochs": 4, "lr": 1e-3}    # inputs steady, mapping broke


@dataclass
class Signal:
    psi: float
    gap: float
    pred_js: float = 0.0
    kind: str = "none"          # whatever the monitor already guessed
    retrains: int = 0
    max_retrains: int = 12


def diagnose(s: Signal) -> str:
    """Name the drift so the audit log and the report can justify the action.

    covariate = inputs moved (this dataset's 36-month sensor ageing lands here)
    concept   = inputs steady but the score dropped (the mapping broke)
    label     = answers/priors moved without feature drift
    none      = nothing worth acting on
    """
    if s.psi >= PSI_STABLE:
        return "covariate"
    if s.gap >= GAP_WATCH:
        return "label" if s.pred_js >= 0.10 else "concept"
    return "none"


def choose(s: Signal) -> Action:
    """Return exactly one whitelisted action (pure function, easy to test)."""
    if s.retrains >= s.max_retrains:
        return Action("none", {}, 0.0, 0.0)

    kind = diagnose(s)

    if kind == "none":
        # Nothing worth spending compute on: staying still is a legitimate move.
        return Action("none", {}, 0.0, 0.0)

    if kind == "covariate":
        if s.psi >= PSI_MILD:
            if s.gap < GAP_ACT:
                # inputs drifted but the model still holds: spend nothing
                return Action("none", {}, 0.0, 0.0)
            return Action("finetune", dict(STRONG), est_gain=s.gap)
        if s.gap >= GAP_WATCH:
            return Action("finetune", dict(MEDIUM), est_gain=s.gap)
        return Action("none", {}, 0.0, 0.0)

    if kind == "concept":
        if s.gap < MIN_GAIN:
            return Action("none", {}, 0.0, 0.0)
        return Action("finetune", dict(CONCEPT), est_gain=s.gap)

    # label shift: the class prior moved; re-weighting recent samples is cheaper
    # than re-fitting every epoch on stale priors.
    return Action("reweight", {"recent_weight": 2.0}, est_gain=s.gap)


def explain(a: Action, s: Signal) -> str:
    """One human sentence for the audit log / defence slide."""
    return (f"psi={s.psi:.3f} gap={s.gap:.3f} pred_js={s.pred_js:.3f} "
            f"-> diagnosis={diagnose(s)} -> action={a.kind}"
            + (f" {a.params}" if a.params else ""))
