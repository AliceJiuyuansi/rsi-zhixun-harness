"""Update policies: static / periodic / trigger, all behind one interface."""
from __future__ import annotations

from dataclasses import dataclass

from src.policy.whitelist import parse_proposal
from src.schema import Action, GateDecision


@dataclass
class PolicyState:
    step: int
    retrains: int
    last_retrain_step: int
    psi_max: float
    error_gap: float
    diagnosis: dict
    est_train_seconds: float = 0.0
    est_train_labels: int = 0


class BasePolicy:
    name = "base"

    def __init__(self, cfg: dict):
        self.cfg = cfg

    def decide(self, s: PolicyState) -> GateDecision:
        return GateDecision(True, "base", Action("none"))


class StaticPolicy(BasePolicy):
    name = "static"

    def decide(self, s: PolicyState) -> GateDecision:
        return GateDecision(False, "static policy: never update", Action("none"))


class PeriodicPolicy(BasePolicy):
    name = "periodic"

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self.every = int(cfg.get("periodic_every", 20))

    def decide(self, s: PolicyState) -> GateDecision:
        due = s.step > 0 and s.step % self.every == 0
        if s.retrains >= int(self.cfg.get("max_retrains", 12)):
            return GateDecision(False, "retrain budget exhausted", Action("none"))
        if due:
            return GateDecision(True, f"periodic: step {s.step} % {self.every} == 0",
                                Action("retrain", est_gain=float("nan")))
        return GateDecision(False, "not scheduled", Action("none"))


class TriggerPolicy(BasePolicy):
    name = "trigger"

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self.psi_thr = float(cfg.get("psi_threshold", 0.25))
        self.err_thr = float(cfg.get("error_threshold", 0.06))
        self.cooldown = int(cfg.get("retrain_cooldown", 4))
        self.max_retrains = int(cfg.get("max_retrains", 12))
        self.min_gain = float(cfg.get("min_estimated_gain", 0.015))

    def decide(self, s: PolicyState) -> GateDecision:
        if s.retrains >= self.max_retrains:
            return GateDecision(False, "retrain budget exhausted", Action("none"))
        since = s.step - s.last_retrain_step
        if s.last_retrain_step >= 0 and since < self.cooldown:
            return GateDecision(False, f"cooldown ({since}/{self.cooldown})", Action("none"))

        feature_drift = s.psi_max >= self.psi_thr
        error_drift = s.error_gap >= self.err_thr
        if not (feature_drift or error_drift):
            return GateDecision(False, f"no trigger (psi={s.psi_max:.3f}, gap={s.error_gap:.3f})",
                                Action("none"))

        # Expected gain is the recoverable error gap; require it to clear a floor.
        est_gain = max(s.error_gap, 0.0)
        if est_gain < self.min_gain:
            return GateDecision(False,
                                f"estimated gain {est_gain:.3f} < {self.min_gain}",
                                Action("none", est_gain=est_gain))

        reason = (f"{'feature ' if feature_drift else ''}{'error' if error_drift else ''}"
                  f"drift detected; est_gain={est_gain:.3f}; diagnosis={s.diagnosis.get('kind')}")
        return GateDecision(True, reason, Action("retrain", est_gain=est_gain,
                                                est_cost=s.est_train_seconds))


class ExternalPolicy(BasePolicy):
    """Slot for an LLM / remote proposer (PyTRIO Decision Model, HTTP service...).

    The proposer is never trusted: its reply is parsed by
    ``policy.whitelist.parse_proposal`` and anything malformed, off-list or
    out-of-range becomes ``none``. A crashing proposer is just another ``none``.
    Even an accepted proposal still has to pass ``harness.guard.Gatekeeper``.
    """

    name = "external"

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self.proposer = cfg.get("proposer")
        self.errors = 0
        self.rejects = 0

    def decide(self, s: PolicyState) -> GateDecision:
        if self.proposer is None:
            return GateDecision(False, "no external proposer configured; none",
                                Action("none"))
        try:
            raw = self.proposer(s)
        except Exception as exc:  # timeout / bad SDK / network: fall back to none
            self.errors += 1
            return GateDecision(False, f"proposer failed ({type(exc).__name__}); none",
                                Action("none"))
        prop = parse_proposal(raw)
        if not prop.ok:
            self.rejects += 1
            return GateDecision(False, f"proposal rejected: {prop.rejected}; none",
                                Action("none"))
        if prop.action.kind == "none":
            return GateDecision(False, "proposer chose none", prop.action)
        return GateDecision(True, "external proposal accepted by whitelist", prop.action)


def build_policy(cfg: dict) -> BasePolicy:
    kind = cfg.get("type", "trigger")
    if kind == "static":
        return StaticPolicy(cfg)
    if kind == "periodic":
        return PeriodicPolicy(cfg)
    if kind == "trigger":
        return TriggerPolicy(cfg)
    if kind == "external":
        return ExternalPolicy(cfg)
    raise ValueError(f"unknown policy: {kind}")
