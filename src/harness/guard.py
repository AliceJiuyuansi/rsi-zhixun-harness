"""Gatekeeper: wires the four control pieces together before anything executes.

    kill-switch -> whitelist -> rate/lifetime budget -> amplitude cap -> approve

It sits between "a policy (or an external LLM) said X" and "the harness does X".
The controller may propose; only the gatekeeper gives permission. Anything that
fails degrades to ``none`` - the current champion keeps serving, which is the
required behaviour for an unattended run.

Contract: docs/04_工程架构与规范.md §4.  Tests: scripts/check_guard.py.
"""
from __future__ import annotations

from pathlib import Path

from src.harness.killswitch import KillSwitch
from src.policy.bounds import AmplitudeLimiter
from src.policy.budget import RetrainBudget
from src.policy.whitelist import IMPLEMENTED_ACTIONS, NONE_ACTION, validate_action
from src.schema import GateDecision
from src.utils.audit import AuditLog

# actions that cost budget; ``none`` never does
BUDGETED = ("retrain", "finetune", "swap_model", "reweight")


class Gatekeeper:
    def __init__(self, cfg: dict | None = None, audit: AuditLog | None = None,
                 kill: KillSwitch | None = None, root: str | Path | None = None,
                 executable: tuple[str, ...] | None = None):
        cfg = cfg or {}
        policy = cfg.get("policy", {}) or {}
        self.audit = audit if audit is not None else AuditLog(None)
        self.kill = kill or KillSwitch(cfg, root=root)
        self.executable = tuple(executable or IMPLEMENTED_ACTIONS)
        self.limiter = AmplitudeLimiter(factor=float(policy.get("amplitude_factor", 2.0)))
        self.budget = RetrainBudget(
            max_retrains=int(policy.get("max_retrains", 12)),
            max_in_window=int(policy.get("max_retrains_per_window", 0)),
            window_steps=int(policy.get("window", 40)),
        )
        self.whitelist_rejects = 0
        self.budget_rejects = 0
        self.kill_rejects = 0
        self.bounds_clips = 0

    # --- the gate ------------------------------------------------------
    def review(self, decision: GateDecision, step: int, retrains: int) -> GateDecision:
        if self.kill.tripped():
            self.kill_rejects += 1
            why = f"kill-switch: {self.kill.status}"
            self.audit.write("kill_switch", step=step, action=str(decision.action.kind),
                             proposed_approved=bool(decision.approved), reason=why)
            return GateDecision(False, why, NONE_ACTION)

        ok, action, why = validate_action(decision.action)
        if not ok:
            self.whitelist_rejects += 1
            self.audit.write("reject_whitelist", step=step,
                             proposed=str(getattr(decision.action, "kind", decision.action)),
                             reason=why)
            return GateDecision(False, f"whitelist reject: {why}", NONE_ACTION)

        if action.kind not in self.executable:
            self.whitelist_rejects += 1
            why = f"action '{action.kind}' is declared but not implemented here"
            self.audit.write("reject_whitelist", step=step, reason=why,
                             proposed=str(action.kind))
            return GateDecision(False, f"whitelist reject: {why}", NONE_ACTION)

        if action.kind in BUDGETED:
            allowed, why = self.budget.check(step, retrains)
            if not allowed:
                self.budget_rejects += 1
                self.audit.write("reject_budget", step=step, action=action.kind, reason=why)
                return GateDecision(False, f"budget reject: {why}", NONE_ACTION)

            clipped, notes = self.limiter.clip(action.kind, action.params)
            if notes:
                self.bounds_clips += 1
                self.audit.write("bounds_clip", step=step, action=action.kind, notes=notes)
                action = type(action)(action.kind, clipped, action.est_cost, action.est_gain)

        return GateDecision(decision.approved, decision.reason, action)

    # --- bookkeeping after execution -----------------------------------
    def commit(self, action, step: int) -> None:
        """Remember an executed action: bounds baseline + budget usage."""
        if action is None:
            return
        if action.params:
            self.limiter.commit(action.kind, action.params)
        if action.kind in BUDGETED:
            self.budget.record(step)

    def note_step_ok(self) -> None:
        self.kill.note_step_ok()

    def note_step_error(self, exc) -> None:
        self.kill.note_step_error(exc)

    def as_dict(self) -> dict:
        return {
            "kill_switch": self.kill.as_dict(),
            "budget": self.budget.as_dict(),
            "bounds": self.limiter.as_dict(),
            "rejects": {
                "whitelist": self.whitelist_rejects,
                "budget": self.budget_rejects,
                "kill_switch": self.kill_rejects,
                "bounds_clips": self.bounds_clips,
            },
        }
