"""Retrain budget (受控 RSI 五件套 #3, docs/04 §4).

Two independent caps, both enforced here (defence in depth - the policies also
check ``max_retrains``):

- lifetime cap: at most ``max_retrains`` updates per run
- rate cap:    at most ``max_in_window`` updates per ``window_steps`` steps
               (``max_in_window <= 0`` disables it; the cooldown in the policy
               is the primary rate limit, this is the structural backstop)

Over budget means the only legal action is ``none``.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RetrainBudget:
    max_retrains: int = 12
    max_in_window: int = 4
    window_steps: int = 40
    steps: list[int] = field(default_factory=list)

    def check(self, step: int, retrains_so_far: int) -> tuple[bool, str]:
        if retrains_so_far >= self.max_retrains:
            return False, f"retrain budget exhausted ({retrains_so_far}/{self.max_retrains})"
        if self.max_in_window > 0:
            recent = [s for s in self.steps if step - int(self.window_steps) < s <= step]
            if len(recent) >= self.max_in_window:
                return False, (f"rate cap reached: {len(recent)} updates in the last "
                               f"{self.window_steps} steps (max {self.max_in_window})")
        return True, ""

    def record(self, step: int) -> None:
        self.steps.append(int(step))

    def as_dict(self) -> dict:
        return {"max_retrains": self.max_retrains,
                "max_in_window": self.max_in_window,
                "window_steps": self.window_steps,
                "retrain_steps": list(self.steps)}
