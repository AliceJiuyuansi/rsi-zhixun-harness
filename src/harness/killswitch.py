"""kill-switch (受控 RSI 五件套 #4, docs/04 §4).

Any one of these freezes all updates and keeps the current champion:

- ``RSI_STOP=1`` / ``RSI_KILL_SWITCH=1``        (operator / grading harness)
- ``policy.kill_switch: true`` in the config    (reproducible experiments)
- a ``STOP`` file in the run dir or ``$RSI_STOP_FILE``
- wall-clock budget ``policy.max_wall_seconds`` / ``$RSI_MAX_WALL_SECONDS``
- ``consecutive_failures`` unexpected step errors (automatic degradation)

Once tripped it stays tripped: a frozen system must not silently resume.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

_TRUTHY = {"1", "true", "yes", "on", "y", "t"}


def _truthy(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in _TRUTHY


class KillSwitch:
    def __init__(self, cfg: dict | None = None, root: str | Path | None = None,
                 env: dict | None = None, now=None):
        cfg = cfg or {}
        policy = cfg.get("policy", {}) or {}
        self.env = os.environ if env is None else env
        self.root = Path(root) if root else None
        self.now = now or time.monotonic
        self.t0 = self.now()
        self.max_failures = int(policy.get("max_consecutive_failures", 3))
        self.consecutive_failures = 0
        self.total_failures = 0
        self.reason: str = ""
        self.stopped = False
        self._cfg_flag = _truthy(policy.get("kill_switch", cfg.get("kill_switch", False)))
        limit = policy.get("max_wall_seconds", self.env.get("RSI_MAX_WALL_SECONDS", 0))
        try:
            self.max_wall_seconds = float(limit or 0)
        except (TypeError, ValueError):
            self.max_wall_seconds = 0.0

    # --- reasons -------------------------------------------------------
    def _check(self) -> tuple[bool, str]:
        for name in ("RSI_STOP", "RSI_KILL_SWITCH"):
            if _truthy(self.env.get(name, "")):
                return True, f"{name} set"
        if self._cfg_flag:
            return True, "policy.kill_switch=true"
        try:
            if self.root is not None:
                stop = self.root / "STOP"
                if stop.exists():
                    return True, f"STOP file present ({stop})"
        except OSError:
            pass
        path = self.env.get("RSI_STOP_FILE", "")
        try:
            if path and Path(path).exists():
                return True, f"RSI_STOP_FILE present ({path})"
        except OSError:
            pass
        if self.max_wall_seconds > 0 and (self.now() - self.t0) >= self.max_wall_seconds:
            return True, f"wall-clock budget {self.max_wall_seconds:g}s exhausted"
        if self.consecutive_failures >= self.max_failures:
            return True, (f"{self.consecutive_failures} consecutive step failures "
                          f"(>= {self.max_failures})")
        return False, ""

    def tripped(self) -> bool:
        """Check every source (cheap: one env lookup + one stat) and latch."""
        if self.stopped:
            return True
        hit, why = self._check()
        if hit:
            self.stopped = True
            self.reason = why
        return self.stopped

    @property
    def status(self) -> str:
        return self.reason or "running"

    # --- automatic degradation ------------------------------------------
    def note_step_ok(self) -> None:
        self.consecutive_failures = 0

    def note_step_error(self, exc: BaseException | str) -> None:
        self.consecutive_failures += 1
        self.total_failures += 1

    def as_dict(self) -> dict:
        return {"stopped": bool(self.stopped or self.tripped()),
                "reason": self.status,
                "max_wall_seconds": self.max_wall_seconds,
                "consecutive_failures": self.consecutive_failures,
                "total_failures": self.total_failures}
