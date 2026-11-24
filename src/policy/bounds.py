"""Amplitude cap (受控 RSI 五件套 #2, docs/04 §4).

One action may move a knob by at most ``factor`` x its previous value (and always
inside the whitelist range). Anything larger is clipped, not rejected, and the
clip is reported so the audit log can show it.
"""
from __future__ import annotations

from src.policy.whitelist import PARAM_SPEC


class AmplitudeLimiter:
    def __init__(self, factor: float = 2.0):
        self.factor = max(float(factor), 1.0)
        self.last: dict[str, dict[str, float]] = {}

    def clip(self, kind: str, params: dict) -> tuple[dict, list[str]]:
        """Clip ``params`` against the last accepted values. Returns (params, notes)."""
        spec = PARAM_SPEC.get(kind, {})
        prev = self.last.get(kind, {})
        out: dict[str, float] = {}
        notes: list[str] = []
        for name, val in (params or {}).items():
            if name not in spec:
                continue  # the whitelist owns this decision
            lo, hi = spec[name]
            v = float(val)
            if name in prev and prev[name] != 0:
                base = float(prev[name])
                lo_b = max(lo, base / self.factor)
                hi_b = min(hi, base * self.factor)
                if lo_b <= hi_b:
                    clipped = min(max(v, lo_b), hi_b)
                    if clipped != v:
                        notes.append(f"{name}: {v:g} -> {clipped:g} (<= {self.factor:g}x of {base:g})")
                        v = clipped
            out[name] = v
        return out, notes

    def commit(self, kind: str, params: dict) -> None:
        if params:
            self.last[kind] = {k: float(v) for k, v in params.items()}

    def as_dict(self) -> dict:
        return {k: dict(v) for k, v in self.last.items()}
