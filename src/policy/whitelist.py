"""Action whitelist (受控 RSI 五件套 #1).

The system may only ever perform the actions listed here. Two entry points:

- ``validate_action`` for in-process policies (static / periodic / trigger):
  checks the enum, the allowed parameter names and their ranges, and clamps
  nothing (rejection is the caller's job).
- ``parse_proposal`` for *external* proposals (an LLM reply, a JSON file, a
  queue message): it **never raises**. Anything unparsable, off-list or
  out-of-range degrades to ``none`` - that is the whole point, because
  "runs to completion unattended" is a hard gate in the rules.

Contract: docs/04_工程架构与规范.md §4.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any

from src.schema import Action

ALLOWED_ACTIONS: tuple[str, ...] = (
    "none", "retrain", "finetune", "swap_model", "reweight",
)

# Declared vocabulary vs what this harness can actually execute. The rest are
# reserved: whitelisting them (params/ranges) is design work already done, but a
# proposal to use them is rejected until an executor exists - never "executed"
# as a silent retrain.
IMPLEMENTED_ACTIONS: tuple[str, ...] = ("none", "retrain")

# action kind -> {param name: (low, high) inclusive}
# Adding an action here is a design decision: it must be answerable "when is it
# chosen, why is it cheaper than retrain, what happens if it is worse".
PARAM_SPEC: dict[str, dict[str, tuple[float, float]]] = {
    "none": {},
    "retrain": {},
    "finetune": {"lr": (1e-4, 1e-2), "epochs": (1, 20)},
    "swap_model": {
        "max_iter": (20, 300),
        "learning_rate": (0.01, 0.3),
        "max_leaf_nodes": (5, 63),
        "l2_regularization": (0.0, 10.0),
    },
    "reweight": {"recent_weight": (1.0, 5.0), "half_life_steps": (1.0, 120.0)},
}

# 幅度上限 (bounds.py) limits how many knobs one action may touch at once.
MAX_PARAMS_PER_ACTION = 2

NONE_ACTION = Action("none", {}, 0.0, 0.0)


def _finite(value: Any) -> float | None:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


@dataclass
class Proposal:
    """A parsed external proposal plus why it was rejected (empty = accepted)."""

    action: Action
    rejected: str = ""

    @property
    def ok(self) -> bool:
        return not self.rejected


def validate_action(action: Any, prev_params: dict | None = None) -> tuple[bool, Action, str]:
    """Return ``(ok, sanitized_action, reason)``. Never raises."""
    if not isinstance(action, Action):
        return False, NONE_ACTION, f"not an Action object: {type(action).__name__}"
    kind = str(action.kind)
    if kind not in ALLOWED_ACTIONS:
        return False, NONE_ACTION, f"action '{kind}' is not whitelisted"
    if kind == "none":
        return True, NONE_ACTION, ""

    params = action.params if isinstance(action.params, dict) else {}
    if action.params and not isinstance(action.params, dict):
        return False, NONE_ACTION, "params is not a dict"

    spec = PARAM_SPEC.get(kind, {})
    unknown = sorted(set(params) - set(spec))
    if unknown:
        return False, NONE_ACTION, f"param(s) {unknown} not allowed for '{kind}'"
    if len(params) > MAX_PARAMS_PER_ACTION:
        return False, NONE_ACTION, (
            f"{len(params)} params changed at once > cap {MAX_PARAMS_PER_ACTION}")

    clean: dict[str, float] = {}
    for name, raw in params.items():
        val = _finite(raw)
        if val is None:
            return False, NONE_ACTION, f"param '{name}' is not a finite number"
        lo, hi = spec[name]
        if val < lo or val > hi:
            return False, NONE_ACTION, (
                f"param '{name}'={val} outside allowed range [{lo}, {hi}]")
        clean[name] = val

    return True, Action(kind, clean, _finite(action.est_cost) or 0.0,
                        _finite(action.est_gain) or 0.0), ""


def _strip_fence(text: str) -> str:
    """Tolerate ```json ... ``` wrappers around an LLM reply."""
    t = text.strip()
    if "```" not in t:
        return t
    body = t.split("```")
    # longest chunk is normally the code block
    body = sorted(body, key=len, reverse=True)
    chunk = body[0].strip()
    if chunk.lower().startswith("json"):
        chunk = chunk[4:].strip()
    return chunk


def parse_proposal(raw: Any) -> Proposal:
    """Parse an untrusted proposal. Any defect -> ``none`` with a reason."""
    if raw is None:
        return Proposal(NONE_ACTION, "empty proposal")
    if isinstance(raw, Action):
        ok, act, why = validate_action(raw)
        return Proposal(act, "" if ok else why)
    if isinstance(raw, str):
        text = _strip_fence(raw)
        if not text:
            return Proposal(NONE_ACTION, "empty proposal text")
        try:
            obj = json.loads(text)
        except json.JSONDecodeError as exc:
            return Proposal(NONE_ACTION, f"malformed JSON: {exc.msg}")
    else:
        obj = raw

    if not isinstance(obj, dict):
        return Proposal(NONE_ACTION, f"proposal is a {type(obj).__name__}, expected object")

    kind = obj.get("kind", obj.get("action"))
    if not isinstance(kind, str):
        return Proposal(NONE_ACTION, "missing 'kind'")
    if kind not in ALLOWED_ACTIONS:
        return Proposal(NONE_ACTION, f"action '{kind}' is not whitelisted")

    params = obj.get("params", {}) or {}
    if not isinstance(params, dict):
        return Proposal(NONE_ACTION, "params is not an object")
    act = Action(kind, params, _finite(obj.get("est_cost")) or 0.0,
                 _finite(obj.get("est_gain")) or 0.0)
    ok, act, why = validate_action(act)
    return Proposal(act, "" if ok else why)


def allowed_actions_text() -> str:
    return ", ".join(ALLOWED_ACTIONS)
