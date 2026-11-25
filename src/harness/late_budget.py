"""C-only bounded exploration; v14 trainer, predictions and gates are unchanged."""
from __future__ import annotations


def plan_c_budget(budget: dict, *, idle_rounds: int, kill_switch: bool) -> dict:
    if kill_switch or budget.get("rule") == "kill_switch":
        return budget
    result = dict(budget)
    signals = result.get("signals") or {}
    if result.get("rule") == "calibration":
        result.update(epochs=1, reason="C same-stage continuation: one epoch, reserve late-stage budget")
    if int(result.get("epochs", 0)) == 0 and idle_rounds >= 2:
        result.update(action="finetune", epochs=2, lr=0.0005,
                      rule="c_idle_probe", reason="Two idle rounds: bounded two-epoch C probe")
    # A real increase in rows is sufficient to justify one bounded attempt.
    if signals.get("data_grew") and int(result.get("epochs", 0)) == 0:
        result.update(action="finetune", epochs=3, lr=0.0015,
                      rule="c_growth_probe", reason="New C training rows: bounded continuation")
    return result


def protect_c_attempt(decision: dict, budget: dict) -> dict:
    """Protect fresh data / idle probes from hold; never override rollback/kill."""
    signals = budget.get("signals") or {}
    needed = signals.get("data_grew") or signals.get("stage_changed") or budget.get("rule") == "c_idle_probe"
    if (needed and budget.get("epochs", 0) > 0
            and decision.get("model_action") == "hold"):
        return {**decision, "action": "finetune", "epochs": budget["epochs"],
                "lr": budget["lr"], "rule": budget["rule"], "reason": budget["reason"],
                "source": "rule", "override": "c_required_probe"}
    return decision


def pace_c_budget(decision: dict, *, spent: int, round_index: int) -> dict:
    """Keep 3 of the existing 13 requested epochs until the last 3 rounds.

    This is a reserve, not extra compute. Charge requested epochs even if early
    stopping or failure prevents completion. Explicit rollback remains untouched.
    """
    remaining = max(0, 13 - max(0, int(spent)))
    reserve = min(3, remaining) if round_index < 8 else 0
    allowed = max(0, remaining - reserve)
    requested = max(0, int(decision.get("epochs", 0)))
    epochs = min(requested, allowed)
    result = {**decision, "epoch_budget": {"limit": 13, "spent_before": int(spent),
              "remaining_before": remaining, "reserved": reserve, "allowed": allowed}}
    if epochs != requested:
        result.update(epochs=epochs, action="finetune" if epochs else "none",
                      lr=decision["lr"] if epochs else 0.0,
                      rule="c_late_reserve" if remaining and reserve else "epoch_limit",
                      source="budget", reason="C 13-epoch limit with late-stage reserve")
    return result


def execution_record(decision, result, promoted, training_error):
    requested = int(decision.get("epochs", 0))
    executed = int(result.config["epochs_run"]) if result is not None else None if requested else 0
    status = ("training_error" if requested and training_error else
              "promoted" if promoted else "trained_rejected" if result is not None else "skipped")
    return {"status": status, "epochs_requested": requested,
            "epochs_executed": executed, "candidate_available": result is not None,
            "promoted": bool(promoted), "skip_reason": decision.get("rule") if not requested else None,
            "epoch_budget": decision.get("epoch_budget"), "override": decision.get("override")}
