"""C-only bounded exploration and late-stage epoch reserve."""
from __future__ import annotations


def plan_c_budget(
    budget: dict,
    *,
    idle_rounds: int,
    kill_switch: bool,
    idle_threshold: int = 2,
    min_calib_gain: float = 0.0,
) -> dict:
    if kill_switch or budget.get("rule") == "kill_switch":
        return budget
    result = dict(budget)
    signals = result.get("signals") or {}
    if result.get("rule") == "calibration":
        last_gain = float(signals.get("last_gain", 0.0) or 0.0)
        if min_calib_gain > 0.0 and last_gain < min_calib_gain:
            result.update(
                action="none",
                epochs=0,
                lr=0.0,
                rule="c_skip_small_chase",
                reason="C same-stage gain below chase threshold; save CPU and late-stage budget",
            )
        else:
            result.update(
                epochs=1,
                reason="C same-stage continuation: one epoch, reserve late-stage budget",
            )
    if int(result.get("epochs", 0)) == 0 and idle_rounds >= int(idle_threshold):
        result.update(
            action="finetune",
            epochs=2,
            lr=0.0005,
            rule="c_idle_probe",
            reason=f"{idle_rounds} idle rounds: bounded two-epoch C probe",
        )
    # A real increase in rows is sufficient to justify one bounded attempt.
    if signals.get("data_grew") and int(result.get("epochs", 0)) == 0:
        result.update(
            action="finetune",
            epochs=3,
            lr=0.0015,
            rule="c_growth_probe",
            reason="New C training rows: bounded continuation",
        )
    return result


def protect_c_attempt(decision: dict, budget: dict) -> dict:
    """Protect fresh data / idle probes from hold; never override rollback/kill."""
    signals = budget.get("signals") or {}
    needed = (
        signals.get("data_grew")
        or signals.get("stage_changed")
        or budget.get("rule") == "c_idle_probe"
    )
    if (
        needed
        and budget.get("epochs", 0) > 0
        and decision.get("model_action") == "hold"
    ):
        return {
            **decision,
            "action": "finetune",
            "epochs": budget["epochs"],
            "lr": budget["lr"],
            "rule": budget["rule"],
            "reason": budget["reason"],
            "source": "rule",
            "override": "c_required_probe",
        }
    return decision


def pace_c_budget(
    decision: dict,
    *,
    spent: int,
    round_index: int,
    limit: int = 13,
    reserve_epochs: int = 3,
    release_round: int = 8,
) -> dict:
    """Reserve late-stage C epochs until the final rounds without adding compute."""
    limit = max(1, int(limit))
    remaining = max(0, limit - max(0, int(spent)))
    reserve = min(int(reserve_epochs), remaining) if round_index < int(release_round) else 0
    allowed = max(0, remaining - reserve)
    requested = max(0, int(decision.get("epochs", 0)))
    epochs = min(requested, allowed)
    result = {
        **decision,
        "epoch_budget": {
            "limit": limit,
            "spent_before": int(spent),
            "remaining_before": remaining,
            "reserved": reserve,
            "allowed": allowed,
        },
    }
    if epochs != requested:
        result.update(
            epochs=epochs,
            action="finetune" if epochs else "none",
            lr=decision["lr"] if epochs else 0.0,
            rule="c_late_reserve" if remaining and reserve else "epoch_limit",
            source="budget",
            reason=f"C {limit}-epoch limit with late-stage reserve",
        )
    return result


def execution_record(decision, result, promoted, training_error) -> dict:
    requested = int(decision.get("epochs", 0))
    executed = int(result.config["epochs_run"]) if result is not None else None if requested else 0
    status = (
        "training_error"
        if requested and training_error
        else "promoted"
        if promoted
        else "trained_rejected"
        if result is not None
        else "skipped"
    )
    return {
        "status": status,
        "epochs_requested": requested,
        "epochs_executed": executed,
        "candidate_available": result is not None,
        "promoted": bool(promoted),
        "skip_reason": decision.get("rule") if not requested else None,
        "epoch_budget": decision.get("epoch_budget"),
        "override": decision.get("override"),
    }
