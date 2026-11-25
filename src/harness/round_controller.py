"""Round-level controller for the platform harness (protocol v2).

Everything here is a pure function of numbers and dictionaries, so the
interesting decisions can be replayed and unit-tested without a host process.
The platform entry (`harness/contest_submission.py`) only wires I/O around it.

Why this module exists
----------------------
The first trial (v1, 2026-11-21) completed all 10 rounds on A/B/C, but the
report exposed four problems that only show up on the real host:

1. regression state initialised ``best_val`` to ``0.0``; every regression round
   then looked like a 600-unit drop and retrained 5 epochs unconditionally;
2. ``ModelReceipt.text`` was empty for Qwen3.5 (the reply lives in
   ``reasoning_content``), so every round finished with ``reason="no model text"``;
3. candidates that were *not* better (rounds 4-10 on C, 3-10 on B) were still
   retrained every round before being rejected - pure waste;
4. drift was measured against the *current* train/validation split, not against
   the previous stage or the model's own reference, so it could not tell
   "new stage arrived" from "same data, different model".

The controller keeps a compact reference profile in native state and makes the
budget a function of (stage change, data growth, drift, last gain, stalls).
"""
from __future__ import annotations

import json
import math
import re
from typing import Any

import numpy as np

# --- drift bands (same semantics as src/policy/strategy.py) --------------
PSI_STABLE = 0.10
PSI_MILD = 0.25

# --- acceptance / gain thresholds ---------------------------------------
CLASS_ACCEPT_TOL = 0.0        # "not worse" is the rule; no free 0.1-point loss
REG_ACCEPT_TOL = 0.0          # relative RMSE must not increase either
CLASS_MIN_GAIN = 0.002        # a promotion must gain at least this much F1
REG_MIN_GAIN = 0.002          # ... or 0.2% relative RMSE
STALL_LIMIT = 2               # two fruitless updates in a row -> stop spending

# --- epoch caps per dataset (measured on the v1 trial) ------------------
EPOCH_CAP = {"A": 5, "B": 4, "C": 3}
FIRST_ROUND_EPOCHS = {"A": 5, "B": 5, "C": 3}
FIRST_ROUND_LR = {"A": 2e-3, "B": 2e-3, "C": 2e-3}
CALIBRATION_EPOCHS = {"classification": 1, "regression": 3}
CALIBRATION_LR = {"classification": 1e-3, "regression": 1e-3}
RECOVER_EPOCHS = {"A": 3, "B": 2, "C": 2}
RECOVER_LR = 1.5e-3
NEW_STAGE_EPOCHS = {"A": 3, "B": 3, "C": 3}

_EPS = 1e-9


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _normalise(shares) -> np.ndarray | None:
    array = np.asarray(shares, dtype=float).reshape(-1)
    if array.size == 0 or not np.all(np.isfinite(array)) or array.sum() <= 0:
        return None
    return np.clip(array / array.sum(), 1e-6, None)


# --------------------------------------------------------------- profiles
def feature_profile(rows, bins: int = 8) -> list[dict] | None:
    """Per-feature histogram profile: quantile edges plus reference shares.

    ``shares`` are what later rounds compare against; keeping only the edges
    would make it impossible to rebuild the reference distribution exactly.
    """
    matrix = np.asarray(rows, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] == 0 or len(matrix) < bins * 2:
        return None
    profile: list[dict] = []
    for column in range(matrix.shape[1]):
        values = matrix[:, column]
        if not np.all(np.isfinite(values)):
            profile.append({})
            continue
        edges = np.unique(np.quantile(values, np.linspace(0.0, 1.0, bins + 1)))
        if edges.size < 3:
            profile.append({})
            continue
        counts, _ = np.histogram(values, bins=edges)
        shares = _normalise(counts)
        if shares is None:
            profile.append({})
            continue
        profile.append({"edges": [round(float(v), 6) for v in edges],
                        "shares": [round(float(v), 6) for v in shares]})
    return profile


def psi_against(profile, rows) -> float:
    """Largest PSI between the stored reference profile and the new rows."""
    matrix = np.asarray(rows, dtype=float)
    if not profile or matrix.ndim != 2 or len(matrix) < 10:
        return 0.0
    if len(profile) != matrix.shape[1]:
        return 0.0
    scores: list[float] = []
    for column, reference in enumerate(profile):
        if not isinstance(reference, dict) or not reference.get("edges"):
            continue
        edges = np.asarray(reference["edges"], dtype=float)
        expected = _normalise(reference.get("shares") or [])
        if expected is None or edges.size < 3:
            continue
        values = matrix[:, column]
        if not np.all(np.isfinite(values)):
            continue
        actual_counts, _ = np.histogram(values, bins=edges)
        actual = _normalise(actual_counts)
        if actual is None:
            continue
        scores.append(float(np.sum((actual - expected) * np.log(actual / expected))))
    return float(max(scores)) if scores else 0.0


def js_divergence(left, right) -> float:
    """Jensen-Shannon distance (sqrt of divergence), 0 = identical."""
    first = _normalise(left)
    second = _normalise(right)
    if first is None or second is None or first.size != second.size:
        return 0.0
    middle = 0.5 * (first + second)
    divergence = 0.5 * np.sum(first * np.log(first / middle)) + \
        0.5 * np.sum(second * np.log(second / middle))
    return float(math.sqrt(max(divergence, 0.0)))


def prediction_profile(predictions, task: str, classes: list[str],
                       reference_edges=None) -> dict | None:
    """Class distribution (classification) or prediction histogram (regression)."""
    array = np.asarray(predictions, dtype=float)
    if array.ndim == 0 or len(array) == 0:
        return None
    if task == "classification":
        matrix = array if array.ndim == 2 else array.reshape(1, -1)
        if matrix.shape[1] != len(classes):
            return None
        labels = matrix.argmax(axis=1)
        counts = np.bincount(labels, minlength=len(classes)).astype(float)
        shares = _normalise(counts)
        if shares is None:
            return None
        return {"kind": "class", "labels": list(classes),
                "shares": [round(float(v), 6) for v in shares],
                "mean_confidence": round(float(matrix.max(axis=1).mean()), 6)}

    values = array.reshape(-1)
    values = values[np.isfinite(values)]
    if values.size < 10:
        return None
    if reference_edges is not None and len(reference_edges) >= 3:
        edges = np.asarray(reference_edges, dtype=float)
    else:
        edges = np.unique(np.quantile(values, np.linspace(0.0, 1.0, 11)))
        if edges.size < 3:
            return None
    counts, _ = np.histogram(values, bins=edges)
    shares = _normalise(counts)
    if shares is None:
        return None
    return {"kind": "value", "edges": [round(float(v), 6) for v in edges],
            "shares": [round(float(v), 6) for v in shares]}


def metric_value(task: str, metrics: dict | None) -> tuple[str, float, bool]:
    """Return (metric name, native value, higher_is_better)."""
    payload = metrics or {}
    if task == "classification":
        if "macro_f1" in payload:
            return "macro_f1", _finite(payload["macro_f1"]), True
        return "accuracy", _finite(payload.get("accuracy")), True
    if "rmse" in payload:
        return "rmse", _finite(payload["rmse"]), False
    return "mae", _finite(payload.get("mae")), False


def score_gap(best: float, last: float, higher_is_better: bool) -> float:
    """How far the last selected model fell behind the best one (0 = no drop)."""
    if higher_is_better:
        return max(best - last, 0.0)
    return max(last - best, 0.0) / max(abs(best), _EPS)


# --------------------------------------------------------------- verdicts
def extract_text_with_source(response: dict | None) -> tuple[str, str]:
    """Prefer the final answer. Reasoning is a parse-only, never-log fallback."""
    if not isinstance(response, dict):
        return "", ""
    choices = response.get("choices")
    first = choices[0] if isinstance(choices, list) and choices and isinstance(choices[0], dict) else {}
    message = first.get("message")
    message = message if isinstance(message, dict) else {}
    for key, value in (("content", message.get("content")), ("output_text", response.get("output_text")),
                       ("reasoning_content", message.get("reasoning_content")), ("reasoning", message.get("reasoning"))):
        if isinstance(value, list):
            value = "".join(item["text"] for item in value
                            if isinstance(item, dict) and isinstance(item.get("text"), str))
        if isinstance(value, str) and value.strip():
            return value.strip(), key
    return "", ""


def extract_text(response: dict | None) -> str:
    return extract_text_with_source(response)[0]


def _normalise_verdict(payload: dict) -> dict | None:
    decision = payload.get("decision", payload.get("action", payload.get("verdict")))
    if not isinstance(decision, str):
        return None
    decision = decision.strip().lower()
    if decision in ("train", "continue", "update", "finetune", "retrain"):
        action = "train"
    elif decision in ("hold", "keep", "none", "wait", "noop", "freeze"):
        action = "hold"
    elif decision in ("rollback", "revert", "reject"):
        action = "rollback"
    else:
        return None
    try:
        return {
            "action": action,
            "epochs": _finite(payload.get("epochs"), 0.0),
            "lr": _finite(payload.get("lr", payload.get("learning_rate")), 0.0),
            "reason": str(payload.get("reason", ""))[:400],
        }
    except (OverflowError, RecursionError):
        return None


def parse_truncated_verdict(text: str, budget: dict | None = None) -> dict | None:
    """Repair only an unambiguous flat decision prefix, never infer an action from prose."""
    if not isinstance(text, str) or "{" not in text:
        return None
    fragment = text[text.index("{"):].rstrip()
    if not re.match(r'\{\s*"(?:decision|action)"\s*:\s*"[a-zA-Z]+"', fragment):
        return None
    # Only a reason is allowed to end mid-string. Never close an unfinished
    # decision or trust a final digit that could be part of a longer number.
    if re.search(r',\s*"reason"\s*:\s*"(?:[^"\\\x00-\x1f]|\\.)*\Z', fragment):
        fragment += '"'
    if fragment.endswith(','):
        fragment = fragment[:-1]
    elif not fragment.endswith('"'):
        return None
    try:
        pairs = json.loads(fragment + '}', object_pairs_hook=list, parse_int=float)
    except (ValueError, RecursionError):
        return None
    payload = dict(pairs)
    allowed = {"decision", "action", "epochs", "lr", "learning_rate", "reason"}
    if (len(payload) != len(pairs) or not payload.keys() <= allowed
            or {"decision", "action"} <= payload.keys() or {"lr", "learning_rate"} <= payload.keys()
            or not isinstance(payload.get("reason", ""), str)):
        return None
    for key in ("epochs", "lr", "learning_rate"):
        if key in payload and (type(payload[key]) not in (int, float) or _finite(payload[key], -1.0) < 0):
            return None
    payload["reason"] = "truncated_json_recovered"
    if budget:
        payload.setdefault("epochs", budget.get("epochs"))
        if "learning_rate" not in payload:
            payload.setdefault("lr", budget.get("lr"))
    return _normalise_verdict(payload)


def parse_verdict(text: str, *, allow_truncated: bool = True) -> dict | None:
    """Read complete JSON first. Live callers must explicitly gate truncation on the receipt."""
    if not isinstance(text, str) or not text.strip():
        return None
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text):
        try:
            payload, _ = decoder.raw_decode(text, match.start())
        except (ValueError, RecursionError):
            continue
        if isinstance(payload, dict):
            verdict = _normalise_verdict(payload)
            if verdict is not None:
                return verdict
    return parse_truncated_verdict(text) if allow_truncated else None


# --------------------------------------------------------------- history
def summarise_history(records, state: dict, task: str) -> dict:
    """Turn platform history + native state into the few numbers we act on."""
    points: list[dict] = []
    for record in records or []:
        if not isinstance(record, dict):
            continue
        configuration = record.get("configuration") or {}
        metrics = record.get("validation_metrics") or {}
        if not metrics:
            candidates = record.get("candidates") or []
            if candidates and isinstance(candidates[0], dict):
                metrics = candidates[0].get("validation_metrics") or {}
        name, _, _ = metric_value(task, metrics)
        value = _finite(metrics.get(name), float("nan"))
        # The submitting harness measured gain on one holdout. Scores from
        # adjacent stages or from different serving policies are not paired.
        gain = _finite(configuration.get("gain"), float("nan"))
        points.append({
            "round": record.get("round_index"),
            "decision": record.get("decision", ""),
            "epochs": int(_finite(configuration.get("epochs"), 0.0)),
            "gain": round(gain, 6) if math.isfinite(gain) else None,
            "value": round(value, 6) if math.isfinite(value) else None,
        })

    native = state.get("decisions") if isinstance(state.get("decisions"), list) else []
    if not points and native:
        points = [dict(item) for item in native if isinstance(item, dict)]

    last_gain = _finite(state.get("last_gain"), 0.0)
    stalled = int(_finite(state.get("stalled"), 0.0))
    last_rejected = bool(state.get("last_rejected", False))
    if "last_gain" not in state and points:
        gains = [p["gain"] for p in points if p.get("gain") is not None]
        if gains:
            last_gain = float(gains[-1])
    if "stalled" not in state and points:
        trailing = 0
        for point in reversed(points):
            gain = point.get("gain")
            if point.get("decision") == "retain" or (gain is not None and gain < CLASS_MIN_GAIN):
                trailing += 1
            else:
                break
        stalled = trailing
        last_rejected = bool(points and points[-1].get("decision") in ("retain", "rollback"))

    recent = points[-3:]
    return {
        "recent": recent,
        "last_gain": round(float(last_gain), 6),
        "stalled": stalled,
        "last_rejected": last_rejected,
        "evidence": "；".join(
            f"第{item.get('round')}轮 {item.get('decision') or '?'}"
            f"(epochs={item.get('epochs')}, gain={item.get('gain')})"
            for item in recent) or "无历史轮次",
    }


# --------------------------------------------------------------- budget
def choose_budget(*, dataset_key: str, task: str, round_index: int,
                  psi_prev: float, psi_holdout: float, data_grew: bool,
                  stage_changed: bool, gap: float, history: dict) -> dict:
    """Return the rule-based budget plus the sentence that justifies it.

    Precedence (cheap first):
      round 1 -> real attempt; new stage -> moderate; recovery -> moderate;
      last update paid off -> one cheap calibration; otherwise -> no spend.
    """
    cap = EPOCH_CAP.get(dataset_key, 3)
    stalled = int(history.get("stalled", 0))
    last_gain = _finite(history.get("last_gain"), 0.0)
    psi = max(_finite(psi_prev), 0.0)
    signals = {"psi_prev": round(psi, 4), "psi_holdout": round(_finite(psi_holdout), 4),
               "data_grew": bool(data_grew), "stage_changed": bool(stage_changed),
               "gap": round(_finite(gap), 6), "stalled": stalled,
               "last_gain": round(last_gain, 6)}

    if history.get("kill_switch"):
        return {"action": "none", "epochs": 0, "lr": 0.0, "rule": "kill_switch",
                "reason": "kill-switch latched; keep the parent model", "signals": signals}

    if round_index <= 1:
        epochs = FIRST_ROUND_EPOCHS.get(dataset_key, 3)
        return {"action": "finetune", "epochs": epochs,
                "lr": FIRST_ROUND_LR.get(dataset_key, 2e-3), "rule": "first_round",
                "reason": "首轮：在父检查点上做一次真实续训，给后续轮次一个可比较的起点",
                "signals": signals}

    if dataset_key == "A" and task == "classification":
        return {"action": "finetune", "epochs": 3 if data_grew else 1,
                "lr": RECOVER_LR if data_grew else CALIBRATION_LR[task],
                "rule": "a_v1_fresh" if data_grew else "a_v1_quiet",
                "reason": "A follows the v1 budget: three epochs on new rows, one on a quiet round",
                "signals": signals}
    if dataset_key == "C" and stage_changed:
        epochs = min(NEW_STAGE_EPOCHS.get("C", 3), cap)
        return {"action": "finetune", "epochs": epochs, "lr": RECOVER_LR,
                "rule": "c_stage_probe",
                "reason": f"Adapt C to the new stage with {epochs} epochs",
                "signals": signals}

    if stalled >= STALL_LIMIT and not stage_changed:
        if data_grew:
            return {"action": "finetune", "epochs": 1, "lr": CALIBRATION_LR.get(task, 1e-3),
                    "rule": "new_data_probe",
                    "reason": "此前连续无收益，但训练数据又增长了；只用 1 个 epoch 做探针，避免重复浪费",
                    "signals": signals}
        return {"action": "none", "epochs": 0, "lr": 0.0, "rule": "stalled",
                "reason": f"连续 {stalled} 轮更新没有带来可测提升，停止消耗算力，保持父模型",
                "signals": signals}

    new_data = stage_changed or data_grew
    if new_data:
        label = "新阶段数据到达" if stage_changed else "同一阶段内训练数据增长"
        if psi >= PSI_MILD:
            epochs = min(NEW_STAGE_EPOCHS.get(dataset_key, 3), cap)
            rule = "new_stage_drift"
            why = f"{label}，且输入相对上一轮移动 PSI={psi:.3f} 超阈值 {PSI_MILD}，做一次中等预算适配"
        else:
            epochs = min(CALIBRATION_EPOCHS.get(task, 1) + 1, cap)
            rule = "new_stage"
            why = f"{label}，输入移动较小（PSI={psi:.3f}），只做一次小步适配"
        return {"action": "finetune", "epochs": epochs, "lr": RECOVER_LR,
                "rule": rule, "reason": why, "signals": signals}

    if gap >= CLASS_MIN_GAIN and psi >= PSI_MILD:
        return {"action": "finetune", "epochs": min(RECOVER_EPOCHS.get(dataset_key, 2), cap),
                "lr": RECOVER_LR, "rule": "covariate_recover",
                "reason": f"输入漂移 PSI={psi:.3f} 且指标回落 {gap:.4f}，做恢复性续训",
                "signals": signals}

    if gap >= 4 * CLASS_MIN_GAIN:
        return {"action": "finetune", "epochs": min(RECOVER_EPOCHS.get(dataset_key, 2), cap),
                "lr": RECOVER_LR, "rule": "concept_recover",
                "reason": f"输入未大动但指标回落 {gap:.4f}，按 concept drift 做小步恢复",
                "signals": signals}

    if last_gain > 0 and stalled == 0:
        epochs = min(CALIBRATION_EPOCHS.get(task, 1), cap)
        return {"action": "finetune", "epochs": epochs,
                "lr": CALIBRATION_LR.get(task, 1e-3), "rule": "calibration",
                "reason": f"上一轮有正收益(+{last_gain:g})，用 {epochs} 个 epoch 做廉价校准",
                "signals": signals}

    return {"action": "none", "epochs": 0, "lr": 0.0, "rule": "quiet",
            "reason": "无新阶段、无显著漂移、上轮无正收益；本轮只观察，不花算力",
            "signals": signals}


def apply_verdict(budget: dict, verdict: dict | None, history: dict,
                  round_index: int) -> dict:
    """Let the model *reduce* the rule budget, never silently enlarge it."""
    if not verdict:
        return {**budget, "model_action": "unavailable", "model_reason": ""}
    if budget.get("action") == "none" or int(_finite(budget.get("epochs"), 0)) <= 0:
        return {**budget, "action": "none", "epochs": 0, "lr": 0.0,
                "model_action": "train_ignored_rule_none" if verdict["action"] == "train"
                else verdict["action"],
                "model_reason": verdict.get("reason", "")}
    model_action = verdict["action"]
    if model_action == "rollback":
        return {**budget, "action": "none", "epochs": 0, "lr": 0.0,
                "rule": "model_rollback", "model_action": "rollback",
                "model_reason": verdict.get("reason", ""),
                "reason": "模型裁决回滚到父模型；本轮不训练"}
    if model_action == "hold":
        signals = budget.get("signals") or {}
        drift_justifies_update = bool(signals.get("stage_changed")) and \
            _finite(signals.get("psi_prev"), 0.0) >= PSI_MILD
        if round_index <= 1 or drift_justifies_update:
            return {**budget, "model_action": "hold_ignored_drift",
                    "model_reason": verdict.get("reason", "")}
        return {**budget, "action": "none", "epochs": 0, "lr": 0.0,
                "rule": "model_hold", "model_action": "hold",
                "model_reason": verdict.get("reason", ""),
                "reason": "模型裁决保持父模型；本轮不训练"}
    if round_index <= 1:
        # The first round is the only guaranteed real attempt; do not let a
        # 4B model shave it down to a token update.
        return {**budget, "model_action": "train_ignored_first_round",
                "model_reason": verdict.get("reason", "")}
    # "train": adopt the smaller epoch count, keep the rule's lr unless the
    # model proposed a sane one. A model may reduce a justified update, but it
    # may not turn a real attempt into a token one.
    floor = 1
    if budget.get("rule") in ("first_round", "new_stage_drift", "covariate_recover"):
        floor = min(int(budget.get("epochs", 1)), 2)
    proposed_epochs = int(verdict.get("epochs") or budget.get("epochs", 0))
    epochs = max(floor, min(int(budget.get("epochs", 0)), proposed_epochs))
    if epochs <= 0:
        epochs = int(budget.get("epochs", 0))
    lr = _finite(verdict.get("lr"), 0.0)
    lr = lr if 1e-4 <= lr <= 1e-2 else budget.get("lr", 1e-3)
    return {**budget, "epochs": epochs, "lr": lr, "action": "finetune" if epochs else "none",
            "model_action": "train", "model_reason": verdict.get("reason", "")}


# --------------------------------------------------------------- acceptance
def _bearing_macro_gain(before: dict, after: dict) -> float | None:
    """Mean relative RMSE improvement over comparable, non-degenerate bearings."""
    left, right = before.get("by_group"), after.get("by_group")
    if not isinstance(left, dict) or not isinstance(right, dict):
        return None
    gains = []
    for key in sorted(left.keys() & right.keys()):
        if not isinstance(left[key], dict) or not isinstance(right[key], dict):
            continue
        old = _finite(left[key].get("rmse"), -1.0)
        new = _finite(right[key].get("rmse"), -1.0)
        if old > _EPS and new >= 0:
            gains.append((old - new) / old)
    return sum(gains) / len(gains) if len(gains) >= 2 else None


def acceptance(before: dict, after: dict, task: str, dataset_key: str = "") -> dict:
    """Should the candidate replace the parent, and is the gain worth a upload?"""
    before_name, before_value, higher = metric_value(task, before)
    after_name, after_value, _ = metric_value(task, after)
    if higher:
        gain = after_value - before_value
        if dataset_key == "A":
            # The v1 macro-F1 tolerance also promoted ties and small regressions.
            accepted = after_value >= before_value - 0.02
            promote = accepted
        else:
            accepted = gain >= -1e-9
            promote = gain >= CLASS_MIN_GAIN
    else:
        gain = (before_value - after_value) / max(abs(before_value), _EPS)
        effective_gain = gain
        accepted = gain >= -1e-9
        nasa_bonus = False
        if dataset_key == "B" and before_name == after_name == "rmse" and gain >= -0.0005:
            macro = _bearing_macro_gain(before, after)
            if macro is not None:
                effective_gain = max(gain, 0.70 * gain + 0.30 * macro)
                accepted = accepted or effective_gain >= REG_MIN_GAIN
        if dataset_key == "C" and before_name == after_name == "rmse":
            delta = _finite(after.get("nasa_log1p"), math.nan) - _finite(before.get("nasa_log1p"), math.nan)
            if delta > 0.50:
                accepted = False
            nasa_bonus = gain >= 0.0 and delta <= -0.08
        promote = effective_gain >= REG_MIN_GAIN or nasa_bonus
    return {
        "metric": after_name or before_name,
        "before": round(float(before_value), 6),
        "after": round(float(after_value), 6),
        "gain": round(float(gain), 6),
        "accepted": bool(accepted),
        "promote": bool(promote and accepted),
    }


def paired_acceptance(before: dict, after: dict, task: str, dataset_key: str) -> dict:
    """Candidate policy: compare only paired scores on the current holdout."""
    result = acceptance(before, after, task, dataset_key)
    primary = "macro_f1" if dataset_key == "A" else "rmse"
    required = (primary, "nasa_log1p") if dataset_key == "C" else (primary,)
    if any(not math.isfinite(_finite(m.get(k), math.nan)) for m in (before, after) for k in required):
        result.update(accepted=False, promote=False, reason="missing_paired_metrics")
    elif dataset_key == "A":
        # Compare the unrounded values: a tie may advance, a real loss may not.
        noninferior = after[primary] >= before[primary] - 1e-9
        result.update(accepted=bool(noninferior), promote=bool(noninferior))
    return result


def clip_regression_prediction(value: float, feature_count: int) -> float:
    """Return a non-negative remaining-life estimate in the original unit."""
    return max(0.0, float(value))


def row_observation(previous: dict, train: int, validation: int, round_index: int) -> dict:
    """从实际加载行数生成 schema3 观测，兼容已有 n_train。"""
    rows = {"train": int(train), "validation": int(validation)}
    previous_rows = previous.get("rows") or {}
    previous_train = int(previous_rows.get("train", previous.get("n_train", 0)))
    history = [item for item in (previous.get("rows_history") or [])
               if isinstance(item, dict) and item.get("round") != int(round_index)]
    history.append({"round": int(round_index), **rows})
    return {"rows": rows, "rows_history": history[-10:], "data_grew": int(train) > previous_train}


def decision_record(*, round_index: int, stage_id: str, dataset_key: str,
                    action: dict, history: dict, psi: dict, model_verdict: str,
                    outcome: dict | None, accepted: bool, region: str = "") -> dict:
    """One compact line for state/decisions.jsonl and the next round's prompt."""
    return {
        "round": int(round_index),
        "stage": stage_id,
        "dataset": dataset_key,
        "rule": action.get("rule", ""),
        "action": action.get("action", "none"),
        "epochs": int(action.get("epochs", 0)),
        "lr": round(_finite(action.get("lr"), 0.0), 6),
        "psi": psi,
        "history": history.get("evidence", ""),
        "model_action": action.get("model_action", "unavailable"),
        "source": action.get("source", "rule"),
        "model_reason": (action.get("model_reason") or "")[:200],
        "model_text": (model_verdict or "")[:200],
        "outcome": outcome or {},
        "accepted": bool(accepted),
        "last_gain": round(_finite(history.get("last_gain"), 0.0), 6),
        "stalled": int(history.get("stalled", 0)),
        "reason": action.get("reason", ""),
        "region": region,
    }
