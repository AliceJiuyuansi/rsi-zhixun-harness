"""Replay the v1 trial report through the v2 budget rules (counterfactual).

The v1 report gives, for every round, the stage, the observed drift PSI, the
observed decision and the before/after validation metrics. Feeding those same
signals to `choose_budget` answers one question:

    under the same conditions, how much of the v1 training spend was
    avoidable once the state bug and the stall rule are fixed?

This is an *analysis*, not a new experiment: outcomes are taken from v1 where a
candidate was produced, and the replay is labelled as counterfactual.

    python scripts/replay_trial.py
"""
from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.harness.round_controller import acceptance, choose_budget, metric_value  # noqa: E402

REPORT = ROOT / "_state" / "trial_report_completed.json"
OUT_JSON = ROOT / "reports" / "v2_replay_v1.json"
OUT_MD = ROOT / "reports" / "v2_replay_v1.md"


def baseline_metrics(dataset: str) -> dict:
    path = ROOT / "data/raw" / f"official_{dataset.lower()}" / "training.json"
    training = json.loads(path.read_text(encoding="utf-8"))
    return training["validation_after"]


def candidate_metrics(round_record: dict) -> dict | None:
    candidates = round_record.get("candidates") or []
    if not candidates:
        return None
    return candidates[0].get("validation_metrics")


def before_metrics(round_record: dict) -> dict | None:
    """The parent score recorded inside the candidate's change_summary.

    ``round.validation_metrics`` is the *selected* artifact (after promotion),
    so it cannot be the "before"; the change summary is the only place the
    platform-facing record keeps the parent's score.
    """
    candidates = round_record.get("candidates") or []
    if candidates:
        summary = candidates[0].get("change_summary") or ""
        match = re.search(r"before=(\{.*?\})\s+after=", summary)
        if match:
            try:
                parsed = ast.literal_eval(match.group(1))
                if isinstance(parsed, dict):
                    return parsed
            except (ValueError, SyntaxError):
                pass
    return round_record.get("validation_metrics")


def replay_dataset(dataset: dict) -> dict:
    key = dataset["key"]
    task = "classification" if key == "A" else "regression"
    before = baseline_metrics(key)
    name, value, higher = metric_value(task, before)
    state = {"best": value, "last": value, "baseline": value,
             "stalled": 0, "last_gain": 0.0, "stage_id": ""}
    rows = []
    for record in dataset["rounds"]:
        stage = record.get("stage_id", "")
        configuration = record.get("configuration") or {}
        v1_epochs = int(float(configuration.get("epochs", 0) or 0))
        psi = float(configuration.get("psi", 0.0) or 0.0)
        stage_changed = stage != state["stage_id"]
        parent_metrics = before_metrics(record) or before
        _, current_before, _ = metric_value(task, parent_metrics)
        if stage_changed:
            state["best"] = current_before
        gap = (state["best"] - current_before) if higher else \
            max(current_before - state["best"], 0.0) / max(abs(state["best"]), 1e-9)
        history = {"stalled": state["stalled"], "last_gain": state["last_gain"]}
        budget = choose_budget(
            dataset_key=key, task=task, round_index=int(record["round_index"]),
            psi_prev=psi if stage_changed else 0.0, psi_holdout=psi,
            data_grew=stage_changed, stage_changed=stage_changed, gap=gap,
            history=history)
        v2_epochs = int(budget["epochs"])
        observed = candidate_metrics(record)
        v1_decision = record.get("decision", "")
        if v2_epochs > 0 and observed:
            outcome = acceptance(parent_metrics, observed, task)
            promoted = bool(outcome["promote"]) and v1_decision == "promote"
        else:
            promoted = False
            outcome = None
        if promoted and observed:
            _, new_value, _ = metric_value(task, observed)
            state["best"] = max(state["best"], new_value) if higher else min(state["best"], new_value)
            state["last"] = new_value
            state["last_gain"] = outcome["gain"]
            state["stalled"] = 0
        elif v2_epochs > 0:
            state["stalled"] += 1
            state["last_gain"] = 0.0
        if not promoted:
            state["last"] = current_before
        state["stage_id"] = stage
        rows.append({
            "round": int(record["round_index"]), "stage": stage,
            "v1_epochs": v1_epochs, "v2_epochs": v2_epochs,
            "v1_decision": v1_decision, "v2_rule": budget["rule"],
            "v2_decision": "promote" if promoted else ("retain" if v2_epochs else "no_train"),
            "psi": round(psi, 4), "stage_changed": stage_changed,
            "gap": round(gap, 6),
        })
    v1_total = sum(item["v1_epochs"] for item in rows)
    v2_total = sum(item["v2_epochs"] for item in rows)
    return {
        "dataset": key, "task": task, "metric": name,
        "v1_epochs": v1_total, "v2_epochs": v2_total,
        "saved_epochs": v1_total - v2_total,
        "v1_cpu_seconds": dataset.get("cpu_seconds"),
        "estimated_saved_cpu_seconds": round((dataset.get("cpu_seconds") or 0.0)
                                             * (v1_total - v2_total) / max(v1_total, 1), 1),
        "rounds": rows,
    }


def main() -> int:
    payload = json.loads(REPORT.read_text(encoding="utf-8"))
    datasets = {item["key"]: item for item in payload["report"]["workspace_report"]["datasets"]}
    results = [replay_dataset(datasets[key]) for key in ("A", "B", "C") if key in datasets]
    summary = {
        "source": "v1 trial report (2026-11-21, submission v1)",
        "note": "counterfactual replay: outcomes copied from v1, v2 budget rules applied to the same signals",
        "datasets": results,
        "totals": {
            "v1_epochs": sum(item["v1_epochs"] for item in results),
            "v2_epochs": sum(item["v2_epochs"] for item in results),
            "saved_epochs": sum(item["saved_epochs"] for item in results),
            "v1_cpu_seconds": round(sum(item["v1_cpu_seconds"] or 0.0 for item in results), 1),
            "estimated_saved_cpu_seconds": round(sum(item["estimated_saved_cpu_seconds"] for item in results), 1),
        },
    }
    OUT_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# v1 试运行报告 → v2 预算规则回放（反事实）",
        "",
        "> 数据源：`_state/trial_report_completed.json`（v1 试运行，10 轮 × A/B/C）。",
        "> **这是回放分析，不是新实验**：结果沿用 v1 已观测到的候选分数，只把同一组信号",
        "> 输入 v2 的 `choose_budget`，用来看 v1 的训练预算里有多少是当时就能省掉的。",
        "",
        "| 数据集 | v1 epochs | v2 epochs | 省下 | v1 CPU 秒 | 估计省下 CPU 秒 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for item in results:
        lines.append(f"| {item['dataset']} | {item['v1_epochs']} | {item['v2_epochs']} | "
                     f"{item['saved_epochs']} | {item['v1_cpu_seconds']:.0f} | "
                     f"{item['estimated_saved_cpu_seconds']:.0f} |")
    totals = summary["totals"]
    lines.append(f"| **合计** | **{totals['v1_epochs']}** | **{totals['v2_epochs']}** | "
                 f"**{totals['saved_epochs']}** | **{totals['v1_cpu_seconds']:.0f}** | "
                 f"**{totals['estimated_saved_cpu_seconds']:.0f}** |")
    lines += ["", "## 逐轮对照", ""]
    for item in results:
        lines.append(f"### {item['dataset']}（{item['metric']}）")
        lines.append("")
        lines.append("| 轮 | 阶段 | v1 epochs | v1 决策 | v2 epochs | v2 规则 | v2 决策 | PSI |")
        lines.append("|---:|---|---:|---|---:|---|---|---:|")
        for row in item["rounds"]:
            lines.append(f"| {row['round']} | {row['stage']} | {row['v1_epochs']} | "
                         f"{row['v1_decision']} | {row['v2_epochs']} | {row['v2_rule']} | "
                         f"{row['v2_decision']} | {row['psi']:.3f} |")
        lines.append("")
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary["totals"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
