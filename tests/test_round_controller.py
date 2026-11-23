"""Unit tests for the v2 round controller (no host, no network).

    python -m unittest discover -s tests -v
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.harness.guard import Gatekeeper  # noqa: E402
from src.harness.killswitch import KillSwitch  # noqa: E402
from src.harness.round_controller import (  # noqa: E402
    acceptance,
    apply_verdict,
    choose_budget,
    clip_regression_prediction,
    extract_text,
    feature_profile,
    js_divergence,
    metric_value,
    parse_verdict,
    prediction_profile,
    psi_against,
    score_gap,
    summarise_history,
)
from src.schema import Action, GateDecision  # noqa: E402


class ProfileTests(unittest.TestCase):
    def test_profile_and_psi_are_zero_on_same_rows(self):
        rng = np.random.default_rng(0)
        rows = rng.normal(size=(256, 4))
        profile = feature_profile(rows)
        self.assertIsNotNone(profile)
        self.assertLess(psi_against(profile, rows), 1e-3)

    def test_psi_grows_when_distribution_moves(self):
        rng = np.random.default_rng(1)
        reference = rng.normal(0, 1, size=(256, 3))
        profile = feature_profile(reference)
        moved = reference + 3.0
        self.assertGreater(psi_against(profile, moved), 1.0)

    def test_js_distance(self):
        self.assertAlmostEqual(js_divergence([0.5, 0.5], [0.5, 0.5]), 0.0, places=6)
        self.assertGreater(js_divergence([0.9, 0.1], [0.1, 0.9]), 0.5)

    def test_prediction_profile_classification(self):
        probabilities = np.array([[0.9, 0.1], [0.2, 0.8], [0.7, 0.3]])
        profile = prediction_profile(probabilities, "classification", ["a", "b"])
        self.assertEqual(profile["kind"], "class")
        self.assertEqual(len(profile["shares"]), 2)
        self.assertAlmostEqual(sum(profile["shares"]), 1.0, places=5)


class VerdictTests(unittest.TestCase):
    def test_reasoning_content_is_used_when_content_empty(self):
        response = {"choices": [{"message": {"content": "", "reasoning_content": "decide train"}}]}
        self.assertEqual(extract_text(response), "decide train")

    def test_list_content(self):
        response = {"choices": [{"message": {"content": [{"type": "text", "text": "hold"}]}}]}
        self.assertEqual(extract_text(response), "hold")

    def test_parse_plain_json(self):
        verdict = parse_verdict('{"decision":"train","epochs":2,"lr":0.001,"reason":"drift"}')
        self.assertEqual(verdict["action"], "train")
        self.assertEqual(verdict["epochs"], 2.0)

    def test_parse_fenced_json_with_chatter(self):
        text = '分析如下：\n```json\n{"action":"hold","epochs":0,"reason":"stable"}\n```\n完毕'
        verdict = parse_verdict(text)
        self.assertEqual(verdict["action"], "hold")

    def test_bad_text_returns_none(self):
        self.assertIsNone(parse_verdict("continue training please"))
        self.assertIsNone(parse_verdict(""))


class HistoryTests(unittest.TestCase):
    def test_digest_from_platform_records(self):
        records = [
            {"round_index": 1, "decision": "promote", "configuration": {"epochs": 5},
             "validation_metrics": {"macro_f1": 0.72}},
            {"round_index": 2, "decision": "retain", "configuration": {"epochs": 1},
             "validation_metrics": {"macro_f1": 0.72}},
        ]
        digest = summarise_history(records, {}, "classification")
        self.assertEqual(digest["stalled"], 1)
        self.assertEqual(len(digest["recent"]), 2)
        self.assertIn("第2轮", digest["evidence"])

    def test_state_is_authoritative_when_records_empty(self):
        state = {"last_gain": 0.01, "stalled": 1, "decisions": [
            {"round": 1, "decision": "promote", "epochs": 3, "gain": 0.01}]}
        digest = summarise_history([], state, "classification")
        self.assertEqual(digest["stalled"], 1)
        self.assertAlmostEqual(digest["last_gain"], 0.01)

    def test_paired_history_uses_recorded_gain(self):
        # v3 C round 1 was +0.009805 on the paired holdout, but differencing
        # the platform's post-cap validation scores produced -0.10701.
        records = [
            {"round_index": 1, "decision": "promote",
             "configuration": {"epochs": 3, "gain": 0.009805},
             "validation_metrics": {"rmse": 58.526016}},
            {"round_index": 2, "decision": "promote",
             "configuration": {"epochs": 3, "gain": 0.003823},
             "validation_metrics": {"rmse": 58.311335}},
        ]
        digest = summarise_history(records, {}, "regression")
        self.assertEqual([point["gain"] for point in digest["recent"]],
                         [0.009805, 0.003823])
        self.assertIn("0.009805", digest["evidence"])

    def test_missing_recorded_gain_stays_unknown(self):
        records = [{"round_index": 1, "decision": "retain",
                    "configuration": {"epochs": 1},
                    "validation_metrics": {"macro_f1": 0.70}}]
        digest = summarise_history(records, {}, "classification")
        self.assertIsNone(digest["recent"][0]["gain"])


class BudgetTests(unittest.TestCase):
    def test_first_round_always_trains(self):
        budget = choose_budget(dataset_key="A", task="classification", round_index=1,
                               psi_prev=0.0, psi_holdout=2.0, data_grew=False,
                               stage_changed=False, gap=0.0, history={})
        self.assertEqual(budget["action"], "finetune")
        self.assertEqual(budget["epochs"], 5)

    def test_stalled_run_stops_spending(self):
        budget = choose_budget(dataset_key="C", task="regression", round_index=6,
                               psi_prev=0.01, psi_holdout=0.01, data_grew=False,
                               stage_changed=False, gap=0.0, history={"stalled": 2})
        self.assertEqual(budget["epochs"], 0)
        self.assertEqual(budget["rule"], "stalled")

    def test_new_stage_with_drift_gets_budget(self):
        budget = choose_budget(dataset_key="A", task="classification", round_index=5,
                               psi_prev=0.6, psi_holdout=0.2, data_grew=True,
                               stage_changed=True, gap=0.0, history={"stalled": 0})
        self.assertEqual(budget["action"], "finetune")
        self.assertGreaterEqual(budget["epochs"], 2)

    def test_quiet_round_does_nothing(self):
        budget = choose_budget(dataset_key="A", task="classification", round_index=4,
                               psi_prev=0.01, psi_holdout=0.01, data_grew=False,
                               stage_changed=False, gap=0.0,
                               history={"stalled": 0, "last_gain": 0.0})
        self.assertEqual(budget["epochs"], 0)

    def test_model_cannot_resurrect_a_noop(self):
        rule = {"action": "none", "epochs": 0, "lr": 0.0, "rule": "stalled",
                "reason": "x", "signals": {"psi_prev": 0.01}}
        out = apply_verdict(rule, {"action": "train", "epochs": 2, "lr": 1e-3,
                                   "reason": "try"}, {}, round_index=5)
        self.assertEqual(out["epochs"], 0)
        self.assertEqual(out["model_action"], "train_ignored_rule_none")

    def test_model_hold_ignored_on_drift(self):
        rule = {"action": "finetune", "epochs": 3, "lr": 1.5e-3, "rule": "new_stage_drift",
                "reason": "x", "signals": {"stage_changed": True, "psi_prev": 0.6}}
        out = apply_verdict(rule, {"action": "hold", "epochs": 0, "lr": 0.0,
                                   "reason": "wait"}, {}, round_index=5)
        self.assertEqual(out["epochs"], 3)

    def test_model_can_only_reduce(self):
        rule = {"action": "finetune", "epochs": 3, "lr": 1.5e-3, "rule": "new_stage",
                "reason": "x", "signals": {}}
        narrowed = apply_verdict(rule, {"action": "train", "epochs": 1, "lr": 1e-3,
                                        "reason": "cheap"}, {}, round_index=4)
        self.assertEqual(narrowed["epochs"], 1)
        held = apply_verdict(rule, {"action": "hold", "epochs": 0, "lr": 0.0,
                                    "reason": "stable"}, {}, round_index=4)
        self.assertEqual(held["epochs"], 0)

    def test_model_hold_ignored_on_first_round(self):
        rule = {"action": "finetune", "epochs": 5, "lr": 2e-3, "rule": "first_round",
                "reason": "x", "signals": {}}
        out = apply_verdict(rule, {"action": "hold", "epochs": 0, "lr": 0.0,
                                   "reason": "wait"}, {}, round_index=1)
        self.assertEqual(out["epochs"], 5)


class AcceptanceTests(unittest.TestCase):
    def test_classification_gain(self):
        out = acceptance({"macro_f1": 0.70}, {"macro_f1": 0.75}, "classification")
        self.assertTrue(out["promote"])
        self.assertAlmostEqual(out["gain"], 0.05)

    def test_classification_noise_not_promoted(self):
        out = acceptance({"macro_f1": 0.70}, {"macro_f1": 0.701}, "classification")
        self.assertTrue(out["accepted"])
        self.assertFalse(out["promote"])

    def test_regression_relative(self):
        out = acceptance({"rmse": 100.0}, {"rmse": 90.0}, "regression")
        self.assertTrue(out["promote"])
        self.assertAlmostEqual(out["gain"], 0.10)
        worse = acceptance({"rmse": 100.0}, {"rmse": 100.1}, "regression")
        self.assertFalse(worse["accepted"])      # "not worse" is strict
        self.assertFalse(worse["promote"])

    def test_remaining_life_has_no_upper_cap(self):
        self.assertEqual(clip_regression_prediction(300.0, 24), 300.0)
        self.assertEqual(clip_regression_prediction(120.0, 24), 120.0)
        self.assertEqual(clip_regression_prediction(300.0, 34), 300.0)
        self.assertEqual(clip_regression_prediction(-5.0, 34), 0.0)

    def test_metric_and_gap(self):
        name, value, higher = metric_value("regression", {"rmse": 50.0})
        self.assertEqual((name, value, higher), ("rmse", 50.0, False))
        self.assertAlmostEqual(score_gap(50.0, 55.0, False), 0.10)


class CrossRoundFuseTests(unittest.TestCase):
    def test_three_failures_latch_across_rounds(self):
        kill = KillSwitch({"policy": {"max_consecutive_failures": 3}})
        for _ in range(3):
            kill.note_step_error("boom")
        self.assertTrue(kill.tripped())
        saved = kill.as_dict()

        restored = KillSwitch({"policy": {"max_consecutive_failures": 3}})
        restored.consecutive_failures = int(saved["consecutive_failures"])
        restored.total_failures = int(saved["total_failures"])
        restored.stopped = True
        restored.reason = saved["reason"] or "latched in a prior round"

        gate = Gatekeeper(cfg={"policy": {"max_retrains": 10,
                                          "max_consecutive_failures": 3}},
                          kill=restored, executable=("none", "finetune"))
        decision = gate.review(
            GateDecision(True, "try again", Action("finetune", {"epochs": 1, "lr": 1e-3})),
            step=4, retrains=3)
        self.assertFalse(decision.approved)
        self.assertEqual(decision.action.kind, "none")
        self.assertIn("kill-switch", decision.reason)


if __name__ == "__main__":
    unittest.main()
