"""真实续训、模型失败降级、回执解析与早停快照记账。"""
from pathlib import Path
import json
import math
import shutil
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "harness")]
import contest_submission as submission
from src.harness.model_receipt import decide_with_receipt
from src.harness.round_controller import extract_text, row_observation
from src.models.official_round import OfficialRound, nasa_metrics, grouped_regression_metrics
from vendor.contest_baselines.data import flatten, read_dataset
from vendor.contest_baselines.model import Model


class ReceiptTests(unittest.TestCase):
    def budget(self):
        return {"action": "finetune", "epochs": 3, "lr": .0015, "rule": "new_stage", "signals": {}, "reason": "test"}

    def decide(self, response=None, error=None):
        def model(*args, **kwargs):
            if error:
                raise error
            return SimpleNamespace(response=response)
        rnd = SimpleNamespace(model=model, max_output_tokens=256)
        return decide_with_receipt(rnd, "test", self.budget(), {}, 4)

    def test_exception_degrades_and_does_not_log_error_text(self):
        decision, _, metadata = self.decide(error=TimeoutError("secret-value"))
        self.assertEqual(decision["epochs"], 3)
        self.assertEqual(metadata["error_type"], "TimeoutError")
        self.assertNotIn("secret-value", json.dumps(metadata))
        self.assertEqual(decision["source"], "rule")

    def test_empty_length_receipt_is_honest(self):
        decision, _, metadata = self.decide({"choices": [{"message": {"content": ""}, "finish_reason": "length"}]})
        self.assertEqual(metadata["status"], "empty")
        self.assertEqual(metadata["finish_reason"], "length")
        self.assertEqual(decision["model_action"], "unavailable")

    def test_effective_hold_is_llm(self):
        decision, _, metadata = self.decide({"output_text": '{"decision":"hold"}'})
        self.assertEqual((decision["epochs"], decision["source"]), (0, "llm"))
        self.assertTrue(metadata["changed_rule"])

    def test_agreement_is_not_claimed_as_llm_change(self):
        decision, _, metadata = self.decide({"output_text": '{"decision":"train","epochs":3,"lr":0.0015}'})
        self.assertEqual(decision["source"], "rule")
        self.assertEqual(metadata["status"], "valid_json")

    def test_bad_shapes_and_non_string_parts(self):
        for response in [None, [], {"choices": [1]}, {"choices": [{"message": [1]}]}]:
            self.assertEqual(extract_text(response), "")
        self.assertEqual(extract_text({"choices": [{"message": {"content": [{"text": {}}, {"text": "ok"}]}}]}), "ok")


class MetricsTests(unittest.TestCase):
    def test_calibration_override_can_disable_default(self):
        self.assertTrue(submission.calibration_enabled("0.7.1"))
        self.assertFalse(submission.calibration_enabled("0.7.1", "0"))
        self.assertFalse(submission.calibration_enabled("0.7", "false"))
        self.assertTrue(submission.calibration_enabled("0.6", "1"))

    def test_nasa_formula_and_extreme(self):
        self.assertAlmostEqual(nasa_metrics([20, 20], [7, 30])["nasa"], 2 * math.expm1(1))
        result = nasa_metrics([0], [1e9])
        self.assertTrue(result["nasa_saturated"])
        self.assertTrue(math.isfinite(result["nasa"]))
        self.assertTrue(math.isfinite(result["nasa_log1p"]))

    def test_groups_and_missing_coverage(self):
        result = grouped_regression_metrics([0, 10, 20], [1, 12, 23], ["b1", "b2", None])
        self.assertEqual(result["by_group"]["b1"]["rmse"], 1)
        self.assertEqual(result["by_group"]["b2"]["rows"], 1)
        self.assertEqual(result["ungrouped_rows"], 1)

    def test_row_schema_migration_and_bounded_history(self):
        state = row_observation({"n_train": 20}, 20, 5, 1)
        self.assertFalse(state["data_grew"])
        for i in range(2, 13):
            state = row_observation(state, 20+i, 5+i, i)
        self.assertTrue(state["data_grew"])
        self.assertEqual(len(state["rows_history"]), 10)

    def test_metadata_group_never_enters_features(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.jsonl"
            path.write_text(json.dumps({"id": "xjtu_Bearing1_4_s00001", "inputs": [1, 2], "answer": 3,
                                        "meta": {"bearing": "b1"}}), encoding="utf-8")
            data = read_dataset(path, "regression")
            self.assertEqual(data["groups"], ["b1"])
            np.testing.assert_array_equal(data["x"], [[1, 2]])


class ContinuationTests(unittest.TestCase):
    def test_saved_best_snapshot_counters_are_distinct_from_executed(self):
        x = np.zeros((8, 2)); y = np.ones(8)
        paths, _ = flatten([0., 0.])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            parent = root / "parent.npz"
            Model("regression", paths, [], x, y, hidden=4).save(parent)
            trainer = OfficialRound()
            with patch.object(trainer, "evaluate", side_effect=[{"rmse": v} for v in [10, 9, 8, 9, 10]]):
                result = trainer.continue_round(parent, x, y, x, y, out_dir=root / "round", epochs=4)
            selected = Model.load(result.checkpoint)
            self.assertEqual(result.epochs_after, selected.meta["epochs"])
            self.assertEqual(result.epochs_after, 2)
            self.assertEqual(result.steps_after, selected.meta["steps"])
            self.assertEqual(result.config["epochs_run"], 4)
            self.assertEqual(result.config["executed_final_epochs"], 4)

    def test_model_timeout_still_attempts_finish_and_saves_state(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths, _ = flatten([0., 0.])
            checkpoint = root / "parent.npz"
            Model("regression", paths, [], np.zeros((32, 2)), np.ones(32), hidden=4).save(checkpoint)
            data = root / "data.jsonl"
            data.write_text("".join(json.dumps({"id": str(i), "inputs": [0., 0.], "answer": 1})+"\n" for i in range(32)), encoding="utf-8")
            finishes = []
            def model(*args, **kwargs):
                raise TimeoutError("sensitive detail")
            rnd = SimpleNamespace(task={"task": "regression"}, dataset_key="B", round_index=1, stage_id="test",
                                  max_output_tokens=256, history=lambda: [], model=model, history_ids=[], call_ids=[],
                                  parent=SimpleNamespace(artifact_id="parent", declared_paths=lambda: ["checkpoint.npz"],
                                                         path=lambda *args, **kwargs: checkpoint),
                                  data=SimpleNamespace(train_path=data, validation_path=data),
                                  save_state=lambda path: shutil.copytree(path, root/"saved"),
                                  finish=lambda **kwargs: finishes.append(kwargs),
                                  submit_candidate=lambda *args, **kwargs: {"artifact_id": "candidate"})
            with patch.object(submission, "WORKSPACE", root/"workspace"), patch.object(submission, "STATE_DIR", root/"state"):
                submission.train(rnd)
            self.assertEqual(len(finishes), 1)
            self.assertEqual(finishes[0]["model_call_ids"], [])  # 不伪造成功调用
            state = json.loads((root/"saved/harness_state.json").read_text(encoding="utf-8"))
            self.assertEqual(state["model_receipt"]["status"], "call_error")
            self.assertEqual(state["rows"], {"train": 32, "validation": 32})
            self.assertNotIn("sensitive detail", json.dumps(state))


if __name__ == "__main__":
    unittest.main(verbosity=2)
