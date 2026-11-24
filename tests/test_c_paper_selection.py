"""论文权重公式、NASA目标梯度、两层选模及历史检查点兼容性。"""
from pathlib import Path
import json
import shutil
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from src.harness.round_controller import acceptance
from src.harness.round_controller import summarise_history
from src.models.c_selection import nasa_noninferior
from src.models.official_round import OfficialRound, nasa_metrics
from vendor.contest_baselines.c_calibration import fit, predict, paper_weights, nasa_log_objective
from vendor.contest_baselines.model import Model, checkpoint_digest
from vendor.contest_baselines.data import flatten


class PaperWeightsTests(unittest.TestCase):
    def test_equation10_threshold_and_project_cap(self):
        error = np.array([0., 2., 9., 10., 20., 200.])
        expected = np.array([0., 1., 4.5, 10., 20., 100.])
        weights = paper_weights(error, np.zeros_like(error))
        np.testing.assert_allclose(weights, expected / expected.mean())
        self.assertAlmostEqual(weights.mean(), 1)

    def test_zero_error_fallback_and_bad_inputs(self):
        np.testing.assert_array_equal(paper_weights([1., 2.], [1., 2.]), [1., 1.])
        with self.assertRaises(ValueError):
            paper_weights([float("nan")], [0.])
        with self.assertRaises(ValueError):
            paper_weights([1.], [0.], weight_cap_cycles=5.)

    def test_weighted_state_is_bounded_and_roundtrips(self):
        rng = np.random.default_rng(2)
        x = rng.normal(size=(120, 24)); x[:, 21] = 0
        y = 60 + 20*x[:, 0]
        parent = np.full(120, 50.)
        state = fit(x, y, parent, method="paper_dw")
        self.assertEqual(state["method"], "paper_dw")
        self.assertTrue(np.all(np.abs(predict(x, parent, state)-parent) <= 10+1e-9))
        model = Model("regression", [str(i) for i in range(24)], [], x, y, hidden=4)
        model.meta["c_calibration"] = state
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory)/"checkpoint.npz"
            model.save(checkpoint)
            loaded = Model.load(checkpoint)
            np.testing.assert_array_equal(model.predict(x), loaded.predict(x))

    def test_legacy_calibration_without_new_metadata_still_predicts(self):
        x = np.zeros((32, 24)); y = np.ones(32)*50
        state = fit(x, y, y, method="ridge")
        for key in ("schema", "method", "fit", "optimization", "correction_limit_cycles"):
            state.pop(key, None)
        np.testing.assert_allclose(predict(x, y, state), y)


class NASALogTests(unittest.TestCase):
    def test_analytic_gradient_matches_finite_difference(self):
        design = np.array([[1., .2], [1., -.5], [1., .7]])
        coef = np.array([.1, -.2]); truth = np.array([10., 30., 12.]); parent = np.array([20., 15., 25.])
        _, gradient = nasa_log_objective(coef, design, truth, parent, 10., .001)
        numerical = []
        for i in range(2):
            delta = np.zeros(2); delta[i] = 1e-6
            plus = nasa_log_objective(coef+delta, design, truth, parent, 10., .001)[0]
            minus = nasa_log_objective(coef-delta, design, truth, parent, 10., .001)[0]
            numerical.append((plus-minus)/2e-6)
        np.testing.assert_allclose(gradient, numerical, rtol=1e-5, atol=1e-7)

    def test_same_exponent_mean_does_not_imply_same_nasa(self):
        balanced = nasa_metrics([0., 0.], [10., 10.])["nasa"]
        concentrated = nasa_metrics([0., 0.], [0., 20.])["nasa"]
        self.assertGreater(concentrated, balanced)

    def test_nasa_optimizer_reduces_its_training_objective(self):
        x = np.zeros((32, 24)); y = np.ones(32)*80; parent = np.ones(32)*50
        state = fit(x, y, parent, method="nasa_log", optimization_steps=10)
        self.assertLess(state["optimization"]["final"], state["optimization"]["initial"])
        self.assertTrue(np.all(predict(x, parent, state) <= parent+10+1e-9))


class CSelectionTests(unittest.TestCase):
    def test_real_round_persists_reference_checkpoint_for_next_round(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"harness"))
        import contest_submission as submission
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            x = np.zeros((32, 24)); paths, _ = flatten([0.]*24)
            selected = root/"parent.npz"
            Model("regression", paths, [], x, np.ones(32)*50, hidden=4).save(selected)
            data = root/"public.jsonl"
            data.write_text("".join(json.dumps({"id": str(i), "inputs": [0.]*24, "answer": 0})+"\n" for i in range(32)), encoding="utf-8")
            state_dir = root/"empty-state"
            for round_index in (1, 2):
                finishes, candidates = [], {}
                def submit(path, **kwargs):
                    candidates["candidate"] = Path(path)/"checkpoint.npz"
                    return {"artifact_id": "candidate"}
                saved = root/f"saved-{round_index}"
                rnd = SimpleNamespace(task={"task": "regression"}, dataset_key="C", round_index=round_index,
                    stage_id="fixed", max_output_tokens=256, history=lambda: [], history_ids=[], call_ids=["local"],
                    model=lambda *a, **kw: SimpleNamespace(response={}),
                    parent=SimpleNamespace(artifact_id="parent", declared_paths=lambda: ["checkpoint.npz"], path=lambda *a, **kw: selected),
                    data=SimpleNamespace(train_path=data, validation_path=data), submit_candidate=submit,
                    save_state=lambda path: shutil.copytree(path, saved), finish=lambda **kw: finishes.append(kw))
                with patch.object(submission, "WORKSPACE", root/"workspace"), patch.object(submission, "STATE_DIR", state_dir):
                    submission.train(rnd)
                state = json.loads((saved/"harness_state.json").read_text(encoding="utf-8"))
                self.assertEqual(checkpoint_digest(saved/"c_rmse_anchor.npz"), state["c_reference"]["checkpoint_sha256"])
                self.assertLessEqual(state["validation_selected"]["rmse"], state["c_reference"]["reference_used_rmse"]*1.01+1e-9)
                if finishes[0]["selected_artifact_id"] == "candidate":
                    selected = candidates["candidate"]
                state_dir = saved

    def test_large_nasa_gain_can_spend_at_most_one_percent_rmse(self):
        result = acceptance({"rmse": 100., "nasa": 1000.}, {"rmse": 100.5, "nasa": 500.},
                            "regression", "C", c_reference_rmse=100.)
        self.assertTrue(result["promote"])
        self.assertGreater(result["policy_gain"], 0)
        history = summarise_history([{"round_index": 1, "decision": "promote", "validation_metrics": {"rmse": 100.5},
                                    "configuration": {"gain": result["gain"], "policy_gain": result["policy_gain"]}}], {}, "regression")
        self.assertGreater(history["last_gain"], 0)
        self.assertFalse(acceptance({"rmse": 100.5, "nasa": 500.}, {"rmse": 101.2, "nasa": 200.},
                                   "regression", "C", c_reference_rmse=100.)["promote"])
        self.assertFalse(acceptance({"rmse": 100., "nasa": 1000.}, {"rmse": 100.5, "nasa": 950.},
                                   "regression", "C", c_reference_rmse=100.)["promote"])

    def test_rmse_gain_cannot_override_worse_nasa(self):
        result = acceptance({"rmse": 100., "nasa": 1000.}, {"rmse": 90., "nasa": 1001.}, "regression", "C")
        self.assertFalse(result["promote"])
        self.assertEqual(result["nasa_gate"]["reason"], "reject_nasa")
        self.assertTrue(acceptance({"rmse": 100.}, {"rmse": 90.}, "regression", "B")["promote"])

    def test_saturated_scores_use_log_score(self):
        maximum = float(np.finfo(float).max)
        self.assertFalse(nasa_noninferior({"nasa": maximum, "nasa_log1p": 710.},
                                         {"nasa": maximum, "nasa_log1p": 711.})["passed"])

    def test_gate_uses_total_not_maximum(self):
        before = {"rmse": 100., "nasa": 20., "nasa_max_row": 5.}
        after = {"rmse": 90., "nasa": 19., "nasa_max_row": 6.}
        self.assertTrue(acceptance(before, after, "regression", "C")["promote"])
        self.assertFalse(nasa_noninferior({"nasa": 10.}, {})["passed"])

    def run_round(self, values):
        x = np.zeros((8, 24)); y = np.ones(8)
        directory = tempfile.TemporaryDirectory(); self.addCleanup(directory.cleanup)
        root = Path(directory.name); parent = root/"parent.npz"
        Model("regression", [str(i) for i in range(24)], [], x, y, hidden=4).save(parent)
        trainer = OfficialRound()
        with patch.object(trainer, "evaluate", side_effect=values):
            result = trainer.continue_round(parent, x, y, x, y, out_dir=root/"round", epochs=len(values)-1)
        return parent, result

    def test_inner_snapshot_ignores_lowest_rmse_with_unsafe_nasa(self):
        _, result = self.run_round([{"rmse": 10., "nasa": 100.}, {"rmse": 9., "nasa": 200.},
                                    {"rmse": 9.5, "nasa": 90.}, {"rmse": 9.4, "nasa": 110.}])
        self.assertEqual(result.epochs_after, 2)
        self.assertEqual(result.metrics_after["nasa"], 90.)

    def test_all_unsafe_epochs_return_exact_parent(self):
        parent, result = self.run_round([{"rmse": 10., "nasa": 100.}, {"rmse": 9., "nasa": 200.},
                                         {"rmse": 8., "nasa": 300.}])
        self.assertEqual(checkpoint_digest(parent), result.checkpoint_sha)
        self.assertTrue(result.config["selected_parent"])
        self.assertEqual(result.config["epochs_run"], 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
