"""Selection and checkpoint invariants for optional round-end weight smoothing."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from src.models.official_round import OfficialRound
from vendor.contest_baselines.c_calibration import fit
from vendor.contest_baselines.model import PARAMETERS, Model, checkpoint_digest


class SmoothingTests(unittest.TestCase):
    def test_selects_refitted_c_blend_from_best_not_last_epoch(self):
        rng = np.random.default_rng(42)
        train = rng.normal(size=(120, 24))
        val = rng.normal(size=(40, 24))
        train[:, 21] = val[:, 21] = 0  # one known C operating condition
        y = 70 + 8 * train[:, 0] - 2 * train[:, 1]
        parent = Model("regression", [f"f{i}" for i in range(24)], [], train, y,
                       hidden=6, batch_size=32, learning_rate=0.02)

        # Give the held-out rows targets predicted by the independently built,
        # train-only calibrated epoch-one blend. Its RMSE should be zero.
        expected = copy.deepcopy(parent)
        expected.fit(train, y, 1)
        for key in PARAMETERS:
            expected.params[key] = 0.85 * expected.params[key] + 0.15 * parent.params[key]
        expected.meta["c_calibration"] = fit(train, y, expected.predict(train))
        y_val = expected.predict(val)
        groups = ["g1", "g2"] * 20

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "parent.npz"
            parent.save(path)
            parent_sha = checkpoint_digest(path)
            trainer = OfficialRound()
            result = trainer.continue_round(path, train, y, val, y_val, out_dir=root / "round",
                                            epochs=3, validation_groups=groups,
                                            smooth_weights=True, calibrate_c=True)
            saved = Model.load(result.checkpoint)
            for key in PARAMETERS:
                np.testing.assert_array_equal(saved.params[key], expected.params[key])
                np.testing.assert_array_equal(saved.first[key], expected.first[key])
                np.testing.assert_array_equal(saved.second[key], expected.second[key])
            np.testing.assert_allclose(saved.meta["c_calibration"]["coef"],
                                       expected.meta["c_calibration"]["coef"])
            self.assertEqual(result.metrics_after, trainer.evaluate(saved, val, y_val, groups=groups))
            self.assertLess(result.metrics_after["rmse"], 1e-12)
            self.assertLess(result.metrics_after["rmse"], min(h["validation"]["rmse"] for h in result.history))
            self.assertEqual([h["epoch"] for h in result.history], [1, 2, 3])
            self.assertEqual((result.epochs_after, result.steps_after), (1, 4))
            self.assertEqual((result.config["epochs_run"], result.config["executed_steps"]), (3, 12))
            report = json.loads((result.out_dir / "training.json").read_text(encoding="utf-8"))
            self.assertEqual(report["validation_after"], result.metrics_after)
            self.assertEqual(report["checkpoint_sha256"], checkpoint_digest(result.checkpoint))
            self.assertEqual(checkpoint_digest(path), parent_sha)
            self.assertEqual(list(result.out_dir.glob(".snapshot*")), [])

    def test_no_strict_gain_keeps_original_checkpoint_bytes(self):
        x = np.arange(16.0).reshape(8, 2)
        y = np.arange(8.0)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "parent.npz"
            Model("regression", ["x", "y"], [], x, y, hidden=4).save(path)
            trainer = OfficialRound()
            # Final epoch (RMSE 3) is worse than the best epoch (RMSE 2).
            scores = [{"rmse": n} for n in (4, 2, 3)]
            with patch.object(trainer, "evaluate", side_effect=scores):
                raw = trainer.continue_round(path, x, y, x, y, out_dir=root / "raw", epochs=2)
            with patch.object(trainer, "evaluate", side_effect=[*scores, ValueError("invalid blend")]):
                fallback = trainer.continue_round(path, x, y, x, y, out_dir=root / "fallback",
                                                  epochs=2, smooth_weights=True)
                self.assertEqual(fallback.config["weight_smoothing"], "invalid_candidate")
                self.assertEqual(fallback.checkpoint.read_bytes(), raw.checkpoint.read_bytes())
            for label, smooth_rmse in [("tie", 2), ("within_tol", 2 - 5e-13), ("worse", 2.5)]:
                with self.subTest(label=label), patch.object(
                        trainer, "evaluate", side_effect=[*scores, {"rmse": smooth_rmse}]) as evaluate:
                    result = trainer.continue_round(path, x, y, x, y, out_dir=root / label,
                                                    epochs=2, smooth_weights=True)
                    self.assertEqual(evaluate.call_count, 4)
                    self.assertEqual(result.metrics_after, {"rmse": 2})
                    self.assertEqual(result.checkpoint.read_bytes(), raw.checkpoint.read_bytes())


if __name__ == "__main__":
    unittest.main()
