"""Unit tests for v0.9.1 (spline_v2) and v0.9.2 (res_mlp_v1) C calibration upgrades."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from harness.contest_submission import calibration_enabled, calibration_mode  # noqa: E402
from src.models.official_round import OfficialRound  # noqa: E402
from vendor.contest_baselines.c_calibration import fit, predict  # noqa: E402
from vendor.contest_baselines.model import Model  # noqa: E402


def _synthetic(rows=360, seed=11):
    rng = np.random.default_rng(seed)
    conditions = rng.choice([0, 20, 42], size=rows)
    sensors = rng.normal(size=(rows, 21)) + conditions[:, None] * 0.05
    settings = np.zeros((rows, 3))
    settings[:, 0] = conditions
    x = np.column_stack([sensors, settings])
    parent = np.maximum(5.0, 90.0 + 18.0 * sensors[:, 0] + conditions * 0.5)
    y = np.maximum(0.0, parent + 15.0 * np.tanh(sensors[:, 1]) + (conditions == 42) * 25.0)
    return x, y, parent


class CalibrationV92Tests(unittest.TestCase):
    def test_release_and_override_mode_selection(self):
        self.assertTrue(calibration_enabled("0.9.1"))
        self.assertTrue(calibration_enabled("0.9.2"))
        self.assertEqual(calibration_mode("0.9"), "v8")
        self.assertEqual(calibration_mode("0.9.1"), "spline_v2")
        self.assertEqual(calibration_mode("0.9.2"), "res_mlp_v1")
        self.assertEqual(calibration_mode("0.9.2", "spline_v2"), "spline_v2")
        self.assertEqual(calibration_mode("0.9.1", "v8"), "v8")

    def test_spline_v2_and_res_mlp_v1_bounds_fallback_and_roundtrip(self):
        x, y, parent = _synthetic()
        for mode, neg_clip, pos_clip in (("spline_v2", 80.0, 40.0), ("res_mlp_v1", 120.0, 42.0)):
            with self.subTest(mode=mode):
                state = fit(x, y, parent, mode=mode)
                self.assertEqual(state["version"], mode)
                loaded = json.loads(json.dumps(state, allow_nan=False))
                pred = predict(x, parent, loaded)
                self.assertTrue(np.all(np.isfinite(pred)))
                self.assertTrue(np.all(pred >= 0.0))
                diff = pred - parent
                self.assertTrue(np.all(diff <= pos_clip + 1e-9))
                self.assertTrue(np.all(diff >= -neg_clip - 1e-9))

                # Unseen condition falls back to non-negative parent prediction
                unseen = x[:8].copy()
                unseen[:, 21] = 999
                fb = predict(unseen, parent[:8], loaded)
                np.testing.assert_allclose(fb, np.maximum(0.0, parent[:8]), rtol=0, atol=1e-12)

                # Batch vs single-row parity
                for i in range(10):
                    single = predict(x[i:i + 1], parent[i:i + 1], loaded)[0]
                    self.assertAlmostEqual(float(single), float(pred[i]), places=12)

    def test_continue_round_checkpoint_roundtrip_for_both_modes(self):
        x, y, _ = _synthetic(rows=200, seed=19)
        paths = [f"f{i}" for i in range(24)]
        base = Model("regression", paths, [], x, y, hidden=8, seed=5)
        for mode in ("spline_v2", "res_mlp_v1"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                (root / "parent").mkdir()
                base.save(root / "parent/checkpoint.npz")
                rr = OfficialRound().continue_round(
                    root / "parent/checkpoint.npz", x, y, x[:60], y[:60],
                    out_dir=root / "round", epochs=1, learning_rate=1e-3,
                    calibrate_c=True, calibration_mode=mode, smooth_weights=True, selection_task="C",
                )
                loaded = Model.load(str(rr.checkpoint))
                self.assertEqual(loaded.meta["c_calibration"]["version"], mode)
                self.assertEqual(rr.config["calibration_mode"], mode)
                batch = loaded.predict(x[:15])
                self.assertTrue(np.all(np.isfinite(batch)))
                for i in range(5):
                    self.assertAlmostEqual(float(loaded.predict(x[i:i + 1])[0]), float(batch[i]), places=12)

    def test_nonfinite_rejected_for_new_modes(self):
        x, y, parent = _synthetic(rows=100)
        x[0, 21] = np.nan
        for mode in ("spline_v2", "res_mlp_v1"):
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                fit(x, y, parent, mode=mode)


if __name__ == "__main__":
    unittest.main()
