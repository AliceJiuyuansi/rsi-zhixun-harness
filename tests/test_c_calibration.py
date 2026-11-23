"""Tests for the optional C operating-condition calibration."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.official_round import OfficialRound  # noqa: E402
from vendor.contest_baselines.c_calibration import fit, predict  # noqa: E402
from vendor.contest_baselines.model import Model  # noqa: E402


def synthetic(rows=240, seed=7):
    rng = np.random.default_rng(seed)
    conditions = rng.integers(0, 3, size=rows)
    sensors = rng.normal(size=(rows, 21)) + conditions[:, None] * 0.4
    settings = np.zeros((rows, 3))
    settings[:, 0] = conditions
    x = np.column_stack([sensors, settings])
    parent = 60 + 12 * sensors[:, 0] + conditions * 3
    y = parent + conditions * 2.5
    return x, y, parent


class CalibrationTests(unittest.TestCase):
    def test_fit_is_bounded_and_unseen_condition_falls_back(self):
        x, y, parent = synthetic()
        state = fit(x, y, parent)
        corrected = predict(x, parent, state)
        self.assertTrue(np.all(np.isfinite(corrected)))
        self.assertTrue(np.all(np.abs(corrected - parent) <= 10.0 + 1e-9))

        unseen = x[:5].copy()
        unseen[:, 21] = 99
        fallback = predict(unseen, parent[:5], state)
        np.testing.assert_allclose(fallback, np.maximum(0, parent[:5]), rtol=0, atol=1e-9)

    def test_checkpoint_roundtrip_keeps_calibration(self):
        x, y, parent = synthetic(rows=120)
        paths = [f"f{i}" for i in range(24)]
        model = Model("regression", paths, [], x, y, hidden=8, seed=3)
        state = fit(x, y, model.predict(x))
        model.meta["c_calibration"] = state
        expected = model.predict(x[:10])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.npz"
            model.save(path)
            loaded = Model.load(str(path))
            np.testing.assert_allclose(loaded.predict(x[:10]), expected, rtol=1e-12, atol=1e-12)

    def test_continue_round_can_fit_calibration(self):
        x, y, parent = synthetic(rows=160)
        paths = [f"f{i}" for i in range(24)]
        model = Model("regression", paths, [], x, y, hidden=8, seed=4)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "parent").mkdir()
            model.save(root / "parent/checkpoint.npz")
            result = OfficialRound().continue_round(
                root / "parent/checkpoint.npz", x, y, x[:40], y[:40],
                out_dir=root / "round", epochs=1, learning_rate=1e-3,
                calibrate_c=True)
            self.assertTrue(result.checkpoint.is_file())
            loaded = Model.load(str(result.checkpoint))
            self.assertIn("c_calibration", loaded.meta)
            self.assertTrue(np.all(np.isfinite(loaded.predict(x[:10]))))


if __name__ == "__main__":
    unittest.main()
