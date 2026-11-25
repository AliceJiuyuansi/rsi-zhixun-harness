"""Numerical and deployment boundaries of the v8 C calibration."""
import json
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vendor.contest_baselines.c_calibration import _best_shift, fit, predict  # noqa: E402


class CalibrationV8Tests(unittest.TestCase):
    def test_weighted_ridge_and_per_condition_sample_threshold(self):
        # Constant sensors isolate the intercept; all conditions share one spline.
        x = np.zeros((191, 24))
        x[:, 21] = np.repeat([0, 1, 2], [64, 64, 63])
        parent = np.full(len(x), 100.0)
        target = np.repeat([-6.0, 8.0, 8.0], [64, 64, 63])
        state = fit(x, parent + target, parent)
        weights = np.where(target < 0, 1.35, 1.0)
        expected = np.average(target, weights=weights) / 1.001
        self.assertAlmostEqual(state["coef"][0], expected, places=12)
        np.testing.assert_array_equal(state["coef"][1:], 0.0)
        self.assertEqual(state["shifts"], {"0": -3.5, "1": 0.5, "2": 0.0})
        loaded = json.loads(json.dumps(state, allow_nan=False))
        np.testing.assert_allclose(predict(x, parent, loaded), parent + expected +
                                   np.repeat([-3.5, 0.5, 0.0], [64, 64, 63]))

    def test_gate_the_risk_minimizer_not_the_best_feasible_point(self):
        # One outlier plus 63 identical errors. The fixtures straddle the gates.
        cases = [
            (28, -0.50, 0.25),  # RMSE +0.14057%, NASA -4.64%
            (20, -0.55, 0.00),  # optimum +0.50: RMSE +0.16095%; +0.25 is feasible
            (36, -2.25, 0.00),  # optimum +0.50: NASA improvement only 1.99948%
            (36, -2.70, 0.50),  # NASA improves 2.09280%
            (6, -0.45, 0.50),   # a 0.04 NASA weight would choose +0.25 instead
        ]
        p = np.full(64, 100.0)
        for outlier, common, expected in cases:
            with self.subTest(outlier=outlier, common=common):
                error = np.r_[outlier, np.full(63, common)]
                self.assertEqual(_best_shift(p, np.zeros(64), p - error), expected)

    def test_fit_and_serve_share_saturation_and_floor(self):
        p = np.full(64, 100.0)
        # Pre-clipping the raw spline (20 -> 10) would learn a negative shift,
        # even though every deployed grid point still saturates at +10.
        self.assertEqual(_best_shift(p, np.full(64, 20.0), p + 5), 0.0)
        # Flooring the baseline and then shifting it would learn a positive shift.
        self.assertEqual(_best_shift(np.ones(64), np.full(64, -5.0), np.ones(64)), 0.0)
        x = np.zeros((4, 24))
        x[:, 21] = [0, 1, 99, 99]
        parent = np.array([100.0, -15.0, 6.0, -6.0])
        moments = [[0.0] * 21, [1.0] * 21]
        state = {"moments": {"0": moments, "1": moments}, "knots": [],
                 "coef": [12.0] + [0.0] * 21, "shifts": {"0": -3.5, "99": -3.5}}
        np.testing.assert_array_equal(predict(x, parent, state), [108.5, 0, 6, 0])
        del state["shifts"]  # legacy checkpoint
        np.testing.assert_array_equal(predict(x, parent, state), [110, 0, 6, 0])

    def test_nasa_objective_denominator_has_unit_floor(self):
        # Only two unsaturated rows move. Their baseline NASA sum is 0.0588:
        # dividing by this tiny sum instead of 1 changes the optimum to +0.25.
        parent = np.full(64, 100.0)
        correction = np.r_[0.0, 0.0, np.full(62, 20.0)]
        y = np.r_[100.5, 100.2525, np.full(62, 110.0)]
        self.assertEqual(_best_shift(parent, correction, y), 0.5)

    def test_zero_and_extreme_nasa_baselines(self):
        zeros = np.zeros(64)
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            self.assertEqual(_best_shift(zeros, zeros, zeros), 0.0)
            # exp(1000) overflows a float, but its ratio still selects the endpoint.
            self.assertEqual(_best_shift(np.full(64, 10000.0), zeros, zeros), -3.5)

    def test_nonfinite_training_data_never_produces_checkpoint_state(self):
        for value in [np.nan, np.inf]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                x = np.zeros((64, 24))
                x[0, 21] = value
                fit(x, np.ones(64), np.ones(64))
        # Even all-finite observations can overflow their per-condition variance.
        x = np.zeros((64, 24))
        x[:32, 0] = 1e200
        with np.errstate(over="ignore"), self.assertRaises(ValueError):
            fit(x, np.ones(64), np.ones(64))


if __name__ == "__main__":
    unittest.main()
