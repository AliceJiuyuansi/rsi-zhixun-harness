"""B calibration blend and old-checkpoint compatibility."""
import unittest

import numpy as np

from vendor.contest_baselines import b_calibration as cal


class AdaptiveBlendTests(unittest.TestCase):
    def test_downward_correction_uses_higher_calibration_weight(self):
        x = np.zeros((3, 2))
        y = np.array([99., 49., 199.])
        state = dict(version=cal.VERSION, indices=[0], mu=[0.], sd=[1.],
                     coef=[np.log(100.), 0.], alpha=.8, clip_hi=8.5)
        legacy = cal.predict(x, y, state)
        adjusted = cal.predict(x, y, dict(state, downward_alpha=.9))
        np.testing.assert_array_equal(adjusted[:2], legacy[:2])
        expected_high = cal.predict(x[2:], y[2:], dict(state, alpha=.9))
        np.testing.assert_array_equal(adjusted[2:], expected_high)
        self.assertLess(adjusted[2], legacy[2])

    def test_rejects_invalid_downward_weight(self):
        state = dict(version=cal.VERSION, indices=[0], mu=[0.], sd=[1.],
                     coef=[0., 0.], alpha=.8, clip_hi=8.5, downward_alpha=.7)
        with self.assertRaises(ValueError):
            cal.predict(np.zeros((1, 2)), [100.], state)
