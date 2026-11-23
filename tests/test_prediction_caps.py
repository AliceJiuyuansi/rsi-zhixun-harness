"""The validation and serving paths call Model.predict for both official schemas."""
import unittest
import numpy as np

from vendor.contest_baselines.model import Model


class PredictionCapTests(unittest.TestCase):
    def test_caps_use_the_public_input_schema(self):
        cases = [(24, '[["key","sensors"],["index",0]]', 260),
                 (34, '[["key","features"],["key","horizontal"],["key","band1_energy_0_2k"]]', 1200),
                 (24, "unrelated", 5000)]
        for n, first, expected in cases:
            with self.subTest(schema=first):
                x = np.zeros((2,n))
                paths = [first]+[str(i) for i in range(1,n)]
                model = Model("regression", paths, [], x, np.full(2,5000.0), hidden=4)
                np.testing.assert_allclose(model.predict(x), expected)
                np.testing.assert_allclose(model.predict(x[:1]), expected)
                model.meta["target_mean"] = 80
                np.testing.assert_allclose(model.predict(x), 80)
