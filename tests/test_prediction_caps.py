"""The validation and serving paths call Model.predict for both official schemas."""
import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch
import numpy as np

from vendor.contest_baselines.model import Model


class PredictionCapTests(unittest.TestCase):
    def test_caps_use_the_public_input_schema(self):
        cases = [(24, '[["key","sensors"],["index",0]]', 5000),
                 (34, '[["key","features"],["key","horizontal"],["key","band1_energy_0_2k"]]', 5000),
                 (24, "unrelated", 5000)]
        for n, first, expected in cases:
            with self.subTest(schema=first):
                x = np.zeros((2,n))
                paths = [first]+[str(i) for i in range(1,n)]
                model = Model("regression", paths, [], x, np.full(2,5000.0), hidden=4)
                np.testing.assert_allclose(model.predict(x), expected)
                np.testing.assert_allclose(model.predict(x[:1]), expected)
                for release in ["0.6", "0.7", "0.7.1", "unexpected"]:
                    with patch.dict("os.environ", {"RSI_RELEASE": release}):
                        np.testing.assert_allclose(model.predict(x), 5000)
                        np.testing.assert_allclose(model.predict(x[:1]), 5000)
                with tempfile.TemporaryDirectory() as directory:
                    checkpoint = Path(directory) / "model.npz"
                    model.save(checkpoint)
                    with patch.dict("os.environ", {"RSI_RELEASE": "0.6"}):
                        np.testing.assert_allclose(Model.load(checkpoint).predict(x), 5000)
                model.meta["target_mean"] = 80
                np.testing.assert_allclose(model.predict(x), 80)
