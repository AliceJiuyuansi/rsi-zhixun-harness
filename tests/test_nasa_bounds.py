import unittest
import numpy as np

from src.eval.nasa_bounds import binary_nasa_bounds, compare_bounds
from src.models.official_round import nasa_metrics


class BinaryBoundsTests(unittest.TestCase):
    def test_exact_score_is_enclosed_for_both_error_directions(self):
        rng = np.random.default_rng(13)
        truth = rng.uniform(0, 450, 10000)
        pred = np.maximum(0, truth+rng.normal(0, 100, 10000))
        exact = nasa_metrics(truth, pred)["nasa_log1p"]
        for m in (1, 8, 32):
            result = binary_nasa_bounds(truth, pred, m)
            self.assertLessEqual(result["lower_log1p"], exact)
            self.assertGreaterEqual(result["upper_log1p"], exact)

    def test_zero_and_boundary(self):
        self.assertEqual(binary_nasa_bounds([0], [0])["upper_log1p"], 0.)
        for value in (np.log(2.)*10, 1e6):
            exact = nasa_metrics([0], [value])["nasa_log1p"]
            bounds = binary_nasa_bounds([0], [value])
            self.assertLessEqual(bounds["lower_log1p"], exact)
            self.assertGreaterEqual(bounds["upper_log1p"], exact)

    def test_ambiguous_comparison_requires_exact_score(self):
        same = binary_nasa_bounds([0, 0], [100, 120])
        self.assertEqual(compare_bounds(same, same), "need_exact")
        small = binary_nasa_bounds([0], [10])
        large = binary_nasa_bounds([0], [100])
        self.assertEqual(compare_bounds(large, small), "definitely_worse")


if __name__ == "__main__":
    unittest.main()
