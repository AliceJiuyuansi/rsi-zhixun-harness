"""Regression coverage for the C endpoint grid and its RMSE gate."""
import unittest
from unittest.mock import patch

import numpy as np

from vendor.contest_baselines import c_calibration as cal


class EndpointSearchTests(unittest.TestCase):
    def test_rmse_rejected_offsets_skip_nasa_evaluation(self):
        prediction = np.linspace(20.0, 100.0, 80)
        # The tiny baseline RMSE excludes every nonzero grid offset.
        y = prediction + 0.001
        with patch.object(cal.np, "expm1", wraps=np.expm1) as expm1:
            endpoint = cal._fit_endpoint(prediction, y)
        self.assertFalse(endpoint["enabled"])
        self.assertEqual(expm1.call_count, 1)

    def test_endpoint_search_matches_full_grid(self):
        rng = np.random.default_rng(735)
        for rows in (63, 64, 1000):
            for bias in (-18.0, 0.0, 24.0):
                with self.subTest(rows=rows, bias=bias):
                    prediction = np.maximum(0.0, rng.normal(60.0, 35.0, rows) + bias)
                    y = np.maximum(0.0, prediction - bias + rng.normal(0.0, 32.0, rows))
                    p20, p80 = np.quantile(prediction, (0.2, 0.8))
                    expected = dict(p20=p20, p80=p80, low=0.0, high=0.0, enabled=False)
                    rmse, log = cal._metrics(prediction - y)
                    if rows >= 64 and p80 > p20 and rmse > 0.0:
                        best = (log, 0.0, 0.0)
                        for low in cal.ENDPOINT_GRID:
                            for high in cal.ENDPOINT_GRID:
                                if low == high == 0:
                                    continue
                                offsets = cal._endpoint_offsets(prediction, dict(expected, low=low, high=high))
                                candidate_rmse, candidate_log = cal._metrics(np.maximum(0.0, prediction + offsets) - y)
                                if candidate_rmse <= rmse * (1.0 + cal.ENDPOINT_RMSE_TOLERANCE) and candidate_log < best[0] - 1e-12:
                                    best = (candidate_log, float(low), float(high))
                        if best[0] <= log + np.log(1.0 - cal.ENDPOINT_MIN_NASA_GAIN):
                            expected.update(low=best[1], high=best[2], enabled=True)
                    self.assertEqual(cal._fit_endpoint(prediction, y), expected)
