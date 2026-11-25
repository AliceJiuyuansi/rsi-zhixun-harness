"""Paired candidate decisions on the same validation rows."""
import unittest

from src.harness.round_controller import acceptance


def bearings(rmse, second):
    return {"rmse": rmse, "by_group": {"major": {"rmse": rmse}, "minor": {"rmse": second}}}


class V8PromotionTests(unittest.TestCase):
    def test_bearing_gain_can_rescue_small_pooled_loss(self):
        before, after = bearings(100, 10), bearings(100.02, 8)
        result = acceptance(before, after, "regression", "B")
        self.assertTrue(result["promote"])
        self.assertAlmostEqual(result["gain"], -0.0002)
        self.assertFalse(acceptance(before, after, "regression", "C")["promote"])
        self.assertFalse(acceptance(before, bearings(100.06, 8), "regression", "B")["promote"])

    def test_one_valid_bearing_is_not_enough(self):
        before, after = bearings(100, 10), bearings(100.02, 8)
        del after["by_group"]["major"]["rmse"]
        self.assertFalse(acceptance(before, after, "regression", "B")["promote"])

    def test_c_nasa_veto_and_bonus(self):
        before = {"rmse": 100, "nasa_log1p": 10}
        for rmse in (99.7, 98):
            after = {"rmse": rmse, "nasa_log1p": 10.8}
            self.assertFalse(acceptance(before, after, "regression", "C")["accepted"])
            self.assertTrue(acceptance(before, after, "regression", "B")["promote"])
        after = {"rmse": 99.9, "nasa_log1p": 9.8}
        self.assertTrue(acceptance(before, after, "regression", "C")["promote"])
        self.assertFalse(acceptance(before, {"rmse": 99.9}, "regression", "C")["promote"])
