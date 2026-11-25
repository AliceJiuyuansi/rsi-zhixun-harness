"""Contract: same parent and same validation rows, no extra training epochs."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from src.harness.round_controller import acceptance, paired_acceptance
from src.models.official_round import OfficialRound
from vendor.contest_baselines.model import Model


def bearing(rmse, small):
    return {"rmse": rmse, "by_group": {"major": {"rmse": rmse}, "minor": {"rmse": small}}}


def engine(rmse, log_nasa):
    return {"rmse": rmse, "nasa_log1p": log_nasa}


class PairedSelectionTests(unittest.TestCase):
    def test_a_keeps_ties_not_regressions(self):
        parent = {"macro_f1": .72}
        self.assertTrue(acceptance(parent, {"macro_f1": .71}, "classification", "A")["promote"])
        for new, expect in [(.71, False), (.72, True), (.73, True)]:
            self.assertEqual(paired_acceptance(parent, {"macro_f1": new}, "classification", "A")["promote"], expect)

    def run_round(self, dataset, metrics, smooth=False):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            x = np.arange(24.).reshape(12, 2)
            y = np.arange(12.)
            parent = root / 'parent.npz'
            Model('regression', ['x', 'y'], [], x, y, hidden=4).save(parent)
            trainer = OfficialRound()
            with patch.object(trainer, 'evaluate', side_effect=metrics):
                return trainer.continue_round(parent, x, y, x, y, out_dir=root/'candidate',
                                              epochs=len(metrics)-1-int(smooth), smooth_weights=smooth,
                                              selection_task=dataset)

    def test_b_rescues_promotable_bearing_candidate_only(self):
        metrics = [bearing(100, 8), bearing(99.95, 8), bearing(100.01, 7)]
        result = self.run_round('B', metrics)
        self.assertEqual(result.metrics_after, metrics[2])
        self.assertEqual(result.config['epochs_run'], 2)
        self.assertEqual([r['eligible'] for r in result.history], [False, True])
        # When the original lowest pooled RMSE can already promote, preserve it.
        metrics[1] = bearing(98, 8)
        self.assertEqual(self.run_round('B', metrics).metrics_after, metrics[1])

    def test_c_risky_best_and_blend_cannot_displace_feasible_snapshot(self):
        metrics = [engine(100, 12), engine(98, 12.1), engine(96, 13), engine(95, 13)]
        result = self.run_round('C', metrics, smooth=True)
        self.assertEqual(result.metrics_after, metrics[1])
        self.assertEqual(result.epochs_after, 1)
        self.assertEqual(result.config['weight_smoothing'], 'kept_original')
        self.assertEqual(result.config['epochs_run'], 2)
        self.assertTrue(all(t >= 0 for t in result.config['component_cpu_seconds'].values()))

    def test_c_without_eligible_snapshot_does_not_promote(self):
        metrics = [engine(100, 12), engine(98, 13), {"rmse": 97}]
        result = self.run_round('C', metrics)
        self.assertFalse(paired_acceptance(metrics[0], result.metrics_after, 'regression', 'C')['promote'])
        self.assertFalse(any(r['eligible'] for r in result.history))
