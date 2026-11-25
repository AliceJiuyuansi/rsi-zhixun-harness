"""Training-only weights preserve legacy checkpoint and optimizer behavior."""
import unittest

import numpy as np

from src.models.official_round import balanced_sample_weights
from vendor.contest_baselines.model import Model


class TrainingWeightsTests(unittest.TestCase):
    def test_label_smoothing_uses_uniform_mass_and_decay_excludes_bias(self):
        x = np.array([[0.0], [1.0], [3.0]])
        y = np.array([0, 1, 1])
        a, b = [Model('classification', ['x'], ['a', 'b'], x, y) for _ in range(2)]
        p = a.predict(x)
        expected = -.9 * np.log(p[np.arange(len(y)), y]).mean() - .1 * np.log(p).mean()
        before = b.params['w1'].copy()
        loss = a.fit(x, y, 1, label_smoothing=.1)
        b.fit(x, y, 1, label_smoothing=.1, weight_decay=.001)
        self.assertAlmostEqual(loss[0], expected)
        np.testing.assert_allclose(b.first['w1'] - a.first['w1'], .0001 * before, atol=1e-14)
        np.testing.assert_array_equal(b.first['b1'], a.first['b1'])

    def test_group_mass_is_equal_at_alpha_one(self):
        weights = balanced_sample_weights(['a', 'b', 'b', 'b', None], 1.0)
        self.assertAlmostEqual(weights.mean(), 1.0)
        self.assertAlmostEqual(weights[0], weights[1:4].sum())
        self.assertIsNone(balanced_sample_weights(['a', 'a', None], 1.0))
        self.assertIsNone(balanced_sample_weights(['a', 'b'], 0.0))

    def test_weighted_gradient_matches_integer_replication(self):
        x = np.array([[0.0], [1.0], [3.0]])
        y = np.array([2.0, 5.0, 7.0])
        counts = np.array([1, 2, 3])
        legacy, weighted, uniform = [Model('regression', ['x'], [], x, y) for _ in range(3)]
        repeated = Model('regression', ['x'], [], x, y)
        legacy.fit(x, y, 1)
        uniform.fit(x, y, 1, sample_weight=np.ones(3))
        losses = weighted.fit(x, y, 1, sample_weight=counts)
        expected = repeated.fit(np.repeat(x, counts, axis=0), np.repeat(y, counts), 1)
        np.testing.assert_allclose(losses, expected, rtol=1e-14)
        for name in legacy.params:
            np.testing.assert_array_equal(uniform.params[name], legacy.params[name])
            np.testing.assert_allclose(weighted.first[name], repeated.first[name], atol=1e-14)

    def test_invalid_weights_leave_checkpoint_state_unmodified(self):
        x = np.array([[0.0], [1.0]])
        y = np.array([2.0, 5.0])
        model = Model('regression', ['x'], [], x, y)
        for weights in ([1], [1, np.nan], [1, -1]):
            with self.subTest(weights=weights), self.assertRaisesRegex(ValueError, 'sample weights'):
                model.fit(x, y, 1, sample_weight=weights)
        self.assertEqual(model.meta['steps'], 0)
