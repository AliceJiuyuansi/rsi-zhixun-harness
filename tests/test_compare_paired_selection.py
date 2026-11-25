"""Synthetic wiring fixture; no public scores, host, SDK or external packages besides NumPy."""
import json
from pathlib import Path
import tempfile
import unittest

from scripts.compare_paired_selection import compare
from vendor.contest_baselines.data import read_dataset
from vendor.contest_baselines.model import Model, checkpoint_digest


class ComparePairedSelection(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        public = self.root / 'public'
        public.mkdir()
        self.out = self.root / 'result'
        self.holdout = self.root / 'later.jsonl'
        self.train = public / 'train.jsonl'
        self.val = public / 'validation.jsonl'

        def write(path, xs, training=False):
            rows = [{'id': f'{path.stem}-{i}', 'inputs': {'x': x},
                     'answer': 'left' if training or x < 0 else 'right'} for i, x in enumerate(xs)]
            path.write_text(''.join(json.dumps(r) + '\n' for r in rows), encoding='utf-8')

        # One training step moves the decision boundary just right of zero.
        # Exactly one positive validation point flips; v8 tolerates it, paired doesn't.
        write(self.train, [-.7, -.3, .3, .7], training=True)
        write(self.val, [-1 - i / 100 for i in range(50)] + [.001] + [1 + i / 100 for i in range(49)])
        write(self.holdout, [-.8, .002, .8])
        classes = ['left', 'right']
        train = read_dataset(self.train, 'classification', classes)
        m = Model('classification', train['paths'], classes, train['x'], train['y'], hidden=1)
        m.mean[:] = 0
        m.scale[:] = 1
        m.params['w1'][:] = 1
        m.params['w2'][:] = [-1, 1]
        m.save(self.root / 'checkpoint.npz')

    def run_pair(self, holdout=None):
        return compare(self.root, 'A', self.out, epochs=1, lr=.01, holdout=holdout)

    def test_real_training_and_independent_scoring(self):
        r = self.run_pair(self.holdout)
        self.assertEqual(r, json.loads((self.out / 'result.json').read_text()))
        v8, paired = (r['arms'][k] for k in ('v8', 'paired'))
        self.assertEqual((v8['selected'], paired['selected']), ('child', 'parent'))
        self.assertEqual(v8['selected_sha256'], checkpoint_digest(self.out / 'v8/checkpoint.npz'))
        self.assertEqual(paired['selected_sha256'], checkpoint_digest(self.root / 'checkpoint.npz'))
        self.assertEqual(v8['selected_holdout']['accuracy'], 2 / 3)
        self.assertEqual(paired['selected_holdout']['accuracy'], 1)
        for arm in (v8, paired):
            self.assertEqual(arm['budget']['executed_steps'], 1)
            self.assertGreater(arm['elapsed_seconds'], 0)
            self.assertGreater(arm['cpu_seconds'], 0)

    def test_no_holdout_does_not_reuse_validation(self):
        report = self.run_pair()
        self.assertIn('no independent holdout', report['evaluation'])
        self.assertNotIn('holdout', report['data'])
        for arm in report['arms'].values():
            self.assertNotIn('selected_holdout', arm)

    def test_holdout_rejects_shared_id_or_renamed_feature_row(self):
        for split in (self.train, self.val):
            orig = json.loads(split.read_text().splitlines()[0])
            for row in ({**orig, 'id': 'renamed'}, {**orig, 'inputs': {'x': 99}}):
                with self.subTest(split=split, row=row):
                    self.holdout.write_text(json.dumps(row))
                    with self.assertRaisesRegex(ValueError, 'holdout overlaps'):
                        self.run_pair(self.holdout)
                    self.assertFalse(self.out.exists())
