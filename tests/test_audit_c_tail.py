"""Local C checkpoint audit: same holdout, stable full-validation NASA shares."""
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.audit_c_tail import _engine_groups, _score, audit
from vendor.contest_baselines.data import flatten, read_dataset
from vendor.contest_baselines.model import Model


def digest(path):
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def constant_checkpoint(path, value, inputs):
    paths, values = flatten(inputs)
    model = Model("regression", paths, [], np.array([values]), np.array([value]), hidden=2)
    model.params["w2"].fill(0)
    model.save(path)


class TailAuditTests(unittest.TestCase):
    def test_cli_paired_metrics_and_engine_tail(self):
        inputs = [0.0] * 24
        truth = [30, 30, 25, 76, 40, 63, 50, 0, 80, 50]
        keys = [("A", 1), ("A", 1), ("B", 1), ("A", 2), ("A", 3), ("A", 4), ("A", 5)]
        meta = [{"source_subset": s, "unit": u} for s, u in keys] + [{"unit": 42}, {"source_subset": "A"}, None]
        with tempfile.TemporaryDirectory() as directory:
            p, c, v = (Path(directory) / name for name in ("p.npz", "c.npz", "validation.jsonl"))
            constant_checkpoint(p, 50, inputs)
            # Child has identical base weights: only the served C calibration
            # changes 50 to 40. Auditing base logits would miss the entire delta.
            child = Model.load(p)
            child.meta["c_calibration"] = {"moments": {"0": [[0] * 21, [1] * 21]},
                                           "knots": [], "coef": [-10] + [0] * 21}
            child.save(c)
            rows = [{"id": f"A_engine{i}", "inputs": inputs, "answer": y, "meta": m}
                    for i, (y, m) in enumerate(zip(truth, meta))]
            v.write_text("\n\n".join(map(json.dumps, rows)), encoding="utf-8")
            cmd = [sys.executable, str(ROOT / "scripts/audit_c_tail.py"), "--parent", str(p), "--child", str(c),
                   "--validation", str(v)]
            raw = subprocess.check_output(cmd, text=True)
            report = json.loads(raw, parse_constant=lambda token: self.fail(f"nonfinite JSON: {token}"))
            self.assertEqual(report["validation_sha256"], digest(v))
            for role, path, value in (("parent", p, 50), ("child", c, 40)):
                result = report[role]
                self.assertEqual(result["checkpoint_sha256"], digest(path))
                err = value - np.array(truth)
                costs = np.expm1(np.where(err > 0, err / 10, -err / 13))
                self.assertAlmostEqual(result["rmse"], np.sqrt(np.mean(err ** 2)))
                self.assertAlmostEqual(result["mae"], np.mean(abs(err)))
                self.assertAlmostEqual(result["nasa_log1p"], math.log1p(costs.sum()))
                self.assertEqual(result["known_engine_count"], 6)
                self.assertEqual(result["unknown"]["rows"], 3)
                self.assertAlmostEqual(result["unknown"]["nasa_share"], costs[-3:].sum() / costs.sum())
                for name, mask in (("over", err > 0), ("under", err < 0), ("exact", err == 0)):
                    self.assertEqual(result["by_direction"][name]["rows"], int(mask.sum()))
                    self.assertAlmostEqual(result["by_direction"][name]["nasa_share"], costs[mask].sum() / costs.sum())
                groups = {key: sum(cost for k, cost in zip(keys, costs) if k == key) for key in keys}
                expected = sorted(groups, key=lambda k: (-groups[k], k))[:5]
                top = result["top5_engines"]
                self.assertEqual([(g["source_subset"], int(g["unit"])) for g in top], expected)
                for group, key in zip(top, expected):
                    self.assertAlmostEqual(group["nasa_share"], groups[key] / costs.sum())
                self.assertAlmostEqual(result["top5_nasa_share"], sum(groups[k] for k in expected) / costs.sum())
            for k in ("rmse", "mae", "nasa_log1p"):
                self.assertAlmostEqual(report["delta_child_minus_parent"][k], report["child"][k] - report["parent"][k])

            data = read_dataset(v, "regression")
            data["ids"].reverse()
            self.assertEqual(_engine_groups(v, data)[("A", "1")], [8, 9])  # join by ID, not position

    def test_huge_tied_penalties_and_small_engine_ranking(self):
        truth = np.array([0, 0, 0, 13, 26.])
        pred = np.array([1e300, 1e300, 1e300, 0, 0])
        groups = {("A", "1"): [0, 1], None: [2], ("A", "2"): [3], ("A", "3"): [4]}
        report = _score(truth, pred, groups)
        json.dumps(report, allow_nan=False)
        self.assertAlmostEqual(report["rmse"] / 1e300, math.sqrt(3 / 5))
        self.assertAlmostEqual(report["nasa_log1p"] / 1e299, 1.)
        self.assertAlmostEqual(report["top5_nasa_share"], 2 / 3)
        self.assertAlmostEqual(report["unknown"]["nasa_share"], 1 / 3)
        self.assertEqual([g["unit"] for g in report["top5_engines"]], ["1", "3", "2"])
        with self.assertRaisesRegex(ValueError, "finite"):
            _score(np.array([0.]), np.array([np.inf]), {None: [0]})

    def test_zero_total_and_only_unknown(self):
        report = _score(np.array([1.]), np.array([1.]), {None: [0]})
        json.dumps(report, allow_nan=False)
        self.assertEqual(report["nasa_log1p"], 0)
        self.assertEqual(report["top5_engines"], [])
        self.assertEqual(report["top5_nasa_share"], 0)
        self.assertEqual(report["unknown"]["by_direction"]["exact"], {"rows": 1, "nasa_share": 0.})

    def test_same_width_schema_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            p, c, v = (Path(directory) / name for name in ("p.npz", "c.npz", "v.jsonl"))
            inputs = [0] * 24
            constant_checkpoint(p, 0, inputs)
            constant_checkpoint(c, 0, {"sensors": inputs})
            with self.assertRaisesRegex(ValueError, "schemas differ"):
                audit(p, c, v)
            row = {"id": "x", "inputs": {"sensors": inputs}, "answer": 0}
            v.write_text(json.dumps(row), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "feature schema"):
                audit(p, p, v)


if __name__ == "__main__":
    unittest.main()
