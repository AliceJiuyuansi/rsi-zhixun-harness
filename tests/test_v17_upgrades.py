"""Unit and integration tests for v17 upgrades (0.17)."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "harness")]
import contest_submission as submission
from src.harness.late_budget import (
    execution_record,
    pace_c_budget,
    plan_c_budget,
    protect_c_attempt,
)
from src.harness.round_controller import acceptance, choose_budget, paired_acceptance
from src.models.official_round import OfficialRound
from vendor.contest_baselines import b_calibration
from vendor.contest_baselines.data import flatten
from vendor.contest_baselines.model import Model


class BCalibrationTests(unittest.TestCase):
    def _synthetic_bearings(self):
        rng = np.random.default_rng(7)
        xs, ys, gs = [], [], []
        for name, n in (("Bearing1_1", 80), ("Bearing2_1", 100), ("Bearing3_1", 400)):
            rul = np.linspace(float(n - 1), 0.0, n)
            # First 6 features increase as RUL decreases (negative corr with RUL);
            # remaining 4 features are noise.
            deg = (1.0 - rul / max(n, 1))[:, None]
            signal = 0.5 + 3.0 * deg + rng.normal(0.0, 0.15, (n, 6))
            noise = rng.normal(0.0, 1.0, (n, 4))
            xs.append(np.hstack([np.maximum(0.01, signal), noise]))
            ys.append(rul)
            gs.extend([name] * n)
        return np.vstack(xs), np.concatenate(ys), gs

    def test_fit_predict_single_equals_batch_and_checkpoint_roundtrip(self):
        x, y, groups = self._synthetic_bearings()
        paths, _ = flatten(x[0].tolist())
        model = Model("regression", paths, [], x, y, hidden=8, seed=42)
        model.fit(x, y, epochs=2)
        base = model.predict(x)
        state = b_calibration.fit(x, y, groups)
        self.assertEqual(state["version"], "sign_log_v1")
        self.assertGreaterEqual(len(state["indices"]), 3)

        model.meta["b_calibration"] = state
        batch = model.predict(x[:16])
        singles = np.array([model.predict(x[i : i + 1])[0] for i in range(16)])
        self.assertLess(float(np.max(np.abs(batch - singles))), 1e-12)
        self.assertFalse(np.allclose(batch, base[:16]))

        with tempfile.TemporaryDirectory() as td:
            ckpt = Path(td) / "checkpoint.npz"
            model.save(ckpt)
            reloaded = Model.load(ckpt)
            np.testing.assert_allclose(reloaded.predict(x[:16]), batch, atol=1e-12)

    def test_corrupted_state_falls_back_to_base_mlp(self):
        x, y, _ = self._synthetic_bearings()
        paths, _ = flatten(x[0].tolist())
        model = Model("regression", paths, [], x, y, hidden=8, seed=42)
        base = model.predict(x[:10])
        model.meta["b_calibration"] = {"version": "broken", "coef": "bad"}
        fallback = model.predict(x[:10])
        np.testing.assert_allclose(fallback, base, atol=1e-12)

    def test_single_group_raises_and_official_round_skips_cleanly(self):
        x, y, _ = self._synthetic_bearings()
        with self.assertRaises(ValueError):
            b_calibration.fit(x[:40], y[:40], ["Bearing1_1"] * 40)

        paths, _ = flatten(x[0].tolist())
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            parent = root / "parent.npz"
            Model("regression", paths, [], x[:60], y[:60], hidden=4).save(parent)
            res = OfficialRound().continue_round(
                parent, x[:60], y[:60], x[60:80], y[60:80],
                out_dir=root / "r1", epochs=1, calibrate_b=True,
                train_groups=["Bearing1_1"] * 60,
            )
            self.assertFalse(res.config["calibrate_b"])
            loaded = Model.load(res.checkpoint)
            self.assertNotIn("b_calibration", loaded.meta)


class LateBudgetAndPolicyTests(unittest.TestCase):
    def test_reserve_and_limit_never_overspend(self):
        budget = {"action": "finetune", "epochs": 3, "lr": 0.0015, "rule": "c_stage_probe", "signals": {}}
        self.assertEqual(pace_c_budget(budget, spent=9, round_index=7, limit=12)["epochs"], 0)
        self.assertEqual(pace_c_budget(budget, spent=9, round_index=8, limit=12)["epochs"], 3)
        for spent in range(15):
            for r in range(1, 11):
                out = pace_c_budget(budget, spent=spent, round_index=r, limit=12)
                self.assertLessEqual(out["epochs"], max(0, 12 - spent))

    def test_skip_small_same_stage_chase_and_protect_stage_change(self):
        small_cal = {"action": "finetune", "epochs": 3, "lr": 0.001, "rule": "calibration",
                     "signals": {"last_gain": 0.001, "data_grew": False, "stage_changed": False}}
        planned = plan_c_budget(small_cal, idle_rounds=0, kill_switch=False, min_calib_gain=0.005)
        self.assertEqual(planned["epochs"], 0)
        self.assertEqual(planned["rule"], "c_skip_small_chase")

        stage_b = {"action": "finetune", "epochs": 3, "lr": 0.0015, "rule": "c_stage_probe",
                   "reason": "stage", "signals": {"stage_changed": True}}
        held = {**stage_b, "action": "none", "epochs": 0, "model_action": "hold"}
        self.assertEqual(protect_c_attempt(held, stage_b)["epochs"], 3)
        rolled = {**stage_b, "action": "none", "epochs": 0, "model_action": "rollback"}
        self.assertEqual(protect_c_attempt(rolled, stage_b)["epochs"], 0)

    def test_quiet_stall_a_and_c_nasa_tol(self):
        b_quiet = choose_budget(
            dataset_key="A", task="classification", round_index=3,
            psi_prev=0.01, psi_holdout=0.01, data_grew=False, stage_changed=False,
            gap=0.0, history={"stalled": 2, "last_gain": 0.0}, quiet_stall_a=True,
        )
        self.assertEqual((b_quiet["epochs"], b_quiet["rule"]), (0, "a_stalled"))
        b_stage = choose_budget(
            dataset_key="A", task="classification", round_index=4,
            psi_prev=0.2, psi_holdout=0.2, data_grew=False, stage_changed=True,
            gap=0.0, history={"stalled": 2, "last_gain": 0.0}, quiet_stall_a=True,
        )
        self.assertEqual(b_stage["epochs"], 3)

        before = {"rmse": 46.91, "nasa_log1p": 21.06}
        after_good_nasa = {"rmse": 46.94, "nasa_log1p": 20.88}
        after_bad_rmse = {"rmse": 47.20, "nasa_log1p": 20.88}
        self.assertFalse(paired_acceptance(before, after_good_nasa, "regression", "C")["promote"])
        self.assertTrue(
            paired_acceptance(before, after_good_nasa, "regression", "C", c_nasa_tol=0.0015)["promote"]
        )
        self.assertFalse(
            paired_acceptance(before, after_bad_rmse, "regression", "C", c_nasa_tol=0.0015)["promote"]
        )


class V17SubmissionIntegrationTests(unittest.TestCase):
    def test_ultra_compact_prompt_is_tiny_and_valid_json(self):
        payload = {
            "dataset": "C", "round": 4, "stage": "C-2", "parent": 46.86, "best": 46.86,
            "gap": 0.0, "rows": {"train": 100000, "validation": 20000, "stage_changed": True, "train_grew": True},
            "drift": {"psi_vs_previous": 0.312}, "stalled": 0, "last_gain": 0.11,
            "rule_budget": {"action": "finetune", "epochs": 3, "lr": 0.0015, "rule": "c_stage_probe"},
        }
        prompt = submission._model_prompt(payload, compact=True, ultra_compact=True)
        self.assertTrue(prompt.startswith("/no_think\n"))
        self.assertLess(len(prompt), 260)
        parsed = json.loads(prompt.splitlines()[-1])
        self.assertEqual((parsed["d"], parsed["r"], parsed["ep"], parsed["chg"]), ("C", 4, 3, 1))

    def test_v17_release_options_and_c_fast_path_synthetic_run(self):
        self.assertEqual(submission.calibration_mode("0.17"), "res_mlp_v2")
        self.assertTrue(submission.calibration_enabled("0.17"))
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            rng = np.random.default_rng(12)
            x = rng.normal(size=(120, 24))
            x[:, 21] = np.where(np.arange(120) % 2 == 0, 1.0, 2.0)
            y = np.maximum(0.0, 40.0 + 3.0 * x[:, 0] - 2.0 * x[:, 1])
            paths, _ = flatten(x[0].tolist())
            parent = root / "parent.npz"
            Model("regression", paths, [], x[:80], y[:80], hidden=4).save(parent)
            train_p, val_p = root / "train.jsonl", root / "val.jsonl"

            def write_jsonl(path, idxs):
                path.write_text(
                    "".join(
                        json.dumps({"id": f"fd001_u1_c{i}", "inputs": x[i].tolist(), "answer": float(y[i])}) + "\n"
                        for i in idxs
                    ),
                    encoding="utf-8",
                )

            write_jsonl(train_p, range(80))
            write_jsonl(val_p, range(80, 120))
            state = root / "state"
            work = root / "work"
            finishes = []
            model_mock = Mock(return_value=SimpleNamespace(response={"output_text": '{"decision":"train","epochs":3,"lr":0.002}'}))
            rnd = SimpleNamespace(
                task={"task": "regression"}, dataset_key="C", round_index=1, stage_id="C-1",
                max_output_tokens=256, history=lambda: [], model=model_mock,
                history_ids=[], call_ids=["call-1"],
                parent=SimpleNamespace(
                    artifact_id="parent", declared_paths=lambda: ["checkpoint.npz"],
                    path=lambda *a, **k: parent,
                ),
                data=SimpleNamespace(train_path=train_p, validation_path=val_p),
                save_state=lambda p: (shutil.rmtree(state, ignore_errors=True), shutil.copytree(p, state)),
                finish=lambda **kw: finishes.append(kw),
                submit_candidate=lambda *a, **k: {"artifact_id": "cand-1"},
            )
            with patch.object(submission, "WORKSPACE", work), patch.object(submission, "STATE_DIR", state), \
                 patch.object(submission, "RELEASE", "0.17"):
                submission.train(rnd)
            _, kwargs = model_mock.call_args
            self.assertEqual(kwargs.get("max_tokens"), 64)
            self.assertIs(kwargs.get("enable_thinking"), False)
            self.assertEqual(len(finishes), 1)
            cfg = finishes[0]["configuration"]
            self.assertEqual(cfg["release"], "0.17")
            self.assertIn("execution", cfg)
            self.assertLessEqual(cfg["epochs"], 3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
