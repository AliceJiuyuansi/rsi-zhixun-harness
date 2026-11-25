"""Verify v17 release integrity, Docker build gates, and calibration compatibility."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
from check_release import IMAGE_PATHS, checkpoint_check, file_checks  # noqa: E402
from vendor.contest_baselines.data import flatten  # noqa: E402
from vendor.contest_baselines.model import Model  # noqa: E402


class ReleaseBundleTests(unittest.TestCase):
    def manifest(self):
        return json.loads((ROOT / "release_manifest.json").read_text(encoding="utf-8"))

    def test_real_release_imports_in_fresh_process(self):
        proc = subprocess.run(
            [sys.executable, str(ROOT / "scripts/check_release.py"), "--root", str(ROOT)],
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        payload = json.loads(proc.stdout)
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["release"], "0.17")

    def test_missing_budget_and_changed_calibrators_are_detected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in self.manifest()["runtime_sha256"]:
                dest = root / name
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(ROOT / name, dest)
            (root / "src/harness/late_budget.py").unlink()
            (root / "vendor/contest_baselines/c_calibration.py").write_text("# stale\n", encoding="utf-8")
            (root / "vendor/contest_baselines/b_calibration.py").write_text("# stale\n", encoding="utf-8")
            errors = file_checks(root, self.manifest(), "source")
            self.assertIn("missing_file:src/harness/late_budget.py", errors)
            self.assertIn("hash_mismatch:vendor/contest_baselines/c_calibration.py", errors)
            self.assertIn("hash_mismatch:vendor/contest_baselines/b_calibration.py", errors)

    def test_image_copy_layout(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in self.manifest()["runtime_sha256"]:
                dest = root / IMAGE_PATHS.get(name, name)
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(ROOT / name, dest)
            self.assertEqual(file_checks(root, self.manifest(), "image"), [])

    def test_bad_manifest_paths_are_rejected(self):
        manifest = self.manifest()
        manifest["runtime_sha256"]["../outside.py"] = "0" * 64
        self.assertIn("invalid_manifest_path:../outside.py", file_checks(ROOT, manifest, "source"))

    def test_docker_preflight_inputs_are_in_build_context(self):
        lines = (ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()
        for rule in ("!scripts/", "!scripts/check_release.py", "!release_manifest.json", "!_sdk/"):
            self.assertIn(rule, lines)
        docker = (ROOT / "harness.Dockerfile").read_text(encoding="utf-8")
        self.assertIn("ARG RSI_RELEASE=0.17", docker)
        self.assertIn("COPY release_manifest.json /app/release_manifest.json", docker)
        self.assertIn("--layout image --require-sdk", docker)

    def test_unknown_c_and_b_calibration_cannot_hide_behind_fallback(self):
        with tempfile.TemporaryDirectory() as temporary:
            x = np.zeros((32, 24))
            paths, _ = flatten(x[0].tolist())

            model_c = Model("regression", paths, [], x, np.ones(32), hidden=4)
            model_c.meta["c_calibration"] = {"version": "unsupported-version"}
            path_c = Path(temporary) / "checkpoint_c.npz"
            model_c.save(path_c)
            with self.assertRaises(ValueError):
                checkpoint_check(path_c)

            model_b = Model("regression", paths, [], x, np.ones(32), hidden=4)
            model_b.meta["b_calibration"] = {"version": "unsupported-version"}
            path_b = Path(temporary) / "checkpoint_b.npz"
            model_b.save(path_b)
            with self.assertRaises(ValueError):
                checkpoint_check(path_b)

    def test_v17_release_flags_and_calibration(self):
        sys.path.insert(0, str(ROOT / "harness"))
        import contest_submission as submission

        self.assertEqual(submission.RELEASE, "0.17")
        self.assertTrue(submission.calibration_enabled("0.17"))
        self.assertEqual(submission.calibration_mode("0.17"), "res_mlp_v2")
        self.assertIn("0.17", submission.ULTRA_COMPACT_RELEASES)
        self.assertIn("0.17", submission.B_CALIBRATION_RELEASES)
        self.assertIn("0.17", submission.C_LATE_BUDGET_RELEASES)


if __name__ == "__main__":
    unittest.main(verbosity=2)
