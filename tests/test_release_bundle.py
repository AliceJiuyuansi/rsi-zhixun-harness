"""Reproduce incomplete/legacy delivery failures without training."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from check_release import file_checks, checkpoint_check
from vendor.contest_baselines.model import Model
from vendor.contest_baselines.data import flatten


class ReleaseBundleTests(unittest.TestCase):
    def manifest(self):
        return json.loads((ROOT/'release_manifest.json').read_text(encoding='utf-8'))

    def test_real_release_imports_in_fresh_process(self):
        r=subprocess.run([sys.executable,str(ROOT/'scripts/check_release.py'),'--root',str(ROOT)],
                         capture_output=True,text=True,encoding='utf-8')
        self.assertEqual(r.returncode,0,r.stdout+r.stderr)
        self.assertTrue(json.loads(r.stdout)['ok'])

    def test_missing_budget_and_changed_calibrator_are_detected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            for name in self.manifest()['runtime_sha256']:
                dest=root/name;dest.parent.mkdir(parents=True,exist_ok=True)
                shutil.copyfile(ROOT/name,dest)
            (root/'src/harness/late_budget.py').unlink()
            (root/'vendor/contest_baselines/c_calibration.py').write_text('# stale module\n',encoding='utf-8')
            errors=file_checks(root,self.manifest(),'source')
            self.assertIn('missing_file:src/harness/late_budget.py',errors)
            self.assertIn('hash_mismatch:vendor/contest_baselines/c_calibration.py',errors)

    def test_image_copy_layout(self):
        from check_release import IMAGE_PATHS
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            for name in self.manifest()['runtime_sha256']:
                dest=root/IMAGE_PATHS.get(name,name);dest.parent.mkdir(parents=True,exist_ok=True)
                shutil.copyfile(ROOT/name,dest)
            self.assertEqual(file_checks(root,self.manifest(),'image'),[])

    def test_bad_manifest_paths_are_rejected(self):
        m=self.manifest();m['runtime_sha256']['../outside.py']='0'*64
        self.assertIn('invalid_manifest_path:../outside.py',file_checks(ROOT,m,'source'))

    def test_docker_preflight_inputs_are_in_build_context(self):
        lines=(ROOT/'.dockerignore').read_text(encoding='utf-8').splitlines()
        for rule in ('!scripts/','!scripts/check_release.py','!release_manifest.json','!_sdk/'):
            self.assertIn(rule,lines)
        docker=(ROOT/'harness.Dockerfile').read_text(encoding='utf-8')
        self.assertIn('COPY release_manifest.json /app/release_manifest.json',docker)
        self.assertIn('--layout image --require-sdk',docker)

    def test_unknown_calibration_cannot_hide_behind_model_fallback(self):
        with tempfile.TemporaryDirectory() as temporary:
            x=np.zeros((32,24));paths,_=flatten(x[0].tolist())
            model=Model('regression',paths,[],x,np.ones(32),hidden=4)
            model.meta['c_calibration']={'version':'unsupported-version'}
            p=Path(temporary)/'checkpoint.npz';model.save(p)
            with self.assertRaises(ValueError): checkpoint_check(p)

    def test_version16_uses_v15_policy_and_calibration(self):
        sys.path.insert(0,str(ROOT/'harness'))
        import contest_submission as submission
        self.assertTrue(submission.calibration_enabled('0.16'))
        self.assertEqual(submission.calibration_mode('0.16'),'res_mlp_v2')
        # Exercise all ten calls with the actual new profile, using the existing
        # disjoint synthetic train/val parent-state-finish test.
        from test_late_budget import LateBudgetTests
        case=LateBudgetTests('test_ten_round_synthetic_parent_state_finish_chain')
        case.release_profile='0.16'
        case.test_ten_round_synthetic_parent_state_finish_chain()


if __name__=='__main__':unittest.main(verbosity=2)
