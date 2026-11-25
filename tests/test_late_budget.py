"""Budget replay and synthetic ten-round wiring (not platform accuracy)."""
import json
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'harness')]
import contest_submission as submission
from src.harness.late_budget import plan_c_budget, protect_c_attempt, pace_c_budget, execution_record
from src.harness.round_controller import choose_budget
from vendor.contest_baselines.model import Model


class LateBudgetTests(unittest.TestCase):
    def budget(self, **extra):
        return {'action': 'finetune', 'epochs': 3, 'lr': .0015,
                'rule': 'calibration', 'signals': {}, 'reason': 'test', **extra}

    def test_same_stage_cheap_but_first_and_new_stage_unchanged(self):
        self.assertEqual(plan_c_budget(self.budget(), idle_rounds=0, kill_switch=False)['epochs'], 1)
        for rule in ('first_round', 'c_stage_probe'):
            self.assertEqual(plan_c_budget(self.budget(rule=rule), idle_rounds=0, kill_switch=False)['epochs'], 3)

    def test_idle_probe_and_kill(self):
        b = self.budget(action='none', epochs=0, rule='stalled')
        self.assertEqual(plan_c_budget(b, idle_rounds=2, kill_switch=False)['epochs'], 2)
        self.assertEqual(plan_c_budget(b, idle_rounds=2, kill_switch=True)['epochs'], 0)

    def test_hold_protected_but_rollback_respected(self):
        b = self.budget(signals={'data_grew': True})
        d = self.budget(action='none', epochs=0, model_action='hold')
        self.assertEqual(protect_c_attempt(d, b)['override'], 'c_required_probe')
        d['model_action'] = 'rollback'
        self.assertEqual(protect_c_attempt(d, b)['epochs'], 0)

    def test_reserve_released_and_cannot_overspend(self):
        b = self.budget()
        self.assertEqual(pace_c_budget(b, spent=10, round_index=7)['epochs'], 0)
        self.assertEqual(pace_c_budget(b, spent=10, round_index=8)['epochs'], 3)
        self.assertEqual(pace_c_budget(b, spent=13, round_index=10)['epochs'], 0)
        for spent in range(18):
            for r in range(1, 11):
                d = pace_c_budget(b, spent=spent, round_index=r)
                self.assertLessEqual(d['epochs'], max(0, 13-spent))

    def test_historical_signal_replay_leaves_r9_budget(self):
        spent=idle=stalled=0; trace=[]
        for r in range(1,11):
            fresh = r in (1,4,7,9)
            b = choose_budget(dataset_key='C', task='regression', round_index=r,
                              psi_prev=0, psi_holdout=0, data_grew=fresh,
                              stage_changed=fresh, gap=0,
                              history={'stalled':stalled,'last_gain':.01 if r<=4 else -.01})
            b = plan_c_budget(b, idle_rounds=idle, kill_switch=False)
            d = pace_c_budget(b, spent=spent, round_index=r)
            n=d['epochs']; trace.append(n); spent+=n
            idle=0 if n else idle+1
            if r>=4 and n: stalled+=1
        self.assertEqual(trace, [3,1,1,3,0,0,2,0,3,0])
        self.assertEqual(spent,13)

    def test_partial_failure_does_not_claim_zero_executed(self):
        e=execution_record(self.budget(),None,False,'error')
        self.assertIsNone(e['epochs_executed'])
        self.assertEqual(e['status'],'training_error')

    def test_release_keeps_v14_calibrator(self):
        self.assertTrue(submission.calibration_enabled('0.15'))
        self.assertEqual(submission.calibration_mode('0.15'),'res_mlp_v2')

    def test_ten_round_synthetic_parent_state_finish_chain(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); rng=np.random.default_rng(12)
            x=rng.normal(size=(100,24)); y=30+x[:,0]
            parent=root/'parent.npz'
            # Flatten uses the model's declared feature paths; derive them exactly.
            from vendor.contest_baselines.data import flatten
            paths,_=flatten(x[0].tolist())
            Model('regression',paths,[],x[:64],y[:64],hidden=4).save(parent)
            state=root/'state'; work=root/'workspace'; history=[]; finishes=[]
            val=root/'val.jsonl'; train=root/'train.jsonl'
            def write(path, indexes):
                path.write_text(''.join(json.dumps({'id':str(i),'inputs':x[i].tolist(),
                    'answer':float(y[i])})+'\n' for i in indexes),encoding='utf-8')
            write(val,range(80,100))
            for r in range(1,11):
                write(train,range(64+min(r,8)))
                def save_state(path):
                    if state.exists(): shutil.rmtree(state)
                    shutil.copytree(path,state)
                def submit(*args, **kwargs):
                    shutil.copyfile(work/f'candidate-round-{r}'/'checkpoint.npz',parent)
                    return {'artifact_id':f'candidate-{r}'}
                rnd=SimpleNamespace(task={'task':'regression'},dataset_key='C',round_index=r,
                    stage_id=f'C-{1+(r-1)//3}',max_output_tokens=256,
                    history=lambda:history,model=lambda *a,**k:SimpleNamespace(response={}),
                    history_ids=[],call_ids=[],parent=SimpleNamespace(artifact_id='parent',
                    declared_paths=lambda:['checkpoint.npz'],path=lambda *a,**k:parent),
                    data=SimpleNamespace(train_path=train,validation_path=val),save_state=save_state,
                    finish=lambda **kw:finishes.append(kw),submit_candidate=submit)
                with patch.object(submission,'WORKSPACE',work),patch.object(submission,'STATE_DIR',state),patch.object(submission,'RELEASE','0.15'):
                    submission.train(rnd)
                c=finishes[-1]['configuration']; history.append({'round_index':r,'configuration':c})
                self.assertNotEqual(c['execution']['status'],'training_error')
                self.assertIn('selection_audit',c) if c['epochs'] else None
            saved=json.loads((state/'harness_state.json').read_text(encoding='utf-8'))
            self.assertEqual(len(finishes),10)
            self.assertLessEqual(saved['requested_epochs_total'],13)
            self.assertEqual(len(saved['rows_history']),10)
            self.assertEqual(sum(f['configuration']['epochs'] for f in finishes),saved['requested_epochs_total'])


if __name__=='__main__': unittest.main(verbosity=2)
