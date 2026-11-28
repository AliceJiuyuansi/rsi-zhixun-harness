"""Closed-form class-balanced update of logits on a frozen gas classifier."""
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from src.models.official_round import OfficialRound, RoundResult, Model, checkpoint_digest


def prepare(train, previous, rnd, run_id, classes):
    rows = {}
    for identity, x, label in zip(train['ids'], train['x'], train['y'], strict=True):
        rows[identity] = hashlib.sha256(np.asarray(x, dtype='<f8').tobytes() + int(label).to_bytes(4, 'little')).hexdigest()
    def digest(mapping):
        return hashlib.sha256(json.dumps(mapping, sort_keys=True).encode()).hexdigest()
    inventory = dict(rows=rows, run=run_id, sha=digest(rows))
    prior = previous.get('a_fit_inventory', {})
    past = prior.get('rows', {})
    if rnd == 1:
        added, status = set(rows), 'initial'
    elif not (run_id and prior.get('run') == run_id and previous.get('dataset') == 'A' and previous.get('round') == rnd-1 and past and prior.get('sha') == digest(past)):
        added, status = set(), 'missing_history'
    elif any(rows[k] != past[k] for k in rows.keys() & past.keys()):
        added, status = set(), 'id_conflict'
    else:
        added, status = rows.keys() - past.keys(), 'verified'
    mask = np.array([k in added for k in train['ids']]); counts = np.bincount(train['y'][mask], minlength=len(classes))
    supported = np.flatnonzero(counts >= 3)
    return inventory, mask, dict(status=status, added=int(mask.sum()), counts=dict(zip(classes, counts.tolist())), supported=supported.tolist(), eligible=len(supported)>=2)


def fit_round(parent, train, panel, mask, out, stage, audit):
    started = time.process_time(); m = Model.load(parent); evaluator = OfficialRound(); before = evaluator.evaluate(m, panel['x'], panel['y']); cols=np.array(audit['supported'])
    keep=mask & np.isin(train['y'],cols); y=train['y'][keep]; h,z=m.forward(m.normalized(train['x'][keep])); design=np.column_stack((h,np.ones(len(h))))
    score=z[:,cols]; target=(y[:,None]==cols[None,:]) * np.log(.95*(len(cols)-1)/.05); diff=target-score; diff-=diff.mean(axis=1, keepdims=True)
    weights=1/(len(cols)*np.bincount(y)[y]); gram=design.T@(weights[:,None]*design)
    correction=np.linalg.solve(gram+.1*np.eye(len(gram)), design.T@(weights[:,None]*diff))
    coefficient=np.asarray(m.meta.get('a_online_residual_v1',np.zeros((len(gram),z.shape[1])))).copy();coefficient[:,cols]+=correction;m.meta['a_online_residual_v1']=coefficient.tolist()
    out=Path(out);out.mkdir(parents=True,exist_ok=False); checkpoint=out/'checkpoint.npz';m.save(checkpoint);after=evaluator.evaluate(Model.load(checkpoint),panel['x'],panel['y']);elapsed=time.process_time()-started
    conf=dict(epochs_run=0,executed_steps=0,closed_form_solves=1,weight_smoothing='not_attempted',component_cpu_seconds={'fit_and_validation':elapsed});ep=m.meta['epochs'];result=RoundResult(out,checkpoint,checkpoint_digest(checkpoint),checkpoint_digest(parent),before,after,seconds=elapsed,epochs_before=ep,epochs_after=ep,steps_after=m.meta['steps'],config=conf)
    report=dict(parent_checkpoint_sha256=result.parent_sha,checkpoint_sha256=result.checkpoint_sha,stage_id=stage,train_rows=len(train['y']),validation_rows=len(panel['y']),validation_before=before,validation_after=after,**conf,audit=audit)
    (out/'training.json').write_text(json.dumps(report,allow_nan=False,indent=2)+'\n');return result
