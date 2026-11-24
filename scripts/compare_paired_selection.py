r"""Offline, single-round v8 vs paired selection; not a platform run or a trajectory.

Example: python -I scripts/compare_paired_selection.py --dataset B --package /data/B \
    --epochs 3 --lr .002 --holdout /data/later/B/public/validation.jsonl --output /tmp/B-run
Omit --holdout for selection diagnostics only. Supply only explicitly public labelled data.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vendor.contest_baselines.model import Model, checkpoint_digest
from vendor.contest_baselines.data import read_dataset
from src.harness import round_controller
from src.models.official_round import OfficialRound


def _feature_keys(data):
    # Parsed float64 rows in the checkpoint's feature order, independent of ID,
    # JSON formatting and label. Normalize signed zero so renaming cannot hide a copy.
    return (hashlib.sha256(row + 0.0).digest() for row in data['x'])


def compare(package, dataset, output, *, epochs=3, lr=None, holdout=None):
    parent = package / 'checkpoint.npz'
    meta = Model.load(parent).meta
    parent_sha = checkpoint_digest(parent)
    task = meta['task']
    if (task != ('classification' if dataset == 'A' else 'regression') or
            (task == 'regression' and (dataset == 'C') != (len(meta['paths']) == 24))):
        raise ValueError('dataset key does not match checkpoint task/features')

    def load(path):
        return read_dataset(path, task, meta['classes'], expected_paths=meta['paths'])

    train = load(package / 'public/train.jsonl')
    val = load(package / 'public/validation.jsonl')
    splits = {'train': train, 'selection_validation': val}
    if holdout is not None:
        test = load(holdout)
        ids, features = set(test['ids']), set(_feature_keys(test))
        for name, split in splits.items():
            if not ids.isdisjoint(split['ids']) or not features.isdisjoint(_feature_keys(split)):
                raise ValueError(f'holdout overlaps {name} by ID or feature row')
        splits['holdout'] = test

    runner = OfficialRound()
    selected_paths = {}
    arms = {}
    gates = [('v8', '', round_controller.acceptance),
             ('paired', dataset, round_controller.paired_acceptance)]
    output.mkdir(parents=True, exist_ok=False)
    for arm, key, gate in gates:
        # The same parent seed/epoch and original train order give identical
        # epoch permutations. No early stopping: actual SGD budgets match too.
        cpu, wall = time.process_time(), time.perf_counter()
        r = runner.continue_round(parent, train['x'], train['y'], val['x'], val['y'],
                                  out_dir=output / arm, epochs=epochs, learning_rate=lr,
                                  patience=0, smooth_weights=True, calibrate_c=dataset == 'C',
                                  validation_groups=val['groups'] if dataset == 'B' else None,
                                  selection_task=key)
        outcome = gate(r.metrics_before, r.metrics_after, task, dataset)
        child = outcome['promote']
        selected_paths[arm] = r.checkpoint if child else parent
        arms[arm] = {'cpu_seconds': time.process_time() - cpu, 'elapsed_seconds': time.perf_counter() - wall,
                     'selection_task': key, 'promotion': outcome, 'selected': 'child' if child else 'parent',
                     'child_sha256': r.checkpoint_sha, 'selected_sha256': r.checkpoint_sha if child else parent_sha,
                     'budget': r.config, 'child_epoch': r.epochs_after, 'child_step': r.steps_after,
                     'validation_parent': r.metrics_before, 'validation_child': r.metrics_after,
                     'selected_validation': r.metrics_after if child else r.metrics_before}
        if r.parent_sha != parent_sha or r.config['epochs_run'] != epochs:
            raise RuntimeError('arms must start from the same parent and execute the full budget')
    if arms['v8']['budget']['executed_steps'] != arms['paired']['budget']['executed_steps']:
        raise RuntimeError('arms executed different training budgets')

    # Holdout metrics never enter training, smoothing or either promotion gate.
    if holdout is not None:
        for arm, path in selected_paths.items():
            arms[arm]['selected_holdout'] = runner.evaluate(Model.load(path), test['x'], test['y'],
                                                             groups=test['groups'] if dataset == 'B' else None)
    report = {'scope': 'offline single round; input provenance not verified; not a platform score',
              'evaluation': ('user-supplied disjoint holdout' if holdout is not None else
                             'selection validation only; no independent holdout'),
              'dataset': dataset, 'parent_sha256': parent_sha, 'parent_seed': meta['seed'],
              'parent_epoch': meta['epochs'], 'parent_step': meta['steps'],
              'data': {k: {'sha256': v['sha256'], 'rows': len(v['ids'])} for k, v in splits.items()},
              'timing_scope': 'continue_round plus promotion per arm; excludes holdout scoring and data loading',
              'arms': arms}
    (output / 'result.json').write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + '\n',
                                        encoding='utf-8')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--package', type=Path, required=True)
    parser.add_argument('--dataset', choices=('A', 'B', 'C'), required=True)
    parser.add_argument('--output', type=Path, required=True, help='new directory for both children and result.json')
    parser.add_argument('--holdout', type=Path, help='independent labelled later-public-stage JSONL')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--lr', type=float, help='learning rate; defaults to parent checkpoint')
    args = parser.parse_args()
    try:
        report = compare(**vars(args))
    except (OSError, ValueError) as exc:
        parser.exit(2, f'{parser.prog}: {exc}\n')
    print(f"{report['evaluation']}: {args.output / 'result.json'}")


if __name__ == '__main__':
    main()
