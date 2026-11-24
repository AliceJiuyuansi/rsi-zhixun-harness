"""Audit two C checkpoints on one public validation file, without training or platform calls.

Example:
    python scripts/audit_c_tail.py --parent parent.npz --child child.npz --validation public/validation.jsonl > c-tail.json

Deltas are child minus parent (negative is better). All NASA shares use the full
validation NASA sum, including unknown engines; shares are 0 if this sum is 0.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.models.official_round import nasa_metrics, regression_metrics
from vendor.contest_baselines.data import json_value, read_dataset, sha256
from vendor.contest_baselines.model import Model, checkpoint_digest


def _engine(meta):
    if not isinstance(meta, dict):
        return None
    subset, unit = meta.get("source_subset"), meta.get("unit")
    if not isinstance(subset, str) or not subset.strip():
        return None
    if type(unit) not in (int, str) or not str(unit).strip():
        return None
    return subset.strip(), str(unit).strip()


def _engine_groups(path, data):
    # read_dataset returns IDs but not C metadata. Bind the second read to its
    # validated bytes and join by sample ID, never by target or row number.
    raw = Path(path).read_bytes()
    if sha256(raw) != data["sha256"]:
        raise ValueError("validation file changed during audit")
    by_id = {}
    for line in raw.decode("utf-8").splitlines():
        if line.strip():
            row = json_value(line)
            by_id[row["id"]] = _engine(row.get("meta"))
    groups = defaultdict(list)
    for index, sample_id in enumerate(data["ids"]):
        groups[by_id[sample_id]].append(index)
    return groups


def _score(truth, pred, groups):
    nasa = nasa_metrics(truth, pred)
    error = pred - truth
    # Evaluate the same RMSE/MAE on scaled residuals: even squaring 1e300 or
    # summing a large batch of finite errors must not overflow the audit JSON.
    scale = float(np.max(np.abs(error))) or 1.0
    metrics = regression_metrics(np.zeros_like(error), error / scale)
    metrics = {k: v * scale for k, v in metrics.items()}
    metrics["nasa_log1p"] = nasa["nasa_log1p"]

    # expm1(t)/exp(peak), without constructing the unrepresentable raw sum.
    # Keeping row weights on a COMMON scale also preserves ties at huge peaks.
    t = np.where(error > 0, error / 10.0, -error / 13.0)
    weights = np.exp(t - t.max()) * -np.expm1(-t)
    total = float(weights.sum())
    if total:
        weights /= total

    def part(indices):
        return {"rows": len(indices), "nasa_share": float(weights[indices].sum())}

    def directions(indices):
        err = error[indices]
        return {name: part(indices[mask]) for name, mask in
                (("over", err > 0), ("under", err < 0), ("exact", err == 0))}

    engines = []
    for key in sorted(k for k in groups if k is not None):
        indices = np.asarray(groups[key], dtype=int)
        # Shares can underflow to zero for several non-tail engines. Rank by
        # their own log scores, not their rounded shares of the global total.
        engines.append({"source_subset": key[0], "unit": key[1], **part(indices),
                        "nasa_log1p": nasa_metrics(truth[indices], pred[indices])["nasa_log1p"],
                        "by_direction": directions(indices)})
    engines.sort(key=lambda g: (g["nasa_log1p"], g["nasa_share"]), reverse=True)
    top = engines[:5]
    missing = np.asarray(groups.get(None, []), dtype=int)
    return {**metrics, "by_direction": directions(np.arange(len(error))),
            "known_engine_count": len(engines), "top5_engines": top,
            "top5_nasa_share": sum(g["nasa_share"] for g in top),
            "unknown": {**part(missing), "by_direction": directions(missing)}}


def audit(parent, child, validation):
    models = [Model.load(path) for path in (parent, child)]
    paths = models[0].meta["paths"]
    if any(m.meta["task"] != "regression" or len(m.meta["paths"]) != 24 for m in models):
        raise ValueError("expected two C regression checkpoints (24 features)")
    if models[1].meta["paths"] != paths:
        raise ValueError("parent and child feature schemas differ")
    data = read_dataset(validation, "regression", expected_paths=paths)
    groups = _engine_groups(validation, data)
    report = {"dataset": "C", "validation_sha256": data["sha256"], "rows": len(data["y"])}
    for role, model, path in zip(("parent", "child"), models, (parent, child)):
        report[role] = {"checkpoint_sha256": checkpoint_digest(path),
                        **_score(data["y"], model.predict(data["x"]), groups)}
    report["delta_child_minus_parent"] = {k: report["child"][k] - report["parent"][k]
                                         for k in ("rmse", "mae", "nasa_log1p")}
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--child", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.parent, args.child, args.validation), indent=2, allow_nan=False))
