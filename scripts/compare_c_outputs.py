"""Score four C output policies on one calibrated checkpoint and public split."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vendor.contest_baselines.c_calibration import predict as apply_calibration
from vendor.contest_baselines.data import read_dataset
from vendor.contest_baselines.model import Model


def scores(predicted, truth, engine_keys, upper_bound_active):
    error = predicted - truth
    penalties = np.expm1(np.where(error > 0, error / 10, -error / 13))
    worst = int(np.argmax(penalties))
    return {
        "rows": len(error), "engines": len(set(engine_keys)),
        "affected_by_upper_bound": int(upper_bound_active.sum()),
        "mae": float(np.abs(error).mean()),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "nasa": float(penalties.sum()),
        "largest_one_row_nasa_share": float(penalties[worst] / penalties.sum()),
        "largest_one_row_signed_error_cycles": float(error[worst]),
    }


def compare(checkpoint, public_validation):
    model = Model.load(str(checkpoint))
    data = read_dataset(public_validation, "regression", expected_paths=model.meta["paths"])
    rows = [json.loads(line) for line in public_validation.read_text(encoding="utf-8").splitlines() if line.strip()]
    truth = np.array([row["answer"] for row in rows], dtype=float)
    assert np.array_equal(truth, data["y"])
    subsets = np.array([r["meta"]["source_subset"] for r in rows])
    engines = np.array([f"{r['meta']['source_subset']}:{r['meta']['unit']}" for r in rows])
    features = data["x"]
    _, logits = model.forward(model.normalized(features))
    base = np.maximum(0, logits[:, 0] * model.meta["target_scale"] + model.meta["target_mean"])
    calibrated = apply_calibration(features, base, model.meta["c_calibration"])
    groups = {"all": np.ones(len(truth), dtype=bool)}
    groups.update({str(name): subsets == name for name in sorted(set(subsets))})
    results = {}
    for name, raw, cap in (("base", base, None), ("cap260", base, 260),
                           ("cal10", calibrated, None), ("cal10_cap260", calibrated, 260)):
        prediction = raw if cap is None else np.minimum(raw, cap)
        results[name] = {label: scores(prediction[mask], truth[mask], engines[mask],
                                        (raw != prediction)[mask]) for label, mask in groups.items()}
    return {
        "scope": "public C validation; same checkpoint and rows for all four policies",
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "validation_sha256": hashlib.sha256(public_validation.read_bytes()).hexdigest(),
        "results": results,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("public_validation", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.write_text(json.dumps(compare(args.checkpoint, args.public_validation),
                                      ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
