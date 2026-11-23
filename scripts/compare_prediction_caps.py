"""Score the same uncalibrated baseline weights at different upper limits."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vendor.contest_baselines.data import read_dataset
from vendor.contest_baselines.model import Model


def compare(dataset, package, checkpoint):
    model = Model.load(str(checkpoint))
    sample = read_dataset(package/"public/validation.jsonl", "regression", expected_paths=model.meta["paths"])
    _, logits = model.forward(model.normalized(sample["x"]))
    base = np.maximum(0, logits[:, 0] * model.meta["target_scale"] + model.meta["target_mean"])
    limits = [1000,1100,1200,1300,1400,1500,1600,1800,2000,2200,2400,2537] if dataset == "B" else [145,200,220,240,260,275,300,320,340,375,420,542]
    results = []
    for limit in [None, *limits]:
        pred = base if limit is None else np.minimum(base, limit)
        error = pred - sample["y"]
        row = {"limit": limit, "affected": int(np.count_nonzero(pred != base)),
               "mae": np.abs(error).mean(), "rmse": np.sqrt(np.mean(error**2))}
        if dataset == "C":
            row["nasa"] = np.expm1(np.where(error > 0, error/10, -error/13)).sum()
        results.append(row)
    return {"rows":len(base), "max_prediction":float(base.max()), "sensitivity":results}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=["B","C"])
    parser.add_argument("package", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.write_text(json.dumps(compare(args.dataset, args.package, args.checkpoint), indent=2)+"\n", encoding="utf-8")
