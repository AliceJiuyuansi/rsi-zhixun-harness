"""Check the delivered code, interfaces and optional checkpoint, without training.

Uses only stdlib and NumPy. Missing SDK is reported separately unless required.
Source paths are mapped to the Docker COPY layout for image checks.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib
import importlib.util
import inspect
import json
from pathlib import Path, PurePosixPath
import sys


IMAGE_PATHS = {
    "harness/contest_submission.py": "contest_submission.py",
    "harness/harness.py": "harness.py",
}

REQUIRED_MODULES = {
    "harness/contest_submission.py",
    "src/harness/late_budget.py",
    "src/harness/round_controller.py",
    "src/harness/model_receipt.py",
    "src/models/official_round.py",
    "vendor/contest_baselines/b_calibration.py",
    "vendor/contest_baselines/c_calibration.py",
    "vendor/contest_baselines/model.py",
    "scripts/check_release.py",
}


def _normalized_sha256(path: Path) -> str:
    data = path.read_bytes()
    if path.suffix in {".py", ".json", ".md", ".txt"}:
        data = data.replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()


def file_checks(root, manifest, layout):
    errors = []
    hashes = manifest.get("runtime_sha256")
    if not isinstance(hashes, dict) or not REQUIRED_MODULES <= hashes.keys():
        return ["manifest_missing_required_modules"]
    for name, expected in hashes.items():
        rel = PurePosixPath(name)
        if rel.is_absolute() or ".." in rel.parts or "\\" in name or ":" in name:
            errors.append("invalid_manifest_path:" + name)
            continue
        path = root / (IMAGE_PATHS.get(name, name) if layout == "image" else name)
        if not path.is_file():
            errors.append("missing_file:" + name)
        elif _normalized_sha256(path) != expected:
            errors.append("hash_mismatch:" + name)
        elif path.suffix == ".py":
            try:
                compile(path.read_bytes(), str(path), "exec")
            except (SyntaxError, ValueError):
                errors.append("invalid_python:" + name)
    return errors


def checkpoint_check(path):
    import numpy as np
    from vendor.contest_baselines.model import Model
    from vendor.contest_baselines import b_calibration, c_calibration

    model = Model.load(path)
    x = np.tile(model.mean, (2, 1))
    c_state = model.meta.get("c_calibration")
    b_state = model.meta.get("b_calibration")
    calibration_tag = "none"
    if model.meta["task"] == "regression" and c_state:
        # Call the calibrator directly so Model.predict's fallback cannot hide
        # an unsupported checkpoint or mismatched legacy calibration module.
        known = sorted(c_state.get("moments", {}), key=float)
        if known and x.shape[1] > 21:
            x[:, 21] = float(known[0])
        _, logits = model.forward(model.normalized(x))
        raw = np.maximum(0, logits[:, 0] * model.meta["target_scale"] + model.meta["target_mean"])
        expected = c_calibration.predict(x, raw, c_state)
        if not np.all(np.isfinite(expected)):
            raise ValueError("c_calibration_nonfinite")
        np.testing.assert_allclose(model.predict(x), expected, rtol=1e-12, atol=1e-9)
        calibration_tag = str(c_state.get("version", "v8"))
    elif model.meta["task"] == "regression" and b_state:
        _, logits = model.forward(model.normalized(x))
        raw = np.maximum(0, logits[:, 0] * model.meta["target_scale"] + model.meta["target_mean"])
        expected = b_calibration.predict(x, raw, b_state)
        if not np.all(np.isfinite(expected)):
            raise ValueError("b_calibration_nonfinite")
        np.testing.assert_allclose(model.predict(x), expected, rtol=1e-12, atol=1e-9)
        calibration_tag = str(b_state.get("version", "unknown"))
    else:
        if not np.all(np.isfinite(model.predict(x))):
            raise ValueError("prediction_nonfinite")
    return {
        "task": model.meta["task"],
        "calibration": calibration_tag,
        "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
    }


def run(root, layout="source", require_sdk=False, checkpoint=None):
    root = Path(root).resolve()
    result = {
        "schema": 1,
        "release": None,
        "layout": layout,
        "ok": False,
        "errors": [],
        "warnings": [],
        "sdk_available": False,
    }
    try:
        manifest = json.loads((root / "release_manifest.json").read_text(encoding="utf-8"))
        result["release"] = manifest["release"]
        result["errors"] = file_checks(root, manifest, layout)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        result["errors"].append("invalid_or_missing_manifest:" + type(exc).__name__)
        return result
    # Never import code from a mixed or tampered bundle.
    if result["errors"]:
        return result
    sys.path[:0] = [str(root), str(root / "harness")]
    try:
        submission = importlib.import_module("contest_submission")
        from src.models.official_round import OfficialRound
        from vendor.contest_baselines import b_calibration, c_calibration

        required_params = (
            "selection_task",
            "smooth_weights",
            "calibration_mode",
            "validation_groups",
            "calibrate_b",
            "b_downward_alpha",
            "fast_c_calibration",
            "c_nasa_tol",
        )
        sig_params = inspect.signature(OfficialRound.continue_round).parameters
        for name in required_params:
            if name not in sig_params:
                result["errors"].append("trainer_missing_parameter:" + name)

        release = manifest["release"]
        if submission.calibration_mode(release) != "res_mlp_v2" or not submission.calibration_enabled(release):
            result["errors"].append("wrong_release_calibration")
        if "res_mlp_v2" not in c_calibration.SUPPORTED_MODES:
            result["errors"].append("missing_res_mlp_v2")
        if getattr(b_calibration, "VERSION", None) != "sign_log_v1":
            result["errors"].append("missing_b_calibration_sign_log_v1")
        if release in ("0.17", "0.18", "0.19", "0.20"):
            for attr in ("ULTRA_COMPACT_RELEASES", "B_CALIBRATION_RELEASES", "C_LATE_BUDGET_RELEASES"):
                if release not in getattr(submission, attr, ()):
                    result["errors"].append(f"release_flag_not_enabled:{attr}")

        entry = root / ("contest_submission.py" if layout == "image" else "harness/contest_submission.py")
        tree = ast.parse(entry.read_text(encoding="utf-8"))
        defaults = [
            n.value.args[1].value
            for n in tree.body
            if isinstance(n, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "RELEASE" for t in n.targets)
            and isinstance(n.value, ast.Call)
            and len(n.value.args) == 2
            and isinstance(n.value.args[1], ast.Constant)
        ]
        if defaults != [release]:
            result["errors"].append("entry_default_release_mismatch")

        if layout == "source":
            docker = (root / "harness.Dockerfile").read_text(encoding="utf-8").replace("\r\n", "\n")
            if f"ARG RSI_RELEASE={release}\n" not in docker:
                result["errors"].append("docker_default_release_mismatch")
            ignore = (root / ".dockerignore").read_text(encoding="utf-8").splitlines()
            for line in ("!scripts/", "!scripts/check_release.py", "!release_manifest.json", "!_sdk/"):
                if line not in ignore:
                    result["errors"].append("docker_context_missing_allow:" + line)

        try:
            result["sdk_available"] = importlib.util.find_spec("contest_harness") is not None
        except (ValueError, ImportError):
            pass
        if require_sdk:
            from contest_harness.cli import main as sdk_main
            if not callable(sdk_main):
                result["errors"].append("sdk_cli_not_callable")
        elif not result["sdk_available"]:
            result["warnings"].append("sdk_not_checked:source_check_is_not_sdk_acceptance")

        if checkpoint:
            result["checkpoint"] = checkpoint_check(checkpoint)
    except Exception as exc:
        result["errors"].append("interface_or_checkpoint_error:" + type(exc).__name__)
    result["ok"] = not result["errors"]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--layout", choices=("source", "image"), default="source")
    parser.add_argument("--require-sdk", action="store_true")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    report = run(args.root, args.layout, args.require_sdk, args.checkpoint)
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.report:
        args.report.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
