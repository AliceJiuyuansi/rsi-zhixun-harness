"""公开包真实续训/父子状态/逐条预测接线验证；本地替身宿主，不是SDK验收。"""
import argparse
import atexit
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=root)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--dataset", choices=["A", "B", "C"], required=True)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--swanlab", action="store_true", help="训练开始前建立实时离线记录；需要开发用SwanLab")
    args = parser.parse_args()
    if not 1 <= args.rounds <= 10:
        parser.error("rounds must be in 1..10")
    source = args.source_root.resolve()
    os.environ["CONTEST_APP"] = str(source)
    sys.path[:0] = [str(source), str(source / "harness")]
    import numpy as np
    import contest_submission as submission
    from src.models.official_round import OfficialRound
    from vendor.contest_baselines.data import read_dataset
    from vendor.contest_baselines.model import Model, checkpoint_digest

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    package = args.package.resolve()
    task = "classification" if args.dataset == "A" else "regression"
    selected = package / "checkpoint.npz"
    selected_id = "public-baseline"
    artifacts = {selected_id: selected}
    history = []
    state_dir = output / "initial-state"
    state_dir.mkdir()
    swan = None
    if args.swanlab:
        for key, name in [("SWANLAB_ROOT", "swanlab-root"), ("SWANLAB_CONFIG_DIR", "swanlab-config"),
                          ("SWANLAB_SECRETS_DIR", "swanlab-secrets")]:
            os.environ[key] = str(output/name)
        import swanlab as swan
        swan.init(project="rsi-zhixun-paper-c", name=f"public-{args.dataset}-continuation", mode="offline",
                  log_dir=str(output/"swanlog"), config={"dataset": args.dataset, "rounds": args.rounds,
                  "checkpoint_sha256": checkpoint_digest(selected), "purpose": "local stub host, real public training",
                  "calibration_method": os.environ.get("RSI_C_CALIBRATION_METHOD", "config_default")})
        atexit.register(swan.finish)
        epoch_sequence = 0
        original_fit = Model.fit
        def logged_fit(model, x, y, epochs):
            nonlocal epoch_sequence
            losses = original_fit(model, x, y, epochs)
            epoch_sequence += 1
            swan.log({"round": round_index, "epoch": int(model.meta["epochs"]),
                      "loss": float(losses[-1])}, step=epoch_sequence)
            return losses
        Model.fit = logged_fit
    results = []
    for round_index in range(1, args.rounds + 1):
        finishes = []
        next_state = output / f"state-{round_index}"
        def submit_candidate(path, **kwargs):
            identity = f"local-{round_index}"
            artifacts[identity] = Path(path) / "checkpoint.npz"
            return {"artifact_id": identity}
        rnd = SimpleNamespace(
            task={"task": task}, dataset_key=args.dataset, round_index=round_index,
            stage_id=f"{args.dataset}-fixed-public-stage", max_output_tokens=256,
            parent=SimpleNamespace(artifact_id=selected_id, declared_paths=lambda: ["checkpoint.npz"],
                                   path=lambda *a, **kw: selected),
            data=SimpleNamespace(train_path=package/"public/train.jsonl", validation_path=package/"public/validation.jsonl"),
            history=lambda: list(history), history_ids=[f"local-{i}" for i in range(1, round_index)],
            call_ids=[f"local-stub-{round_index}"],  # 仅本地替身ID，不是平台调用证据
            model=lambda *a, **kw: SimpleNamespace(response={}),
            submit_candidate=submit_candidate,
            save_state=lambda path: shutil.copytree(path, next_state),
            finish=lambda **kw: finishes.append(kw))
        submission.WORKSPACE = output / "workspace"
        submission.STATE_DIR = state_dir
        submission.train(rnd)
        if len(finishes) != 1:
            raise AssertionError("expected one finish per round")
        finish = finishes[0]
        if finish["configuration"].get("execution", {}).get("status") == "training_error":
            raise RuntimeError("real training failed; inspect the round log")
        selected_id = finish["selected_artifact_id"]
        selected = artifacts[selected_id]
        state_dir = next_state
        selected_model = Model.load(selected)
        data = read_dataset(package/"public/validation.jsonl", task, selected_model.meta["classes"] or None,
                            expected_paths=selected_model.meta["paths"])
        metrics = OfficialRound().evaluate(selected_model, data["x"], data["y"])
        if swan is not None:
            swan.log({"round_completed": round_index, **{f"selected/{k}":v for k,v in metrics.items()
                      if isinstance(v,(int,float))}})
        history.append({"round_index": round_index, "configuration": finish["configuration"],
                        "validation_metrics": metrics, "decision": "promote" if finish["configuration"]["promoted"] else "retain"})
        results.append({"round": round_index, "epochs": finish["configuration"]["epochs"],
                        "promoted": finish["configuration"]["promoted"], "metrics": metrics,
                        "checkpoint_sha256": checkpoint_digest(selected),
                        "configuration": finish["configuration"]})
        print(f"{args.dataset} round={round_index} epochs={results[-1]['epochs']} promoted={results[-1]['promoted']}", flush=True)

    batch = selected_model.predict(data["x"])
    rows = [json.loads(line) for line in (package/"public/validation.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    single = []
    for row in rows:
        answer = submission.predict(SimpleNamespace(inputs=row["inputs"]), selected.parent)
        if task == "regression":
            single.append(answer["value"])
        else:
            single.append(selected_model.meta["classes"].index(answer[0]["label"]))
    expected = batch if task == "regression" else np.argmax(batch, axis=1)
    delta = float(np.max(np.abs(np.asarray(single)-expected)))
    np.testing.assert_allclose(single, expected, rtol=1e-12, atol=1e-9)
    report = {"purpose": "public real-training local wiring; not SDK or platform",
              "dataset": args.dataset, "rounds": results, "rows": len(rows),
              "batch_prediction_sha256": hashlib.sha256(np.asarray(batch).tobytes()).hexdigest(),
              "single_vs_batch_max_delta": delta,
              "selected_checkpoint_sha256": checkpoint_digest(selected),
              "validation_sha256": data["sha256"]}
    (output/"result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"dataset": args.dataset, "metrics": results[-1]["metrics"], "single_vs_batch_max_delta": delta}, ensure_ascii=False), flush=True)
    if swan is not None:
        swan.finish()
        atexit.unregister(swan.finish)


if __name__ == "__main__":
    main()
