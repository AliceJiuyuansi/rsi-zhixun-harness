"""Run full harness rounds locally against the SDK's in-process fake host.

This is the cheap rehearsal before spending a platform trial:
it drives `harness/harness.py train` with a real start frame, the real parent
checkpoint and the official public split, then exercises `predict`.

    python scripts/test_harness_local.py --dataset A --rounds 10
    python scripts/test_harness_local.py --dataset C --rounds 6 --max-train-rows 12000
    python scripts/test_harness_local.py --dataset A --rounds 2 --docker --image zhixun-rsi-harness:0.2

v2 additions:
  * A/B/C official packages (task + metric switch);
  * state and parent chain really carry over between rounds;
  * optional per-round growth of the training file (new stage simulation);
  * the stub model returns a JSON verdict, sometimes in `reasoning_content`.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SDK = ROOT / "_sdk" / "contest-harness-sdk-1.0.5"
sys.path.insert(0, str(SDK))

from tests.support import Host  # noqa: E402

DATASETS = {
    "A": {"package": ROOT / "data/raw/official_a", "task": "classification", "metric": "macro_f1"},
    "B": {"package": ROOT / "data/raw/official_b", "task": "regression", "metric": "rmse"},
    "C": {"package": ROOT / "data/raw/official_c", "task": "regression", "metric": "rmse"},
}


def sha(path: Path) -> str:
    import hashlib
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def start_frame(package: dict, train_path: Path, validation_path: Path,
                parent_sha: str, round_index: int, stage_id: str, parent_id: str) -> dict:
    parent = {
        "artifact_id": parent_id,
        "artifact_sha256": "sha256:" + "cd" * 32,
        "artifact": {"checkpoint_paths": ["checkpoint.npz"], "code_paths": []},
        "files": [{"path": "checkpoint.npz", "sha256": parent_sha}],
    }
    return {
        "type": "start",
        "protocol": "contest.harness.workspace.v2",
        "run_id": "local-run",
        "stage_id": stage_id,
        "prompt": "工业模型续训任务：本轮请继续训练父检查点并提交候选。",
        "prompt_version": "industrial-workspace-flexible-v3",
        "rounds_per_dataset": 10,
        "dataset_key": package["key"],
        "round_index": round_index,
        "parent_checkpoint_sha256": parent_sha,
        "task_contract": {
            "dataset_key": package["key"],
            "task": package["task"],
            "metric": package["metric"],
            "metric_direction": "higher" if package["task"] == "classification" else "lower",
            "public_projection": {"train_path": str(train_path),
                                  "validation_path": str(validation_path)},
        },
        "model_name": "base:local-stub",
        "parent_artifact": parent,
        "artifacts": [parent],
        "state_sha256": "",
        "state_path": None,
        "max_output_tokens": 256,
        "resource_limits": {"artifact_upload_protocol": "chunked-v1",
                            "artifact_bytes": 512 * 1024 * 1024,
                            "state_bytes": 512 * 1024 * 1024,
                            "candidate_upload_bytes_per_round": 1024 * 1024 * 1024,
                            "state_upload_bytes_per_round": 512 * 1024 * 1024},
        "current_validation": {},
    }


def make_stub(round_box: dict, mode: str):
    """Pretend to be the platform LLM. Deterministic, offline, JSON verdict."""

    def stub_model(bound_messages, options):
        round_index = round_box.get("round", 1)
        if mode == "empty":
            return {"request_id": f"call-stub-{round_index}",
                    "choices": [{"message": {"role": "assistant", "content": ""}, "finish_reason": "length"}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 256, "total_tokens": 266}}
        if mode == "hold":
            decision, epochs = "hold", 0
        elif round_index <= 2:
            decision, epochs = "train", 2
        else:
            decision, epochs = "train", 1
        verdict = json.dumps({"decision": decision, "epochs": epochs, "lr": 0.0015,
                              "reason": f"本地彩排第{round_index}轮：{decision}"},
                             ensure_ascii=False)
        message = {"role": "assistant", "content": verdict}
        # Every other round, exercise the Qwen-style reasoning_content fallback.
        if round_index % 2 == 0:
            message = {"role": "assistant", "content": "", "reasoning_content": verdict}
        return {"request_id": f"call-stub-{round_index}",
                "choices": [{"message": message, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}}

    return stub_model


def _prefix_rows(path: Path, keep: int) -> list[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return lines[:max(32, keep)] if keep < len(lines) else lines


def run(dataset: str, rounds: int, *, docker: bool = False, image: str = "zhixun-rsi-harness:0.2",
        grow: bool = False, stage_every: int = 2, max_train_rows: int = 0,
        timeout: int = 900, stub_mode: str = "train", save_selected: Path | None = None,
        package_dir: Path | None = None) -> int:
    package = dict(DATASETS[dataset])
    package["key"] = dataset
    source = package_dir or package["package"]
    tmp = Path(tempfile.mkdtemp(prefix=f"harness-local-{dataset}-"))
    data = tmp / "data"
    data.mkdir()
    state_dir = tmp / "state"
    state_dir.mkdir()

    full_train = source / "public/train.jsonl"
    validation = source / "public/validation.jsonl"
    train_lines = full_train.read_text(encoding="utf-8").splitlines()
    if max_train_rows:
        train_lines = train_lines[:max_train_rows]
    full_count = len(train_lines)

    parent_dir = tmp / "previous"
    parent_dir.mkdir()
    shutil.copyfile(source / "checkpoint.npz", parent_dir / "checkpoint.npz")
    parent_sha = sha(parent_dir / "checkpoint.npz")
    parent_id = f"baseline-{dataset}"

    env = dict(os.environ)
    env.update({
        "PYTHONPATH": os.pathsep.join([str(SDK), str(ROOT)]),
        "CONTEST_APP": str(ROOT),
        "CONTEST_SUBMISSION": "contest_submission",
        "CONTEST_PARENT_ROOT": str(parent_dir),
        "CONTEST_STATE_DIR": str(state_dir),
        "CONTEST_WORK": str(tmp / "workspace"),
        "CONTEST_HARNESS": str(ROOT / "harness/harness.py"),
        "PYTHONIOENCODING": "utf-8",
    })
    (tmp / "workspace").mkdir()

    def command(*argv: str) -> list[str]:
        if not docker:
            return [sys.executable, str(ROOT / "harness/harness.py"), *argv]
        return ["docker", "run", "--rm", "-i", "--network", "none",
                "-v", f"{data}:/data:rw",
                "-v", f"{parent_dir}:/previous:rw",
                "-v", f"{state_dir}:/state:rw",
                "-v", f"{tmp / 'workspace'}:/workspace",
                "-e", "CONTEST_HARNESS=/app/harness.py",
                "-e", "CONTEST_PARENT_ROOT=",
                "-e", f"CONTEST_STATE_DIR=/state",
                image, *argv]

    run_env = None if docker else env

    described = subprocess.run(command("describe"), env=run_env,
                               capture_output=True, text=True, encoding="utf-8",
                               errors="replace", timeout=180)
    print("describe:", described.stdout.strip()[:200], described.stderr.strip()[:200])
    if described.returncode != 0:
        return 1

    history: list[dict] = []
    round_box = {"round": 1}
    for k in range(1, rounds + 1):
        keep = full_count
        if grow:
            keep = int(round(full_count * (0.4 + 0.6 * (k - 1) / max(rounds - 1, 1))))
            keep = min(keep, max(32, max_train_rows or full_count))
        (data / "train.jsonl").write_text(
            "\n".join(train_lines[:max(32, keep)]) + "\n", encoding="utf-8")
        shutil.copyfile(validation, data / "validation.jsonl")
        stage_id = f"{dataset}-stage-{(k - 1) // max(stage_every, 1) + 1:02d}"
        frame = start_frame(package, "/data/train.jsonl" if docker else data / "train.jsonl",
                            "/data/validation.jsonl" if docker else data / "validation.jsonl",
                            parent_sha, k, stage_id, parent_id)
        round_box["round"] = k
        process = subprocess.Popen(
            command("train"),
            env=None if docker else env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace")
        host = Host(frame, make_stub(round_box, stub_mode), history=list(history))
        thread = host.serve(process)
        stderr = process.stderr.read()
        code = process.wait(timeout=timeout)
        thread.join(timeout=10)
        print(f"\n--- {dataset} round {k} ({stage_id}, rows={max(32, keep)}) exit={code} ---")
        print(stderr.strip()[-1600:])
        if code != 0 or host.failure:
            print("FAILED:", host.failure)
            return 1

        record = host.record or {}
        selected = record.get("selected_artifact_id", parent_id)
        state_file = tmp / "workspace" / f"state-round-{k}" / "harness_state.json"
        state = json.loads(state_file.read_text(encoding="utf-8")) if state_file.exists() else {}
        metric_name = state.get("metric", package["metric"])
        history.append({
            "id": f"{dataset}-round-{k:02d}",
            "round_index": k,
            "decision": "promote" if selected != parent_id else "retain",
            "configuration": record.get("configuration") or {},
            "validation_metrics": {metric_name: state.get("last", 0.0)},
        })
        print("selected:", selected, "| state:", {key: state.get(key) for key in
              ("metric", "best", "last", "last_gain", "stalled", "updates", "promotions", "requested_epochs_total")})

        if selected != parent_id:
            candidate = tmp / "workspace" / f"candidate-round-{k}" / "checkpoint.npz"
            if not candidate.exists():
                print("FAILED: selected candidate checkpoint missing:", candidate)
                return 1
            shutil.copyfile(candidate, parent_dir / "checkpoint.npz")
            parent_sha = sha(parent_dir / "checkpoint.npz")
            parent_id = f"model-local-{dataset}-{k}"
        # promote this round's state to /state for the next process
        if state_file.exists():
            shutil.copyfile(state_file, state_dir / "harness_state.json")

    print("\n--- predict ---")
    sample = json.loads(validation.read_text(encoding="utf-8").splitlines()[0])
    task = package["task"]
    predict_input = json.dumps({"task": task, "inputs": sample["inputs"]}) + "\n"
    if docker:
        proc = subprocess.run(
            ["docker", "run", "--rm", "-i", "--network", "none",
             "-v", f"{parent_dir}:/model:ro",
             "-e", "CONTEST_ARTIFACT_PATH=/model/checkpoint.npz",
             image, "predict"],
            input=predict_input, capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=180)
    else:
        env["CONTEST_ARTIFACT_PATH"] = str(parent_dir / "checkpoint.npz")
        proc = subprocess.run([sys.executable, str(ROOT / "harness/harness.py"), "predict"],
                              input=predict_input, env=env, capture_output=True,
                              text=True, encoding="utf-8", errors="replace", timeout=120)
    print("stdout:", proc.stdout.strip()[:400])
    print("rc:", proc.returncode, "stderr:", proc.stderr.strip()[:300])
    ok = proc.returncode == 0 and proc.stdout.strip()
    if not ok:
        print("TEMP:", tmp)
    else:
        if save_selected is not None:
            save_selected.parent.mkdir(exist_ok=True, parents=True)
            shutil.copyfile(parent_dir / "checkpoint.npz", save_selected)
        shutil.rmtree(tmp, ignore_errors=True)
    return 0 if ok else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="A", choices=sorted(DATASETS))
    parser.add_argument("--package", type=Path, help="official dataset package (overrides data/raw/official_*)")
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--grow", action="store_true", help="grow the training file each round")
    parser.add_argument("--stage-every", type=int, default=2)
    parser.add_argument("--max-train-rows", type=int, default=0)
    parser.add_argument("--docker", action="store_true", help="run inside the harness image")
    parser.add_argument("--image", default="zhixun-rsi-harness:0.2")
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--stub", default="train", choices=("train", "hold", "empty"))
    parser.add_argument("--save-selected", type=Path, help="copy the final selected checkpoint after a successful rehearsal")
    args = parser.parse_args()
    raise SystemExit(run(args.dataset, args.rounds, docker=args.docker, image=args.image,
                         grow=args.grow, stage_every=args.stage_every,
                         max_train_rows=args.max_train_rows, timeout=args.timeout,
                         stub_mode=args.stub, save_selected=args.save_selected, package_dir=args.package))
