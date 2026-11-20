"""Run one full harness round locally against the SDK's in-process fake host.

This is the "cheap rehearsal" before spending one of the 10 platform trial runs:
it drives `harness/harness.py train` with a real start frame, real parent
checkpoint and the official public split, then exercises `predict`.

    python scripts/test_harness_local.py
    python scripts/test_harness_local.py --rounds 3      # several rounds, state carries over
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
SDK = ROOT / "_sdk" / "contest-harness-sdk-1.0.4"
sys.path.insert(0, str(SDK))

from tests.support import Host  # noqa: E402

PACKAGE = ROOT / "data/raw/official_a"


def sha(path: Path) -> str:
    import hashlib
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def start_frame(train_path: Path, validation_path: Path, parent_sha: str, round_index: int) -> dict:
    parent = {
        "artifact_id": "baseline-a" if round_index <= 1 else f"model-test-{round_index - 1}",
        "artifact_sha256": "sha256:" + "cd" * 32,
        "artifact": {"checkpoint_paths": ["checkpoint.npz"], "code_paths": []},
        "files": [{"path": "checkpoint.npz", "sha256": parent_sha}],
    }
    return {
        "type": "start",
        "protocol": "contest.harness.workspace.v2",
        "run_id": "local-run",
        "stage_id": "public-development",
        "prompt": "工业模型续训任务：本轮请继续训练父检查点并提交候选。",
        "prompt_version": "industrial-workspace-flexible-v3",
        "rounds_per_dataset": 10,
        "dataset_key": "A",
        "round_index": round_index,
        "parent_checkpoint_sha256": parent_sha,
        "task_contract": {
            "dataset_key": "A",
            "task": "classification",
            "metric": "macro_f1",
            "metric_direction": "higher",
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


def stub_model(bound_messages, options):
    """Pretend to be the platform LLM: deterministic, no network."""
    text = "继续训练 5 轮，然后在公开验证集上比较，未提升则保持父模型。"
    return {"request_id": "call-stub-1",
            "choices": [{"message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}}


def run(rounds: int, docker: bool = False, image: str = "zhixun-rsi-harness:0.1") -> int:
    tmp = Path(tempfile.mkdtemp(prefix="harness-local-"))
    data = tmp / "data"
    data.mkdir()
    shutil.copyfile(PACKAGE / "public/train.jsonl", data / "train.jsonl")
    shutil.copyfile(PACKAGE / "public/validation.jsonl", data / "validation.jsonl")

    parent_dir = tmp / "previous"
    parent_dir.mkdir()
    shutil.copyfile(PACKAGE / "checkpoint.npz", parent_dir / "checkpoint.npz")
    parent_sha = sha(parent_dir / "checkpoint.npz")

    env = dict(os.environ)
    env.update({
        "PYTHONPATH": os.pathsep.join([str(SDK), str(ROOT)]),
        "CONTEST_APP": str(ROOT),
        "CONTEST_SUBMISSION": "contest_submission",
        "CONTEST_PARENT_ROOT": str(parent_dir),
        "CONTEST_WORK": str(tmp / "workspace"),
        "CONTEST_HARNESS": str(ROOT / "harness/harness.py"),
        "PYTHONIOENCODING": "utf-8",
    })
    (tmp / "workspace").mkdir()

    def command(*argv: str) -> list[str]:
        if not docker:
            return [sys.executable, str(ROOT / "harness/harness.py"), *argv]
        return ["docker", "run", "--rm", "-i", "--network", "none",
                "-v", f"{data}:/data:ro",
                "-v", f"{parent_dir}:/previous:ro",
                "-v", f"{tmp / 'workspace'}:/workspace",
                "-e", "CONTEST_HARNESS=/app/harness.py",
                "-e", "CONTEST_PARENT_ROOT=",
                image, *argv]

    run_env = None if docker else env

    # describe first: the platform refuses to train if this does not pass
    described = subprocess.run(command("describe"), env=run_env or None,
                               capture_output=True, text=True, encoding="utf-8",
                               errors="replace", timeout=180)
    print("describe:", described.stdout.strip()[:200], described.stderr.strip()[:200])
    if described.returncode != 0:
        return 1

    current_ckpt = parent_dir / "checkpoint.npz"
    current_sha = parent_sha
    for k in range(1, rounds + 1):
        train_path = data / "train.jsonl" if not docker else "/data/train.jsonl"
        val_path = data / "validation.jsonl" if not docker else "/data/validation.jsonl"
        frame = start_frame(train_path, val_path, current_sha, k)
        process = subprocess.Popen(
            command("train"),
            env=None if docker else env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace")
        host = Host(frame, stub_model, history=[{"id": f"round-{i}"} for i in range(k - 1)])
        thread = host.serve(process)
        stderr = process.stderr.read()
        code = process.wait(timeout=300)
        thread.join(timeout=10)
        print(f"\n--- round {k} exit={code} ---")
        print(stderr.strip()[-1200:])
        if code != 0 or host.failure:
            print("FAILED:", host.failure)
            return 1
        record = host.record or {}
        print("selected:", record.get("selected_artifact_id"),
              "call_ids:", record.get("model_call_ids"))

    # predict on a few validation rows with the final artifact path
    print("\n--- predict ---")
    sample = json.loads((data / "validation.jsonl").read_text(encoding="utf-8").splitlines()[0])
    env["CONTEST_ARTIFACT_PATH"] = str(current_ckpt)
    if docker:
        proc = subprocess.run(
            ["docker", "run", "--rm", "-i", "--network", "none",
             "-v", f"{parent_dir}:/model:ro",
             "-e", "CONTEST_ARTIFACT_PATH=/model/checkpoint.npz",
             image, "predict"],
            input=json.dumps({"task": "classification", "inputs": sample["inputs"]}) + "\n",
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=180)
    else:
        proc = subprocess.run([sys.executable, str(ROOT / "harness/harness.py"), "predict"],
                              input=json.dumps({"task": "classification", "inputs": sample["inputs"]}) + "\n",
                              env=env, capture_output=True, text=True, timeout=120)
    print("stdout:", proc.stdout.strip()[:300])
    print("rc:", proc.returncode, "stderr:", proc.stderr.strip()[:300])
    shutil.rmtree(tmp, ignore_errors=True)
    return 0 if proc.returncode == 0 and proc.stdout.strip() else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--docker", action="store_true", help="run inside the harness image")
    ap.add_argument("--image", default="zhixun-rsi-harness:0.1")
    args = ap.parse_args()
    raise SystemExit(run(args.rounds, docker=args.docker, image=args.image))
