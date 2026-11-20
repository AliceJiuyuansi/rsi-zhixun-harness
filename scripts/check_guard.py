"""Fault-injection acceptance test for the controlled-RSI five-piece.

docs/04 §4 requires: 故意喂非法动作、超预算、超次数、坏 JSON、断数据，
系统都必须**降级而非崩溃**。 Each fault class below is injected into the real
harness (not a mock), and the check is that the run still finishes with a result.

    python scripts/check_guard.py          # exit 0 = all classes pass

Covered:
  F1 whitelist  - action enum, parameter names, parameter ranges
  F2 bad JSON   - malformed / fenced / non-object / non-numeric LLM replies
  F3 budget     - lifetime cap + rate cap (max_retrains / per-window)
  F4 amplitude  - a single action may not move a knob by more than factor x
  F5 kill-switch- env flag, STOP file, wall clock, config flag
  F6 stream data- mid-run feed failure degrades step by step, never raises
  F7 repeated   - consecutive failures trip the kill-switch automatically
  F8 missing    - missing dataset is reported, entrypoint still exits 0
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.harness.guard import Gatekeeper  # noqa: E402
from src.harness.killswitch import KillSwitch  # noqa: E402
from src.policy.bounds import AmplitudeLimiter  # noqa: E402
from src.policy.policies import ExternalPolicy, PolicyState, build_policy  # noqa: E402
from src.policy.whitelist import parse_proposal, validate_action  # noqa: E402
from src.schema import Action, GateDecision  # noqa: E402

RESULTS: list[tuple[str, bool, str]] = []
TMP = Path(tempfile.mkdtemp(prefix="rsi-guard-"))


def check(name: str, fn) -> None:
    try:
        detail = fn() or ""
        RESULTS.append((name, True, str(detail)))
    except AssertionError as exc:
        RESULTS.append((name, False, str(exc)))
    except Exception as exc:  # a crash here is itself a failure
        RESULTS.append((name, False, f"unexpected {type(exc).__name__}: {exc}"))


def tiny_cfg(policy: dict | None = None, n_steps: int = 40) -> dict:
    cfg = {
        "project": "rsi-zhixun-check",
        "workspace": "local",
        "data": {
            "kind": "synthetic", "n_features": 8, "n_classes": 4, "n_steps": n_steps,
            "batch_size": 32, "sigma": 1.0, "class_scale": 1.9, "seed": 7,
            "means_seed": 2026, "warmup_steps": 8, "label_lag": 1,
            "task": "classification",
            "drift": {"kind": "covariate_sudden", "start": 16, "magnitude": 2.2},
        },
        "model": {"kind": "hgb", "max_iter": 30, "learning_rate": 0.15,
                  "max_leaf_nodes": 15, "min_samples_leaf": 10,
                  "l2_regularization": 0.1, "random_state": 0},
        "policy": {
            "type": "trigger", "window": 10, "psi_threshold": 0.2,
            "error_threshold": 0.05, "retrain_cooldown": 2, "max_retrains": 4,
            "min_estimated_gain": 0.005, "promote_margin": 0.02,
            "rollback_threshold": 0.10, "rollback_patience": 3,
            "periodic_every": 8, "min_train_samples": 40,
        },
        "eval": {"rolling_window": 5},
        "cost": {"alpha_gpu": 1.0, "beta_label": 0.001, "gamma_token": 0.0},
    }
    if policy:
        cfg["policy"].update(policy)
    return cfg


def state(step: int = 10) -> PolicyState:
    return PolicyState(step=step, retrains=0, last_retrain_step=-1, psi_max=0.3,
                       error_gap=0.1, diagnosis={"kind": "covariate"})


# --------------------------------------------------------------------------
# F1 whitelist
# --------------------------------------------------------------------------
def f1_whitelist() -> str:
    ok, act, why = validate_action(Action("rm -rf /"))
    assert not ok and act.kind == "none", "unknown action must be rejected"
    ok, act, why = validate_action(Action("retrain", {"lr": 0.01}))
    assert not ok and "not allowed" in why, f"retrain takes no params: {why}"
    ok, act, why = validate_action(Action("finetune", {"lr": 1.0}))
    assert not ok and "outside allowed range" in why, f"lr=1.0 must be out of range: {why}"
    ok, act, why = validate_action(Action("finetune", {"lr": 0.001, "epochs": 3,
                                                      "momentum": 0.9}))
    assert not ok, "unknown param must be rejected"
    ok, act, why = validate_action(Action("swap_model", {"max_iter": 100,
                                                         "learning_rate": 0.05,
                                                         "max_leaf_nodes": 31}))
    assert not ok and "cap" in why, f"too many knobs at once: {why}"
    ok, act, why = validate_action(Action("finetune", {"lr": 0.001, "epochs": 3}))
    assert ok and act.kind == "finetune", f"legal action rejected: {why}"
    ok, act, why = validate_action("retrain")
    assert not ok, "a bare string is not an Action"

    # declared-but-not-implemented actions must be rejected, never turned into a
    # silent retrain (the executor for finetune/swap_model/reweight does not exist)
    g = Gatekeeper(tiny_cfg())
    out = g.review(GateDecision(True, "finetune please",
                                Action("finetune", {"lr": 0.001, "epochs": 3})),
                   step=5, retrains=0)
    assert not out.approved and out.action.kind == "none", out
    assert "not implemented" in out.reason, out.reason
    return "enum + param name + range + knob-count + not-implemented all enforced"


# --------------------------------------------------------------------------
# F2 bad JSON / hostile LLM output
# --------------------------------------------------------------------------
GARBAGE = [
    '{"kind": "retrain"',
    "not json at all",
    "[]",
    "null",
    '{"action": "sql_injection"}',
    '{"kind": "finetune", "params": {"lr": "fast"}}',
    '{"kind": "finetune", "params": {"lr": NaN}}',
    '{"kind": "retrain", "params": "drop table"}',
    '{"kind": 42}',
    '"retrain"',
    12345,
    b"\xff\xfe binary",
    "",
]


def f2_bad_json() -> str:
    for raw in GARBAGE:
        prop = parse_proposal(raw)  # must never raise
        assert prop.action.kind == "none", f"garbage accepted: {raw!r}"
        assert prop.rejected, f"garbage must carry a reason: {raw!r}"
    prop = parse_proposal('```json\n{"kind": "retrain"}\n```')
    assert prop.ok and prop.action.kind == "retrain", "fenced JSON should parse"
    prop = parse_proposal('{"action": "none"}')
    assert prop.ok and prop.action.kind == "none"
    prop = parse_proposal('{"kind": "retrain", "est_gain": "lots"}')
    assert prop.ok and prop.action.est_gain == 0.0, "non-numeric estimate must become 0"

    # proposer that raises / returns junk must degrade to none, not crash
    def raiser(_s):
        raise TimeoutError("PyTRIO unreachable")

    for proposer, expect in ((raiser, "proposer failed"),
                             (lambda s: "garbage", "proposal rejected"),
                             (lambda s: None, "proposal rejected"),
                             (lambda s: '{"kind":"retrain","params":{"lr":0.01}}',
                              "proposal rejected")):
        pol = ExternalPolicy({"type": "external", "proposer": proposer})
        dec = pol.decide(state())
        assert not dec.approved and dec.action.kind == "none", f"bad proposer approved: {dec}"
        assert expect in dec.reason, f"reason missing '{expect}': {dec.reason}"
    return f"{len(GARBAGE)} garbage payloads + 4 bad proposers all -> none"


# --------------------------------------------------------------------------
# F3 budget (lifetime + rate)
# --------------------------------------------------------------------------
def f3_budget() -> str:
    guard = Gatekeeper(tiny_cfg({"max_retrains": 2, "max_retrains_per_window": 2,
                                 "window": 50}))
    dec = GateDecision(True, "test", Action("retrain"))
    approved = 0
    for step in (10, 20, 30):
        out = guard.review(dec, step=step, retrains=approved)
        if out.approved:
            approved += 1
            guard.commit(out.action, step)
    assert approved == 2, f"lifetime cap not enforced (approved {approved})"
    out = guard.review(dec, step=40, retrains=approved)
    assert not out.approved and "budget reject" in out.reason, out.reason

    guard2 = Gatekeeper(tiny_cfg({"max_retrains": 10, "max_retrains_per_window": 1,
                                  "window": 30}))
    out = guard2.review(dec, step=10, retrains=0)
    assert out.approved
    guard2.commit(out.action, 10)
    out = guard2.review(dec, step=15, retrains=1)
    assert not out.approved and "rate cap" in out.reason, out.reason
    out = guard2.review(dec, step=45, retrains=1)
    assert out.approved, "rate window should have rolled over"
    return "lifetime cap 2/2 then reject; rate cap 1/30 steps"


# --------------------------------------------------------------------------
# F4 amplitude cap
# --------------------------------------------------------------------------
def f4_amplitude() -> str:
    lim = AmplitudeLimiter(factor=2.0)
    lim.commit("swap_model", {"learning_rate": 0.1})
    params, notes = lim.clip("swap_model", {"learning_rate": 0.5})
    assert abs(params["learning_rate"] - 0.2) < 1e-9, params
    assert notes, "clip must be reported"

    guard = Gatekeeper(tiny_cfg({"amplitude_factor": 2.0}),
                       executable=("none", "retrain", "swap_model"))
    guard.limiter.commit("swap_model", {"max_iter": 100})
    out = guard.review(GateDecision(True, "hpo", Action("swap_model", {"max_iter": 250})),
                       step=5, retrains=0)
    assert out.approved and out.action.params["max_iter"] == 200.0, out.action.params
    return "0.1 -> 0.5 clipped to 0.2; max_iter 100 -> 250 clipped to 200"


# --------------------------------------------------------------------------
# F5 kill-switch sources
# --------------------------------------------------------------------------
def f5_kill_switch() -> str:
    assert KillSwitch({}, env={"RSI_STOP": "1"}).tripped()
    assert KillSwitch({}, env={"RSI_KILL_SWITCH": "true"}).tripped()
    assert not KillSwitch({}, env={}).tripped()

    cfg_flag = KillSwitch({"policy": {"kill_switch": True}}, env={})
    assert cfg_flag.tripped() and "kill_switch" in cfg_flag.reason

    stopdir = TMP / "stoprun"
    stopdir.mkdir(parents=True, exist_ok=True)
    (stopdir / "STOP").write_text("stop", encoding="utf-8")
    file_ks = KillSwitch({}, root=stopdir, env={})
    assert file_ks.tripped() and "STOP file" in file_ks.reason

    wall = KillSwitch({"policy": {"max_wall_seconds": 0.01}}, env={})
    time.sleep(0.02)
    assert wall.tripped() and "wall-clock" in wall.reason

    auto = KillSwitch({"policy": {"max_consecutive_failures": 2}}, env={})
    auto.note_step_error("boom")
    auto.note_step_error("boom")
    assert auto.tripped() and "consecutive" in auto.reason

    g = Gatekeeper(tiny_cfg(), kill=KillSwitch({}, env={"RSI_STOP": "1"}))
    out = g.review(GateDecision(True, "retrain please", Action("retrain")), step=9, retrains=0)
    assert not out.approved and out.action.kind == "none" and "kill-switch" in out.reason
    return "env / config / STOP file / wall clock / auto all freeze the model"


# --------------------------------------------------------------------------
# F6 mid-run data feed failure
# --------------------------------------------------------------------------
class FailingStream:
    """Wraps a real Stream and drops the feed on selected steps."""

    def __init__(self, inner, fail_steps):
        self.inner = inner
        self.steps = inner.steps
        self.fail_steps = set(int(s) for s in fail_steps)

    def batch(self, step):
        return self.inner.batch(step)

    def warmup(self, n):
        return self.inner.warmup(n)

    def concat_steps(self, steps):
        return self.inner.concat_steps(steps)

    def recent(self, step, window_steps=40):
        if int(step) in self.fail_steps:
            raise ConnectionError("sensor gateway dropped the batch")
        return self.inner.recent(step, window_steps)

    def history(self, step, label_lag=1, window_steps=40):
        return self.inner.history(step, label_lag, window_steps)


def _run_with_stream(stream_factory, outdir: Path, cfg: dict) -> dict:
    import src.harness.loop as loop
    from src.data.loader import build_stream as real_build

    def patched(c, seed):
        st, meta = real_build(c, seed)
        return stream_factory(st), meta

    original = loop.build_stream
    loop.build_stream = patched
    try:
        return loop.run_experiment(cfg, "fault-injected", "trigger", 0,
                                   str(outdir), swan_mode="disabled")
    finally:
        loop.build_stream = original


def f6_bad_stream() -> str:
    outdir = TMP / "f6"
    cfg = tiny_cfg()
    # three isolated drops: below the consecutive-failure threshold, so this
    # checks pure per-step degradation (F7 covers the automatic kill-switch).
    summary = _run_with_stream(
        lambda st: FailingStream(st, [20, 21, 24]), outdir, cfg)
    assert summary["degraded_steps"] == 3, f"degraded_steps={summary['degraded_steps']}"
    assert summary["guard"]["kill_switch"]["stopped"] is False, "should not trip here"
    assert summary["n_steps"] > 0
    audit = (outdir / "audit.jsonl").read_text(encoding="utf-8").strip().splitlines()
    events = [json.loads(line)["event"] for line in audit]
    assert events.count("degraded") == 3, events
    assert "run_end" in events and "run_start" in events, events
    rows = json.loads((outdir / "metrics.json").read_text(encoding="utf-8"))
    assert any(r.get("degraded") for r in rows), "degraded rows must be marked"
    return f"4 dropped feeds -> {summary['degraded_steps']} degraded steps, run finished, audit has {len(audit)} lines"


# --------------------------------------------------------------------------
# F7 repeated failures trip the kill-switch automatically
# --------------------------------------------------------------------------
def f7_consecutive_failures() -> str:
    outdir = TMP / "f7"
    cfg = tiny_cfg({"max_consecutive_failures": 3})
    summary = _run_with_stream(
        lambda st: FailingStream(st, list(range(12, 100))), outdir, cfg)
    assert summary["guard"]["kill_switch"]["stopped"] is True, summary["guard"]
    assert "consecutive" in summary["guard"]["kill_switch"]["reason"], summary["guard"]
    assert summary["degraded_steps"] > 0
    return f"3 consecutive failures -> kill-switch: {summary['guard']['kill_switch']['reason']}"


# --------------------------------------------------------------------------
# F8 missing dataset: entrypoint still exits 0 and reports the failure
# --------------------------------------------------------------------------
def f8_missing_data() -> str:
    import scripts.entrypoint as entry

    out = TMP / "f8"
    old = {k: os.environ.get(k) for k in ("RSI_OUT", "RSI_STREAM", "RSI_POLICIES",
                                          "RSI_SWAN_MODE")}
    os.environ.update({"RSI_OUT": str(out), "RSI_STREAM": "no_such_stream",
                       "RSI_POLICIES": "trigger", "RSI_SWAN_MODE": "disabled"})
    try:
        rc = entry.main()
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    assert rc == 0, f"entrypoint must exit 0, got {rc}"
    report = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert report["failures"], "missing data must be reported, not hidden"
    assert report["exit_code"] == 0
    return f"missing config -> exit 0, failure recorded: {list(report['failures'].values())[0][:60]}"


# --------------------------------------------------------------------------
# Bonus: a healthy guarded run still behaves (no false rejections)
# --------------------------------------------------------------------------
def healthy_run() -> str:
    outdir = TMP / "healthy"
    summary = _run_with_stream(lambda st: st, outdir, tiny_cfg())
    assert summary["degraded_steps"] == 0, summary["degraded_steps"]
    assert summary["retrains"] > 0, "guard should not block a healthy run"
    guard = summary["guard"]
    assert guard["rejects"]["whitelist"] == 0, guard["rejects"]
    assert (outdir / "guard.json").exists()
    return (f"{summary['retrains']} retrains, {summary['promotions']} promotions, "
            f"0 rejects, audit={summary['audit']['records']} records")


def main() -> int:
    print("=" * 70)
    print("受控 RSI 五件套 · 故障注入验收 / controlled-RSI fault injection")
    print("=" * 70)
    check("F1 白名单 whitelist", f1_whitelist)
    check("F2 坏 JSON / bad proposals", f2_bad_json)
    check("F3 次数上限 budget", f3_budget)
    check("F4 幅度上限 amplitude", f4_amplitude)
    check("F5 kill-switch", f5_kill_switch)
    check("F6 断数据 stream faults", f6_bad_stream)
    check("F7 连续异常 auto kill", f7_consecutive_failures)
    check("F8 数据缺失 missing dataset", f8_missing_data)
    check("  健康跑到尾 healthy run", healthy_run)

    width = max(len(n) for n, _, _ in RESULTS)
    passed = 0
    print()
    for name, ok, detail in RESULTS:
        flag = "PASS" if ok else "FAIL"
        print(f"[{flag}] {name:<{width}}  {detail}")
        passed += int(ok)
    print("-" * 70)
    print(f"{passed}/{len(RESULTS)} fault classes degraded gracefully")
    shutil.rmtree(TMP, ignore_errors=True)
    return 0 if passed == len(RESULTS) else 1


if __name__ == "__main__":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    raise SystemExit(main())
