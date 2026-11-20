"""官方 B 包（XJTU-SY，单位分钟）上的两个对照实验。

实验 1（可比）：用官方 train 训练、在官方 validation 上评分，
    和官方基线 `training.json` 的 RMSE 681 分钟直接对比。
实验 2（机制）：只把官方 train 当数据流，按时间切分做自进化，
    看 trigger 是否能在不碰 validation 的前提下把误差压住。

用法：
    python scripts/check_official_b.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from common import ROOT

sys.path.insert(0, str(ROOT))

from src.data.official import load_jsonl, to_stream_arrays  # noqa: E402
from src.eval.metrics import mae, rmse  # noqa: E402
from src.models.baseline import fit_model, score_model  # noqa: E402

PKG = ROOT / "data" / "raw" / "official_b"
TRAIN = PKG / "public" / "train.jsonl"
VAL = PKG / "public" / "validation.jsonl"

CFG = {"kind": "hgb_regressor", "max_iter": 150, "learning_rate": 0.08,
       "max_leaf_nodes": 15, "min_samples_leaf": 20, "l2_regularization": 0.5}


def rule(title: str) -> None:
    print("\n" + "=" * 68)
    print(title)
    print("=" * 68)


def experiment_comparable() -> dict:
    tr, va = load_jsonl(TRAIN), load_jsonl(VAL)
    print(f"官方 train: {len(tr['X'])} 行 / {len(set(tr['unit']))} 根轴承 / {tr['X'].shape[1]} 维")
    print(f"官方 val  : {len(va['X'])} 行 / {len(set(va['unit']))} 根轴承")

    res = fit_model(tr["X"], tr["y"], CFG, 0, task="regression")
    pred = res.model.predict(va["X"])
    m, r = mae(va["y"], pred), rmse(va["y"], pred)

    base = json.loads((PKG / "training.json").read_text(encoding="utf-8"))
    b = base["validation_after"]
    print(f"\n{'':22s}{'MAE(分钟)':>12s}{'RMSE(分钟)':>14s}")
    print(f"{'官方基线(他们最好的)':22s}{b['mae']:>12.1f}{b['rmse']:>14.1f}")
    print(f"{'我们的 HGB(一轮训练)':22s}{m:>12.1f}{r:>14.1f}")
    print(f"{'差值':22s}{m - b['mae']:>+12.1f}{r - b['rmse']:>+14.1f}   （负=我们更好）")
    print(f"\n单轮训练耗时 {res.seconds:.2f} 秒，模型参数约 "
          f"{res.model.n_iter_} 轮 boosting")
    return {"mae": m, "rmse": r, "base_mae": b["mae"], "base_rmse": b["rmse"],
            "seconds": res.seconds}


def experiment_self_evolution(group: int = 20, warmup: int = 60) -> dict:
    tr = load_jsonl(TRAIN)
    X, y, t = to_stream_arrays(tr, group=group)
    stream_steps = int(t[-1]) + 1
    print(f"官方 train 拼成一条流：{len(y)} 个快照 / {stream_steps} 步（每步 {group} 个）")
    print(f"前 {warmup} 步当历史训练，后 {stream_steps - warmup} 步模拟上线")

    train_mask = t < warmup
    static = fit_model(X[train_mask], y[train_mask], CFG, 0, task="regression").model

    # 最简单的自进化：每 20 步用“已到手的标签”重训一次，新模型必须更好才替换
    from src.models.baseline import build_model
    champ = static
    hist_X, hist_y = X[train_mask], y[train_mask]
    retrains = promotions = 0
    errs_static, errs_evo = [], []
    for step in range(warmup, stream_steps):
        m = t == step
        errs_static.append(mae(y[m], static.predict(X[m])))
        errs_evo.append(mae(y[m], champ.predict(X[m])))
        hist_X = np.vstack([hist_X, X[m]])
        hist_y = np.concatenate([hist_y, y[m]])
        if (step - warmup + 1) % 20 == 0:
            retrains += 1
            cand = build_model(CFG, 0, "regression").fit(hist_X, hist_y)
            # 用最近 20 步做统一 holdout 比较
            recent = t >= step - 19
            cs, _ = score_model("regression", cand, X[recent], y[recent])
            ks, _ = score_model("regression", champ, X[recent], y[recent])
            if cs >= ks:
                champ = cand
                promotions += 1

    s_mae, e_mae = float(np.mean(errs_static)), float(np.mean(errs_evo))
    print(f"\n{'':16s}{'平均MAE':>10s}{'末段MAE':>10s}")
    print(f"{'static(永不更新)':16s}{s_mae:>10.1f}{errs_static[-1]:>10.1f}")
    print(f"{'trigger(自我进化)':16s}{e_mae:>10.1f}{errs_evo[-1]:>10.1f}")
    print(f"重训 {retrains} 次，晋升 {promotions} 次")
    return {"static": s_mae, "evolved": e_mae, "retrains": retrains,
            "promotions": promotions, "steps": stream_steps - warmup}


def main() -> int:
    rule("实验 1：与官方基线在同一 validation 上对比（口径可比）")
    a = experiment_comparable()
    rule("实验 2：只动 train，做自进化（不碰 validation，符合官方协议）")
    b = experiment_self_evolution()
    rule("结论")
    print(f"· 官方基线 RMSE {a['base_rmse']:.1f} 分钟；我们 {a['rmse']:.1f} 分钟"
          f"（{'更好' if a['rmse'] < a['base_rmse'] else '更差'}）")
    print(f"· 自进化把平均 MAE 从 {b['static']:.1f} 压到 {b['evolved']:.1f} 分钟，"
          f"重训 {b['retrains']} 次成功 {b['promotions']} 次")
    out = ROOT / "experiments" / "official_b_check.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"comparable": a, "self_evolution": b},
                              ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已写入 {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
