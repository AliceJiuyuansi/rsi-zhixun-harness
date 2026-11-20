"""一步一步看：原始振动 -> 特征 -> 训练集 -> 模型 -> 漂移后的误差。

运行：
    python scripts/inspect_data.py
每一步都会打印"这一步在干什么、数字长什么样"。
"""
from __future__ import annotations

import numpy as np

from common import ROOT

from src.data.xjtu import FEATURE_NAMES, bearing_files, extract_features
from src.models.baseline import fit_model

RAW = ROOT / "XJTU-SY_Bearing_Datasets" / "Data" / "XJTU-SY_Bearing_Datasets" / "XJTU-SY_Bearing_Datasets" / "40Hz10kN" / "Bearing3_4"
NPZ = ROOT / "data" / "processed" / "xjtu" / "40Hz10kN__Bearing3_4.npz"

SHOW = ["h_rms", "v_rms", "h_kurtosis", "h_crest", "h_centroid", "h_band2"]


def rule(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def main() -> int:
    files = bearing_files(RAW)
    print(f"这个轴承一共有 {len(files)} 个快照文件：{files[0].name} ... {files[-1].name}")

    rule("第 1 步：原始数据长什么样（一个 CSV = 1.28 秒的振动）")
    for label, path in (("第一个（全新）", files[0]), ("最后一个（快坏了）", files[-1])):
        arr = np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.float64)
        print(f"\n{label}  {path.name}")
        print(f"  形状: {arr.shape}  = 32768 个采样点 × 2 个通道（水平/垂直）")
        print(f"  前 3 行:\n{np.round(arr[:3], 3)}")
        print(f"  水平通道均值 {arr[:, 0].mean():+.3f}，标准差 {arr[:, 0].std():.3f}")

    rule("第 2 步：把 65536 个数字压缩成 36 个特征")
    f0 = extract_features(files[0])
    f1 = extract_features(files[-1])
    print(f"{'特征':<12}{'全新':>10}{'将坏':>10}   含义")
    meaning = {
        "h_rms": "振动能量大小",
        "v_rms": "振动能量大小",
        "h_kurtosis": "冲击成分多少",
        "h_crest": "尖峰突出程度",
        "h_centroid": "能量集中在多高频率",
        "h_band2": "中高频能量占比",
    }
    for name in SHOW:
        j = FEATURE_NAMES.index(name)
        print(f"{name:<12}{f0[j]:>10.3f}{f1[j]:>10.3f}   {meaning[name]}")

    rule("第 3 步：数据流怎么切（9,216 个快照里取这一个轴承）")
    d = np.load(NPZ, allow_pickle=False)
    X, rul = d["X"], d["rul"]
    print(f"缓存文件: {NPZ.name}")
    print(f"  X 形状 {X.shape}  = {X.shape[0]} 个快照 × {X.shape[1]} 个特征")
    print(f"  标签 rul 共 {len(rul)} 个，从 {rul[0]:.3f} 到 {rul[-1]:.3f}（1=全新，0=失效）")
    print("  实验配置: stride=3（每 3 个取 1 个），group=6（每 6 个算一步）")
    n_step = len(rul[::3]) // 6
    print(f"  -> 最终 {n_step} 步，每步 6 个样本；前 25 步拿来训练，后面 59 步当数据流")

    rule("第 4 步：模型在训练集上学，然后在流上逐段预测")
    Xs, ys = X[::3], rul[::3]
    n = (len(Xs) // 6) * 6
    Xs, ys = Xs[:n], ys[:n]
    steps = np.arange(len(Xs)) // 6
    warm = 25
    train_mask = steps < warm
    cfg = {"kind": "hgb_regressor", "max_iter": 120, "learning_rate": 0.08,
           "max_leaf_nodes": 15, "min_samples_leaf": 20, "l2_regularization": 0.5}
    res = fit_model(Xs[train_mask], ys[train_mask], cfg, 0, task="regression")
    print(f"用前 {warm} 步（{train_mask.sum()} 个样本）训练完成，耗时 {res.seconds:.3f} 秒")

    print(f"\n{'位置':<10}{'真实寿命':>10}{'模型预测':>10}{'误差':>8}")
    last = steps.max()
    for name, st in (("开始", warm), ("1/3", warm + (last - warm) // 3),
                     ("2/3", warm + 2 * (last - warm) // 3), ("结束", last)):
        m = steps == st
        pred = res.model.predict(Xs[m])
        truth = ys[m]
        print(f"{name:<10}{truth.mean():>10.3f}{pred.mean():>10.3f}"
              f"{abs(truth - pred).mean():>8.3f}")

    print("\n结论：模型没变，数据变了 -> 误差从 0.04 涨到 0.8，这就是漂移。")
    print("下一步就是让系统自己发现这件事，并决定要不要重训（见 docs/07）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
