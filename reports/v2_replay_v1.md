# v1 试运行报告 → v2 预算规则回放（反事实）

> 数据源：`_state/trial_report_completed.json`（v1 试运行，10 轮 × A/B/C）。
> **这是回放分析，不是新实验**：结果沿用 v1 已观测到的候选分数，只把同一组信号
> 输入 v2 的 `choose_budget`，用来看 v1 的训练预算里有多少是当时就能省掉的。

| 数据集 | v1 epochs | v2 epochs | 省下 | v1 CPU 秒 | 估计省下 CPU 秒 |
|---|---:|---:|---:|---:|---:|
| A | 28 | 28 | 0 | 993 | 0 |
| B | 50 | 17 | 33 | 368 | 243 |
| C | 41 | 18 | 23 | 1868 | 1048 |
| **合计** | **119** | **63** | **56** | **3229** | **1291** |

## 逐轮对照

### A（macro_f1）

| 轮 | 阶段 | v1 epochs | v1 决策 | v2 epochs | v2 规则 | v2 决策 | PSI |
|---:|---|---:|---|---:|---|---|---:|
| 1 | A-stage-01 | 5 | promote | 5 | first_round | promote | 2.024 |
| 2 | A-stage-01 | 1 | promote | 1 | calibration | retain | 2.024 |
| 3 | A-stage-02 | 3 | promote | 3 | new_stage_drift | promote | 0.679 |
| 4 | A-stage-02 | 1 | promote | 1 | calibration | promote | 0.679 |
| 5 | A-stage-03 | 3 | promote | 3 | new_stage_drift | promote | 0.671 |
| 6 | A-stage-04 | 3 | promote | 3 | new_stage_drift | promote | 0.620 |
| 7 | A-stage-05 | 3 | promote | 3 | new_stage_drift | promote | 0.551 |
| 8 | A-stage-06 | 3 | promote | 3 | new_stage_drift | promote | 0.521 |
| 9 | A-stage-07 | 3 | promote | 3 | new_stage_drift | promote | 0.518 |
| 10 | A-stage-08 | 3 | promote | 3 | new_stage_drift | promote | 0.522 |

### B（rmse）

| 轮 | 阶段 | v1 epochs | v1 决策 | v2 epochs | v2 规则 | v2 决策 | PSI |
|---:|---|---:|---|---:|---|---|---:|
| 1 | B-stage-01 | 5 | promote | 5 | first_round | promote | 7.369 |
| 2 | B-stage-01 | 5 | promote | 3 | calibration | promote | 7.369 |
| 3 | B-stage-01 | 5 | retain | 3 | calibration | retain | 7.369 |
| 4 | B-stage-01 | 5 | retain | 0 | quiet | no_train | 7.369 |
| 5 | B-stage-02 | 5 | retain | 3 | new_stage_drift | retain | 7.368 |
| 6 | B-stage-02 | 5 | retain | 0 | stalled | no_train | 7.368 |
| 7 | B-stage-02 | 5 | retain | 0 | stalled | no_train | 7.368 |
| 8 | B-stage-03 | 5 | retain | 3 | new_stage_drift | retain | 6.804 |
| 9 | B-stage-03 | 5 | retain | 0 | stalled | no_train | 6.804 |
| 10 | B-stage-03 | 5 | retain | 0 | stalled | no_train | 6.804 |

### C（rmse）

| 轮 | 阶段 | v1 epochs | v1 决策 | v2 epochs | v2 规则 | v2 决策 | PSI |
|---:|---|---:|---|---:|---|---|---:|
| 1 | C-stage-01 | 5 | promote | 3 | first_round | promote | 0.018 |
| 2 | C-stage-01 | 4 | promote | 3 | calibration | promote | 0.018 |
| 3 | C-stage-01 | 4 | promote | 3 | calibration | promote | 0.018 |
| 4 | C-stage-02 | 4 | retain | 3 | new_stage | retain | 0.017 |
| 5 | C-stage-02 | 4 | retain | 0 | quiet | no_train | 0.017 |
| 6 | C-stage-02 | 4 | retain | 0 | quiet | no_train | 0.017 |
| 7 | C-stage-03 | 4 | retain | 3 | new_stage | retain | 0.017 |
| 8 | C-stage-03 | 4 | retain | 0 | stalled | no_train | 0.017 |
| 9 | C-stage-04 | 4 | retain | 3 | new_stage | retain | 0.017 |
| 10 | C-stage-04 | 4 | retain | 0 | stalled | no_train | 0.017 |

---

## 实测对照（v2 平台试运行，2026-11-23）

回放预测 63 epoch；v2 实测 66 epoch，差 +3（+4.8%）：

| 数据集 | 回放 v2 | 实测 v2 | 差异 |
|---|---:|---:|---:|
| A | 28 | 23 | -5 |
| B | 17 | 25 | +8 |
| C | 18 | 18 | 0 |
| **合计** | **63** | **66** | **+3** |

实测关键指标（v1 → v2）：

- A macro-F1：0.8735 → 0.8674（-0.69%）
- B MAE / RMSE：506.67 / 708.58 → **251.70 / 352.08（约 -50%）**
- C：41.12 / 56.38 / NASA 8.85×10¹⁰（与 v1 相同，v2 仍选第 3 轮模型）
- 墙钟：8,160 → 3,917 s（-52%）；CPU：3,229 → 3,044 s（-5.7%）

结论：回放对“省一半训练”的方向判断准确（63 vs 66）；但它没有预测到 B 的指标减半，
也没有预测到 A 的小幅回落。B 的收益来自“校准 3 epoch + 第 8/9 轮再训”，
不是单纯的“少训”。

