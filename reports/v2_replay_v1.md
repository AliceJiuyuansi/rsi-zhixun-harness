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

