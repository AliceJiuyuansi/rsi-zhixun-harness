# A / B 组优化指导方案（供下游 Agent 执行）

> **定位**：面向下游执行 Agent（Claude Code / Codex / 子 Agent）的端到端工程任务单。  
> **目标**：在**完全冻结 C 组（v14）与 Token 优化（v15）收益**的前提下，通过本地严格消融与抗漂移/跨轴承泛化验证，挖掘 **B 组（轴承剩余寿命）** 与 **A 组（气体识别）** 的确定性提升，为平台**最后 1 次试运行**做好打包准备。

---

## 0. 背景与当前基线

### 0.1 平台历史战报（隐藏测试集）

| 平台版（源码） | A Macro-F1 ↑ | B RMSE（分钟）↓ | C RMSE（循环）↓ | C NASA 惩罚 ↓ | Token |
|---|---:|---:|---:|---:|---:|
| 第 9 版（`0.7.2`） | `0.873469` | **`341.618`** | `54.454` | `7.952×10¹⁰` | 31,924 |
| 第 11 版（`0.9`） | `0.873469` | `357.785` | `55.433` | `1.159×10¹¹` | 47,598 |
| 第 12 版（`0.14`） | `0.873469` | `357.785` | **`53.216`** | **`9.225×10⁹`** | 47,483 |

### 0.2 仓库现状（动手前务必先核对）
- 当前所在分支：`v16`（基于 `v15` `ce61390`），工作区已有未提交的 B 组 `b_drift_recovery` 探索改动（见 `git status` / `git diff`）及 `experiments/v16-b/` 实验目录。
- **平台额度极度稀缺**：试运行已用 **9 / 10 次**（仅剩 **1 次试运行** + 3 次正式评测）。**严禁自行上传镜像或触发平台试运行**，全部工作在本地完成并输出 Go/No-Go 评审报告。

---

## 1. 硬性红线（绝对不可违反）

1. **冻结 C 组**：禁止修改 `vendor/contest_baselines/c_calibration.py` 以及 C 组的训练/校准/门控逻辑。交付前必须验证 C 组在本地 10 轮闭环中的产物与指标**零回归**。
2. **保留 v15 Token 优化**：保留 `COMPACT_MODEL_RELEASES`（`enable_thinking=False`、`max_tokens=256` 及精简 prompt）机制。
3. **版本开关隔离**：所有新增策略必须通过 `RELEASE` 版本号开关（例如 `0.16` / `0.17`）隔离，保证旧版本（`0.14` / `0.15`）行为可复现且所有既有单元测试继续通过。
4. **检查点与模型硬契约**（见 `vendor/contest_baselines/model.py`）：
   - 必须基于父检查点通过 `OfficialRound.continue_round` 续训，保留 `format="contest.cpu-mlp.v1"`；
   - `w1/b1/w2/b2` 形状、dtype（`float64`）、`parameter_count < 100000`、检查点 `< 16 MB` 不可破坏；
   - 若引入轻量后置校准器，必须像 `c_calibration` 一样序列化到 `model.meta` 中，且对无该字段的旧检查点向后兼容。
5. **单条推理无状态一致**：`model.predict(x)` 对单条样本 `x[i:i+1]` 与整批 `x` 的输出必须逐位一致（`max_abs_diff < 1e-12`）。**严禁**使用 batch-level test-time normalization、跨样本平滑或依赖样本顺序的状态。
6. **严禁验证集泄漏**：任何样本权重、类别先验、特征统计或后置校准参数**只能从 `train` 拟合**；`validation` 仅用于评估与门控选择。
7. **尊重现有工作区**：先检查 `git diff` 中 `v16` 已有改动，在此基础上增量开展实验或用消融数据决定取舍，不得粗暴 `git reset --hard`。

---

## 2. 关键代码与数据位置

- 核心入口：`harness/contest_submission.py`（`train()`、`RELEASE`、`task_limit`）
- 预算与门控：`src/harness/round_controller.py`（`choose_budget`、`apply_verdict`、`paired_acceptance`）
- 续训与评估：`src/models/official_round.py`（`OfficialRound.continue_round`、`evaluate`、`grouped_regression_metrics`）
- 模型与优化器：`vendor/contest_baselines/model.py`（`Model.fit`、`Model.predict`、`Model.validate`）
- 数据读取：`vendor/contest_baselines/data.py`（`read_dataset`，已返回 `groups`）
- 本地公开基线包：`D:/北京创新赛/baselines/{A-gas, B-bearing, C-engine}/`
- 既有验证脚本参考：`experiments/v15-sdk/verify.py`、`experiments/v16-b/compare.py`

---

## 3. Workstream 1：B 组（轴承剩余寿命）优化【优先级 P0】

### 3.1 问题诊断
- **问题 1（可追回的 16.2 分钟）**：第 12 版（`0.14`）隐藏 RMSE `357.785`，比第 9 版（`0.7.2`）的 `341.618` 差 16.2 分钟。线上日志显示 `B-stage-01` 首轮因 `paired_selection` 过于保守（`681 → 660.39` vs `0.7.2` 的 `648.97`），且第 3~7 轮连续 5 轮停滞。
- **问题 2（严重轴承样本失衡）**：
  - 公开训练集（7,032 行，9 根轴承）中，`Bearing3_1`（2,538 行）+ `Bearing3_2`（2,496 行）两根超长寿命轴承占了 **71.6%** 的样本；而短寿命轴承 `Bearing1_1/1_2/1_3` 每根仅 123~161 行。
  - 这导致模型严重偏向高 RUL：在公开验证集上，`Bearing1_4`（最大真值 121 分钟）平均高估 **`+166.5` 分钟**（RMSE `242.0`），`Bearing2_4`（最大真值 41 分钟）平均高估 **`+326.3` 分钟**（RMSE `390.8`）。
  - 隐藏测试集仅有 **200 条**，极可能包含短/中寿命轴承。
- **防过拟合警示**：公开验证集（1,679 行）中 **`Bearing3_4` 单根轴承占 90%（1,515 行）**。如果只看 pooled validation RMSE，模型会剧烈过拟合 `Bearing3_4` 而摧毁短寿命轴承（实测纯 Ridge 把 pooled val RMSE 压到 `484`，但留一轴承 LOBO CV 崩到 `1,214`，`Bearing1_4` RMSE 恶化到 `672`）。

### 3.2 分步消融任务（逐项独立验证，再测组合）

1. **B-1：复核 `v16` 现有改动与 `0.7.2` 差距**
   - 核查 `experiments/v16-b/` 及 `round_controller.py` 中 `b_drift_recovery` 与 `paired_selection` 对 B 组首轮及停滞轮次的影响。
   - 明确回答：仅靠策略/门控调整，能否稳定追回 `0.7.2` 的收益，且在多顺序/留一轴承测试中不引入高方差？

2. **B-2：按轴承等权训练（Group-Balanced Sample Weighting）**
   - 在 `vendor/contest_baselines/model.py` 的 `Model.fit` 中增加可选参数 `sample_weight: np.ndarray | None = None`（默认 `None`，保证向后兼容），在反向传播梯度中按样本权重归一化加权。
   - 在 `OfficialRound.continue_round` 中，对 B 组按 `train_groups` 计算权重：
     $$w_i \propto \left(\frac{1}{N_{g(i)}}\right)^\alpha, \quad \frac{1}{N}\sum_i w_i = 1$$
   - 消融 $\alpha \in \{0.0, 0.5, 0.75, 1.0\}$（当无 `groups` 或组数 $< 2$ 时自动退化为均匀权重）。

3. **B-3：稳健回归损失 / 轻度高估抑制（仅在 B-2 有效后叠加测试）**
   - 对比标准 MSE 与 Huber 损失（在归一化目标空间设置 $\delta \in \{1.0, 1.5, 2.0\}$），观察是否进一步降低跨轴承方差。
   - 如尝试后置校准（`b_calibration` 存于 `model.meta`），**严禁**使用高自由度线性拟合（易重蹈 Ridge LOBO 崩溃覆辙），只允许极低自由度、强正则且有父模型兜底门控的校准。

### 3.3 B 组本地验收标准（必须全部满足才算有效）
构建统一的 B 组评测脚本 `experiments/v16-ab/eval_b.py`，报告以下全部指标：
1. **9 折留一轴承交叉验证（LOBO CV on `train.jsonl`）**：
   - `LOBO macro-RMSE`（9 根轴承各自 RMSE 的等权均值）显著下降；
   - 短寿命轴承子集（`Bearing1_*`、`Bearing2_*`）平均 RMSE 显著下降，且 `Bearing3_*` 不出现灾难性恶化。
2. **公开验证集按组指标（`validation.jsonl`）**：
   - `Bearing1_4` RMSE 明显低于当前基线（`< 299.5`，力争 `< 240`）；
   - `Bearing2_4` RMSE 明显低于当前基线（`< 370.8`，力争 `< 300`）；
   - `Bearing3_4` 与整体 pooled RMSE 不出现大幅回退（pooled RMSE 保持 `≤ 660`）。
3. **预算与稳定性**：10 轮离线 SDK 闭环总 epochs `≤ 25`，无剧烈震荡或异常大步回退。

---

## 4. Workstream 2：A 组（气体识别）优化【优先级 P1】

### 4.1 问题诊断
- **域偏移鸿沟**：第 12 版 A 组在公开验证集末轮 Macro-F1 达 `0.9721`（第 9 轮）/ `0.9697`（第 10 轮），但隐藏测试集（4,886 条，未来传感器漂移批次）始终停在 **`0.873469`**（v1~v14 无一突破），存在约 **10 个百分点**的跨批次泛化缺口。
- **少数类坍塌**：本地公开集 `train`（445 行）→ `val`（1,244 行）已存在显著漂移（`PSI=2.228`）；`Acetaldehyde` 训练仅 30 行（最少类），在父模型上验证集 **F1 = `0.0000`**（109 条全漏，大量误判为 `Ethanol`）。
- **单纯多训无效**：在本地把 `OfficialRound` 从 5 epoch 拉到 30 epoch，验证 Macro-F1 仅从 `0.7258` 移到 `0.7264`；且历史 v2 表明**减少新阶段 epoch（3→2）会导致隐藏分从 `0.8735` 跌到 `0.8674`**，因此不能靠简单削减预算优化 A 组。
- **第 10 轮微幅回退**：线上报告中第 10 轮验证集从 `0.9721` 跌到 `0.9697` 仍被晋升，需检查是否因晋升容忍度过宽导致末轮噪声覆盖更优父模型。

### 4.2 分步消融任务

1. **A-1：类别均衡加权 + 温和正则（L2 / Label Smoothing）**
   - 复用 `Model.fit` 的 `sample_weight` 或新增分类 `class_weight`：对类别 $c$ 赋予权重 $w_c \propto (N / N_c)^\beta$，消融 $\beta \in \{0.0, 0.3, 0.5, 0.75\}$。
   - 联合测试温和 L2 权重衰减（`weight_decay` $\in \{0, 10^{-4}, 5\times 10^{-4}, 10^{-3}\}$）与 Label Smoothing（$\epsilon \in \{0.0, 0.05, 0.1\}$），缓解小样本（445 行）对源批次特征的过拟合。

2. **A-2：抗漂移特征鲁棒化 / 轻量残差 Logit 校准（`a_calibration`）**
   - 分析 128 维气体传感器特征在 `train` vs `val` 上的漂移模式（哪些维度是尺度/基线漂移，哪些相对比值或逐样本归一化特征更稳定）。
   - 尝试**单样本无状态**的轻量校准（如存于 `model.meta["a_calibration"]`）：
     - 选项 i：对输入特征做逐样本 L2 归一化或分传感器块归一化后的残差线性分类头（与原 MLP logits 融合，系数由训练集交叉验证确定，强 L2 正则）；
     - 选项 ii：Logit 温度缩放 + 类别先验/边际校准（仅基于训练集分布）。
   - 必须自带验证集门控：仅当校准后 `macro_f1` 严格提升且不破坏 `accuracy` 时才启用，否则回退纯父/子 MLP。

3. **A-3：A 组晋升门控复核**
   - 检查 `contest_submission.py` / `round_controller.py` 中 A 组末轮回退仍被晋升的原因；评估收紧 A 组晋升门槛（例如拒绝负增益晋升）对多阶段 10 轮闭环的影响。

### 4.3 A 组本地验收标准（必须包含抗漂移压力测试）
构建统一的 A 组评测脚本 `experiments/v16-ab/eval_a.py`，报告以下全部指标：
1. **公开跨批次验证（`train → val`，天然 `PSI=2.228`）**：
   - `macro_f1` 显著高于当前基线（`0.7258`），且 `accuracy` 不明显下降；
   - `Acetaldehyde` 等弱势类别 F1 显著脱离 `0.0000`。
2. **合成漂移压力测试（模拟隐藏测试集未来漂移）**：
   - 在 `validation.jsonl` 上构造 3 组确定性漂移基准（固定随机种子）：
     - 轻度乘性+加性基线漂移（±5% 尺度 + 0.1σ 偏移）；
     - 中度传感器漂移（±12% 尺度 + 0.25σ 偏移）；
     - 特征通道子集衰减（模拟部分传感器老化）。
   - 新方案在 3 组漂移测试上的平均 `macro_f1` 必须显著高于 `v14/v15` 基线，且在无漂移原验证集上不退化。
3. **离线 SDK 10 轮闭环**：总 epochs `≤ 28`，单条 vs 批量 `predict()` 差异 `< 1e-12`。

---

## 5. 执行流程与交付清单

下游 Agent 请按以下顺序推进，**每一步均需保留可复现脚本与 JSON 日志**（统一存入 `experiments/v16-ab/`）：

1. **阶段 1 · 基线对齐**：运行 `pytest` 确认当前 79+ 单测状态；读取 `git diff` 摸清 `v16` 已有改动；跑出 A、B 当前基线的 `eval_a.py` / `eval_b.py` 基准数字。
2. **阶段 2 · B 组消融与定型**：依次完成 `B-1 → B-2 → B-3`，输出对比表；只保留在 **LOBO CV + 短寿命轴承** 上均稳定胜出的改动。
3. **阶段 3 · A 组消融与定型**：依次完成 `A-1 → A-2 → A-3`，输出对比表；只保留在 **跨批次 val + 合成漂移压力测试** 上均稳定胜出的改动。
4. **阶段 4 · 集成与全量回归验证**：
   - 补齐新增逻辑的单元测试（放 `tests/`），运行全量 `pytest` 通过；
   - 运行单条 vs 批量预测一致性检查（A / B / C 全覆盖）；
   - 参考 `experiments/v15-sdk/verify.py`，用正式 SDK 1.0.4 跑通 A/B/C 各 10 轮离线闭环：
     - 验证 **C 组检查点 SHA-256 与指标与 v14/v15 完全一致（零回归）**；
     - 记录 A / B / C 的耗时、CPU、总 epochs 与最终验证指标。
5. **阶段 5 · 文档与决策建议**：
   - 更新 `docs/status.md` 顶部记录本次实验结果；
   - 输出最终汇报，包含：**改动文件清单、A/B 消融对比表、C 组零回归证明、以及是否建议动用最后 1 次试运行的 Go/No-Go 结论**。
   - **停止在本地验证完成处，等待人工确认后再决定是否构建镜像和提交平台。**
