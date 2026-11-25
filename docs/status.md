# 项目进度快照

<!-- latest-trials -->
截至 2026-11-25 官网共显示 9 次试运行。平台第 12 版（源码 v14）是最新已测版本，A 并列最好、C 的 RMSE 和 NASA 惩罚均为最低。B 最低 RMSE 来自平台第 9 版（源码 0.7.2），官网当前选中该版。 [官网运行列表](https://rsi-index.115.zone/#runs)
<!-- /latest-trials -->

> 每次会话/每周例会结束更新。最新的在最上面。
> 格式固定，方便 agent 与人快速接手。

## 2026-11-25 · v17 客观分 35 分专项优化（`RSI_RELEASE=0.17`）

- **目标与改动**：针对平台第 13 版（`036e178`，客观分 18.50/35，其中 `resource_efficiency=0.00/5`、`mean_gain=7.64/15`、`worst_stage=1.64/5`），在分支 `v17` 合入五项面向 35 分客观评测的升级：
  1. **资源效率（Token + CPU 双压减）**：默认 `RSI_RELEASE=0.17`，每轮仍执行 1 次真实 SDK 模型调用，显式设置 `enable_thinking=False`、`max_tokens=64` 与超紧凑提示（约 216 字符），预计 30 轮总 Token 从 47,617 压至约 3,000（硬上限 ~4,400）；C 校准改为 `fast_c_calibration=True`（仅对最优 raw-RMSE epoch 拟合 1 次 `res_mlp_v2`，单轮速度提升约 2.5×）；A 组停滞静默轮跳过多余训练（`quiet_stall_a=True`）。
  2. **B 组轴承跨阶段校准（`vendor/contest_baselines/b_calibration.py`）**：纯训练侧、确定性、逐样本符号一致 `log1p(RUL)` 校准器（`sign_log_v1`），按轴承等权（`bw=1.0`）并用 `alpha=0.80` 与续训 MLP 在 `log1p` 空间几何融合，同时保留 `v16` 的 `b_drift_recovery`。
  3. **C 组后期预算预留 + NASA 门禁放宽（`src/harness/late_budget.py`、`src/harness/round_controller.py`）**：预留 3 epoch 至第 8–10 轮（`limit=12, reserve_epochs=3, release_round=8, idle_threshold=3`），并设置 `c_nasa_tol=0.0015`（允许 `nasa_log1p <= -0.10` 且 RMSE 波动 `>= -0.15%` 晋升），修复第 13 版 Stage 4 R09 拒收问题。
  4. **安全与零回归**：全部校准纯训练侧拟合、不读写验证集标签；`checkpoint.npz` 保持 15 个数组、`<= 16 MiB`，异常或缺失字段自动回退基础 MLP；成对父子零退化门禁保持不变。
  5. **发布一致性防混包门禁（借鉴 `origin/v16` 并适配 `v17`）**：新增 `release_manifest.json`、`scripts/check_release.py`（支持跨平台行尾归一化 SHA-256、同时校验 `b_calibration` + `c_calibration` + `v17` 开关与 `OfficialRound` 签名）、`scripts/build_source_bundle.py` 及 `tests/test_release_bundle.py`；更新 `.dockerignore` 放行校验脚本与清单，并在 `harness.Dockerfile` 中内置构建期 `--layout image --require-sdk` 预检。
- **本地验证结果**：
  - 全量单测 `python -m unittest discover -s tests`：**100 项全部通过**（含 8 项 `tests/test_v17_upgrades.py` 与 7 项 `tests/test_release_bundle.py`），`python scripts/check_release.py --root .` 返回 `ok: true`。
  - SDK 1.0.4 CLI 验证：`harness.py describe` 退出 0，`harness.py predict` 在 A/B/C 三个检查点逐条与批量预测完全一致（最大差 `< 1e-12`）。
  - 官方公开包 10 轮端到端对比（`0.15` → `0.17`）：
    - **A-gas**：`macro_f1=0.7187`、`acc=0.8859`（一致），训练轮次 `14 → 8` epoch，提示 `731 → 216` 字符，`max_tokens 256 → 64`。
    - **B-bearing**：验证 RMSE **`654.97 → 613.36`**（相对父模型 `681.16` 收益 `+3.8% → +9.95%`），MAE `546.08 → 480.55`，三根验证轴承全面改善（`Bearing1_4: 283.8→252.8`、`Bearing2_4: 375.7→201.3`、`Bearing3_4: 681.9→640.8`）；9 折 LOBO Macro-RMSE `716.3 → ~426`（-40%），三阶段渐进回放 Stage 1/2/3 收益 `+13.3% / +11.4% / +9.4%`。
    - **C-engine**：验证 NASA **`8.030×10⁸ → 7.674×10⁸`**（`nasa_log1p 20.5039 → 20.4586`），RMSE `46.6749`，已用 epoch `12 → 8`（成功为 R8–R10 留出 3 epoch），总耗时 `77.60s → 63.13s`。
- **镜像与评测状态**：本机 Windows 无 Docker CLI，尚未构建 `zhixun-rsi-harness:0.17` 镜像或上传平台评测。

## 2026-11-25 · A/B 抗漂移验收，保留最后一次试运行

- **No-Go。** B 的 α=1 加权把九折 LOBO Macro-RMSE 从 691.99 降至 674.02 分钟，短寿命均值从 540.53 降至 513.56 分钟，但 `Bearing2_4` 在三种顺序中两次超过 370.8。A 的 48 组训练组合和训练侧校准对照均未改善 Acetaldehyde 的 F1。两组旧门控复核后均保留现有上线策略。
- **B 漂移探针保留在 `RSI_RELEASE=0.16`。** 等量切窗留出 RMSE 815.568 → 648.101，同轴承留出 757 行，总请求 10 → 16 epoch。标准十轮与 `0.15` 的检查点及 RMSE 一致。默认源码与 Docker 构建参数回到 `0.15`。
- **C 十轮零回归。** 源码 `0.14`、`0.15`、`0.16` 在相同输入和空回执下各跑完十轮，所有轮次选中检查点与晋升轨迹一致，最终 SHA-256 为 `809b9f19d016e8098710b243f27557252863c1f4ef1d96f1cefa23c0359764e8`，RMSE 47.664854、NASA 2.002412×10⁹。A/B/C 请求 epoch 28/25/13，单条与批量最大差均小于 `1e-12`。85 项测试及 56 个子测试通过。
- 本地脚本、日志、消融表和复现步骤见 [`experiments/v16-ab/结论.md`](../experiments/v16-ab/结论.md)。本轮停在本地验证，未构建镜像、上传或触发平台评测。

## 2026-11-25 · v16 B 组停滞漂移恢复

- 基于 `ce61390` 的模型调用优化，在 B 组新增同阶段恢复探针。连续更新无收益后，同时观察到输入漂移和 RMSE 上升才恢复训练，仍由预算限制和同轮父子验证控制晋升。
- 公开轴承工况 1 → 2 → 3 的等量换窗回放中，第 5 轮由停训变为探针并晋升。757 条同轴承独立留出行的 RMSE 从 815.568 降至 648.101 分钟，总请求从 10 增至 16 epoch。稳定工况与单次切换对照结果一致。81 项单测通过，A/C 本地 SDK 各三轮配对结果一致。尚无平台试运行。 [改进方案与验证](版本演进记录.md#源码-016--b-组停滞后响应同阶段漂移)

## 2026-11-25 · 官网复核与 v15 模型调用优化

| 平台版 | A macro-F1 ↑ | B RMSE（分钟）↓ | C RMSE（循环）↓ | C NASA 惩罚 ↓ | Token |
|---|---:|---:|---:|---:|---:|
| 1 | 0.873469 | 708.575 | 56.382 | 8.853×10¹⁰ | 22,366 |
| 5 | 0.867426 | 352.083 | 54.961 | 6.741×10¹⁰ | 32,055 |
| 9（0.7.2） | 0.873469 | 341.618 | 54.454 | 7.952×10¹⁰ | 31,924 |
| 11（0.9） | 0.873469 | 357.785 | 55.433 | 1.159×10¹¹ | 47,598 |
| 12（0.14） | 0.873469 | 357.785 | 53.216 | 9.225×10⁹ | 47,483 |

- 最早和最新试运行 ID 分别为 `d4c5b175040579c02de129da134263ed`、`6d4cf213aa55efd4650a75fea003f35d`。官网报告显示两次各完成 A/B/C 十轮，隐藏预测量分别为 4,886 / 200 / 9,687 条。第 12 版相对第 11 版的 C RMSE 降 4.0%，NASA 惩罚降 92.0%，C CPU 增 31.6%。[官网运行列表](https://rsi-index.115.zone/#runs)
- 第 12 版 30 次模型回执全为 `length` 和空文本，输出计费合计 23,040 Token，实际训练选择全部来自规则。基于 `61aec2b` 的 v15 保留每轮一次模型调用，在 SDK 请求中设置 `enable_thinking=False`、最多 256 输出 Token，并删除提示里的重复证据和无须返回的理由字段。30 轮的输出预算上限为 7,680 Token，实际用量和模型裁决效果待平台验证。
- v15 本地 79 项单测通过。使用正式 SDK 1.0.4 和本地替身宿主，v14 与 v15 均完成 A/B/C 各十轮。同一空回执条件下，各轮训练与晋升轨迹、最终检查点 SHA-256 和预测输出一致。另用有效 JSON 回执验证了 B 组可下调预算。v15 尚无线上试运行结果。
- A 公开末轮 macro-F1 为 0.969707，隐藏为 0.873469。历史试运行未出现更高隐藏分。B 从第 12 版恢复到第 9 版水平，对应 RMSE 降 4.5%、MAE 降 5.2%，样本规模为 200 条，幅度需要复跑确认。 [官网运行列表](https://rsi-index.115.zone/#runs)

## 2026-11-24：v14 融合 `v9.2` 与 `v13` 优点（`res_mlp_v2` + 分组观测 + 完整审计）

- **融合目标（分支 `v14`，发布默认 `RELEASE=0.14`）**：以通过全量单测与线上复盘修复的 `v9.2` 为完整底座，吸收 `v13` 的有效创新并修补 `v13` 的边界瑕疵与工程漏项：
  1. `vendor/contest_baselines/c_calibration.py`：新增 `res_mlp_v2` 模式，在 `v9.2` 双向残差 MLP（`[-120, +42]`、`over_boost=1.15`、`_best_shift_asym`）基础上，融合 `v13` 的**分工况标准化 `parent_moments`**（全局 + 分工况双尺度 `p_norm`）与**端点线性修正 `_fit_endpoint`**（收紧至 `rmse_tol=0.002`、`grid=[-6..8]`），并修复 `v13` 两处边界问题：`_fit_endpoint` 评估候选时保持 `np.maximum(0.0, ...)` 非负一致、推理时 `endpoint` 仅作用于已知工况（`known`）以保证未知工况完全回退父模型；保留 `v8`/`spline_v2`/`res_mlp_v1` 向后兼容。
  2. `vendor/contest_baselines/data.py` & `harness/contest_submission.py`：合入 `v13` 的 C-MAPSS `FD001..FD004` 分组提取（`meta.source_subset` / ID 正则）并在 `rnd.dataset_key in ("B", "C")` 传入观测性 `validation_groups`；补齐 `selection_audit` 对 `0.9.1`/`0.9.2`/`0.14` 的输出；同步保留 `v9.2` 的 3-epoch 新阶段预算修复、C 组 `nasa_bonus` 门控与中间 epoch 校准节流。
  3. `harness.Dockerfile`：默认 `ARG RSI_RELEASE=0.14`，与代码默认严格对齐。
- **本地验证**：
  - 77 项单元测试与 5 核心模块 `py_compile` 全通过；
  - 同父检查点公开 C 组消融（113,089 训练行 / 496 台发动机，23,049 验证行）：VAL RMSE `47.010`、NASA `1.027e9`（`logN=20.750`，较 `v9.2` 的 `1.277e9` 降 **19.6%**，较 `v13` 完整激活的 `1.589e9` 降 **35.4%**，较 `v8/v9` 的 `3.857e9` 降 **73.4%**）；3 折按发动机 OOF RMSE `52.326`、`logN=28.650`（较 `v9.2` 再降 11.2%，较 `v13` 降 61.6%）；
  - 真实公开 A/B/C 3 轮闭环：A/B 零回归（检查点 SHA256 与指标完全一致）；C 组 3 轮后 VAL RMSE `51.550 → 46.919`、MAE `36.820 → 33.591`、NASA `5.040e9 → 9.830e8`（`logN 22.341 → 20.706`，**-80.5%**；较 `v9.2` 3 轮后的 `1.309e9` 再降 **24.9%**），3 轮总耗时 `7.71s`，批量 vs 逐行最大差 `5.68e-14`。

## 2026-11-24：v9.2 基于网页评测报告的 C 组全面修正（`res_mlp_v1` + 控制器 + CPU 节流）

- **线上复盘根因（对比 8 次平台试运行，重点对照 Run 0 `0.9` / Run 1 `0.7.2` / Run 3）**：
  1. `0.9` 在 C 隐藏集 NASA 反升至 `115.91B`（`0.7.2` 为 `79.52B`，Run 3 为 `67.41B`），主因是 `v8/v9` 校准器给 `target < 0` 加权 `1.35` 并叠加 `[-3.5, +0.5]` 负向平移，而真实 C 组 NASA 97%+ 由极端**低估尾部**主导，单向压低适得其反；
  2. `0.9` 的 `c_stage_probe` 将 `C-stage-02/03/04`（R4/R7/R9）新阶段预算砍成 `1 epoch`，导致 R9 虽已实现 RMSE（`50.468→50.443`）与 `nasa_log1p`（`23.191→23.066`）双降仍因未过门槛被拒，全程停留在 R3 旧检查点；
  3. `official_round.py` 每 epoch 重复全量拟合校准器，使 `0.9` 线上校准 CPU 达训练本身的 ~6.5 倍。
- **本轮修改方向（分支 `v9.2`，发布默认 `RELEASE=0.9.2`）**：
  1. `vendor/contest_baselines/c_calibration.py`：保留 `res_mlp_v1` 双向工况残差 MLP（`[-120, +42]`）与 `spline_v2` 备选，按 3 折 OOF 消融将 `over_boost` 从 `1.65` 回调至 `1.15`（避免隐藏集低估雪崩，同时保留温和防高估），并将 `_best_shift_asym` 搜索网格从 49 点精简至 25 点；
  2. `src/harness/round_controller.py`：`NEW_STAGE_EPOCHS["C"]` 与 `c_stage_probe` 恢复至 `3 epochs`（对齐 `0.7.2` 在 R4/R9 晋升的表现）；C 组 `nasa_bonus` 放宽为 `gain >= 0.0 and delta <= -0.08`（RMSE 不退步且 NASA 降幅 ≥7.7% 即允许晋升）；
  3. `src/models/official_round.py`：非 `v8` 校准模式在多 epoch 续训中跳过未校准验证 RMSE 明显退化（`> 1.005×best`）的中间 epoch 重拟合，降低线上 CPU 开销，同时保持 `v8` 路径字节级不变。
- **本地验证**：76 项单元测试与 4 模块 `py_compile` 全通过；真实公开 C 组 3 轮闭环（`verify_upgraded_repo.py`）：VAL RMSE `50.325 → 46.976`（-6.7%）、MAE `36.302 → 33.653`（-7.3%）、NASA `2.806e9 → 1.267e9`（**-54.8%**，`logN 21.755 → 20.960`）；3 折按发动机 OOF RMSE `54.087 → 52.080`、`logN 31.009 → 28.863`（**-88.3%**）；A/B 零回归（检查点 SHA256 与指标完全一致）；批量 vs 逐行最大差 `8.5e-14`。尚待平台试运行验收隐藏集表现。

## 0.9 本地候选：三题同轮约束选模

- 保留 v8 的优化器、请求预算和输出策略；A 允许持平，B/C 先检查同轮可晋升性再选快照。增加 CPU 分项计量、父子指标、C 尾部复核和无需平台的配对验收工具。证据和使用方法见[候选说明](../reports/版本方案/同轮约束选模.md)。
- 仅完成本地机制验证；没有运行官方公开包完整轨迹或隐藏试运行，不据此修改正式推荐。

## v8 源码候选：预算约束下的更新与选模

- 基于 `16f1620` 实现回执解析、C 分工况风险偏移、B/C 晋升门控和可选权重平滑。约束和冲突取舍见 [v8 版本说明](../reports/版本方案/v8.md)。
- 本地 Python 3.12 + NumPy：原 44 项与新增 16 项测试全部通过，6 个指定模块通过 `py_compile`。未运行官方 SDK、构建赛事镜像或消耗平台次数；尚无平台成绩，不替代 v5。
- 下一步：固定公开阶段做单模块消融，分别核对 A/B/C、CPU、token 与真实模型回执。

## 2026-11-24：v7.1 本地发布安全修复

- 采用新源码包基线，未合入旧cap130/绝对NASA门禁。A沿用v7预算与容差；B/C无固定上限，C工况校准默认启用。
- 修复v7依赖环境开关、Docker却默认0.6的陷阱。预测不随RSI_RELEASE变化；显式校准开关0可关闭新拟合。
- 模型失败/空响应/坏结构降级规则，记录回执状态、finish_reason、字段名和错误类型，不记录异常消息、不伪造成功调用。
- schema3持久化真实行数历史；新增NASA稳定对数求和与B逐轴承观测，不改变选模。早停报告区分保存快照和执行消耗。
- 44项单测、8模块py_compile通过。真实公开A/B/C各3轮新旧对照：检查点/批量预测哈希、晋升序列一致；逐条与批量最大差<1e-12。此为本地替身宿主，不是SDK验收。
- 旧check_guard未通过：输入包缺src.data，F6/F7/healthy报ModuleNotFoundError。没有补造加载器或改断言；.gitignore仅忽略根data，避免忽略src/data，npz权重仍忽略。
- 本机无SDK和Docker daemon；没有远端、平台或SwanLab云端写入。见 `reports/v7.1_交付_2026-11-24.md` 和 `reports/数学分析更新_2026-11-24.md`。

---

## 2026-11-23：P0 源码交付与数学不可行性诊断（并行分支）

- 本地 `fix/cap-nasa-gate`：C24 特征回归按请求 `min(0.9*max(value,0),130)`；服务输出 NASA>1e6 或 >父分 1.2 倍拒绝晋升，另报原始 NASA；schema3 增加 rows/rows_history/data_grew 并兼容 n_train。
- 9 项 LLM 兼容、5 项预测入口、33 项控制器/安全测试全通过，入口 `py_compile` 退出 0；没有新增依赖。真实续训合成接线验证父模型保留、审计、save_state 和 finish，不冒充 SDK 全流程。
- 附件 3 显示 v3 已完成：C MAE 43.787114 / RMSE 62.881078 / NASA 2.440040142e12；A 仍约 0.8674。仓库下面“v3 正在跑”为历史状态。
- 公开 C 基线验证 23,049 行、最大真值 434、9,163 行 >130；原始 NASA 5.8234e9，cap145 后 7.3670e10，指定新策略后 2.3263e11。cap130 与 NASA 1e6 门禁不可行，候选将被冻结；**没有达到平台验收，不建议直接消耗试运行**。
- 附件 2/3 模型裁决各 30/30 unavailable，有效 LLM 覆盖率 0；不能把 A 从 0.8735 降到 0.8674 归因于 LLM。保持 A 代码与 P1 策略不变，后续先做固定预算消融。
- 附件 2 配置的 0 epoch 实为 B 第 4/6/7 轮、C 第 5/6/8/10 轮，共 7/30；“未训练/保留父模型”的页面文案还包含已训练但未晋升情形。
- 阻塞：历史镜像 OCI digest/检查点哈希不可得；新检出缺 SDK，Docker daemon 未运行，尚未做 SDK/容器验收。未推送 PR、上传、试运行或补录 SwanLab。
- 详细交付、风险回退与 PR 描述草案：`reports/P0_交付_2026-11-23.md`；数学专项：`reports/数学建模与优化报告.md`。
  **这两个文件在并行分支 `fix/cap-nasa-gate`，未合并到本仓库**，引用时注意。

---

## 2026-11-24（00:40）：v4/v5 本地审计完成，平台版本已建、未试运行

### 结论

- **A/B 冻结**：A 的 -0.69% 暂按可接受处理，不做 A 预算消融；B 的 251.70 不动。
- **v4**（无上限 + 三项工程修复）与 **v5**（v4 + C 工况校准）都已完成本地全链路验证，
  平台版本 4/5 均 `valid`；**v2 仍是正式候选**。
- 没有消耗新的试运行：trial 已用 3/10，正式评测 0/3。

### C 公开验证（23,049 条，同阶段第 3 轮模型）

| 方案 | MAE | RMSE | NASA |
|---|---:|---:|---:|
| v4 无上限 | 37.856 | 51.691 | 4.321×10⁹ |
| **v5 校准 ±10** | **36.536** | **50.049** | **2.378×10⁹** |
| 变化 | -3.5% | **-3.18%** | **-44.97%** |

- 四个 FD 子集 RMSE 均下降；NASA 在 FD001/FD004 下降、FD002/FD003 略升；
- 训练/验证发动机无交集；校准只用公开训练集、推理只用 NumPy；
- 全量 113,089 行、3 轮 SDK、23,049 条逐条 `predict()` 已验证；隐藏收益未验证。

### 工程修复

| 问题 | 修复 | 验证 |
|---|---|---|
| 特征路径重复编码 | `lru_cache(2048)` + tuple path | 六个文件读取 CPU 23.13→7.56 s（-67.3%），本机复测 23.54→7.94（-66.3%） |
| 历史增益写反 | 用每轮 `configuration.gain` | v3 C 第 1 轮由 -10.701% 修正为 +0.9805%；60 条记录核对 |
| 连续失败不熔断 | 恢复 `kill_state`，成功/失败计数 | 3 次失败后第 4/5 轮零预算；单测 30 项 |

### 平台版本

| 版本 | 镜像 | submission id | 状态 |
|---|---|---|---|
| v4 | `zhixun-rsi-harness:0.4`，archive `47311235…` | `d4ba36296650c173917634d0ab38f105` | valid |
| v5 | `zhixun-rsi-harness:0.5`，archive `938ccaab…` | `5a45bc60cf3f4ea34b35a3fa1f99bd4d` | valid |
| v2 | 正式候选 | `8636a3dcbc…` | valid，已有平台成绩 |

- 容器：`0.4` A 2 轮、`0.5` C 2 轮断网彩排 exit 0；`describe` 均通过；
- 单测 30 项、`check_guard.py` 9/9 PASS；
- 平台文档 `系统说明.md` 已更新到 v4/v5 状态。

### 下一步

1. **先试运行 v4**（1 次）：验证撤销 145 后 C 回到 v2 水平，A/B 不变；
2. **再试运行 v5**（1 次）：只验证 C 校准，比较 MAE/RMSE/NASA/CPU；
3. 若 v5 平台退化，保留 v4/v2；若改善，再把 v5 选为正式候选；
4. 剩余 7 次试运行留给重复/最终确认，正式评测 3 次不动。

---

## 2026-11-23（23:40）：v3 完成，C 的 145 上限被实测证伪

### 平台结果（run `27143ad15abd6f1a4daf998200625545`，submission v3，镜像 0.3）

| 数据集 | v2 | v3 | 变化 |
|---|---:|---:|---:|
| A macro-F1 | 0.8674 | 0.8674 | 0（A/B 与 v2 逐位相同） |
| B MAE / RMSE | 251.70 / 352.08 | 251.70 / 352.08 | 0 |
| C MAE（循环） | 41.12 | **43.79** | +6.5% |
| C RMSE | 56.38 | **62.88** | +11.5% |
| C NASA | 8.85×10¹⁰ | **2.44×10¹²** | **27.6×** |
| 总 epoch | 66 | 66 | 0 |
| 墙钟 / CPU | 3,917 s / 3,044 s | 3,730 s / 3,075 s | -4.8% / +1.0% |

### 结论

- v3 只改了 C 的 `min(value,145)`，A/B 逐位相同，所以 C 的变差可以归因到上限；
- C 的 RUL 真值最大 **542**，公开验证集 **33%** 的样本 >145；上限把真实高 RUL 压成低估，
  NASA 对低估也是 `exp(-d/13)-1`，所以从 8.85×10¹⁰ 变成 2.44×10¹²；
- **145 这个参数被实测证伪，不是“再调大一点”的问题**：v4 直接移除上限；
- 资源没有明显收益（epoch 相同，CPU +1%），说明上限只改变预测，不改变训练。

### 下一步（冻结 A/B，C 先无上限、再校准）

1. **v4：只移除 C 的 145 上限 + 三项工程修复**，A/B 预算保持 v2；
   跑 1 次试运行，预期 A 0.8674、B 251.70、C 回到 41.12/8.85×10¹⁰。
2. **v5：在 v4 上只加 C 工况校准**（±10），A/B 不动；
   跑 1 次，比较 C 的 MAE/RMSE/NASA 和 CPU；隐藏收益未验证，不得直接替换 v2。
3. **A/B 冻结**：不再做 A 预算消融，不用平台次数追 0.006；B 的 251.70 不动。
4. v2 继续作为已有平台成绩的正式候选；v4/v5 只有平台验证后才替换。
5. 剩余 7 次试运行留给重复/最终确认；正式评测 3 次不动。

### 新增事实

- **Q23**：C 的 RUL 真值最大 542（公开训练集），145 上限会切断 33% 的真实高 RUL；
  任何“尾部上限”必须 ≥ 训练集 99.5 分位（419）或直接去掉。

---

## 2026-11-23（22:30）：v2 试运行完成，B 指标减半、资源减半

### 平台结果（run `2b638d5aa090722ebc32e1c3f577e000`，submission v2）

| 数据集 | v1 | v2 | 变化 |
|---|---:|---:|---:|
| A macro-F1 | 0.8735 | 0.8674 | **-0.0060（-0.69%）** |
| A accuracy | 0.8795 | 0.8729 | -0.0065 |
| B MAE（分钟） | 506.67 | **251.70** | **-50.3%** |
| B RMSE | 708.58 | **352.08** | **-50.3%** |
| C MAE / RMSE | 41.12 / 56.38 | 41.12 / 56.38 | 0（同一模型） |
| C NASA | 8.85×10¹⁰ | 8.85×10¹⁰ | 0（0.2 镜像无上限） |
| 总 epoch | 119 | **66** | -44.5% |
| CPU 秒 | 3,229 | 3,044 | -5.7% |
| 墙钟秒 | 8,160 | 3,917 | **-52.0%** |
| training_seconds | 4,471 | 2,016 | **-54.9%** |
| total tokens | 22,366 | 32,087 | +43% |

### 完成

- v2 三题各 10 轮全部完成，`training_complete=true`；
- **B 收益最大**：MAE/RMSE 减半；v2 在 B 第 2/3/8/9 轮拿到正收益，最终选中第 9 轮；
- **A 小幅回落**：总 epoch 28→23，部分新阶段只给 2 epoch；A 的新阶段预算需要调回 3；
- **C 完全没变**：v2 最终仍选第 3 轮模型，停手只省了 23 epoch，没改变产物；
- 停手规则生效：B 第 4/6/7 轮、C 第 5/6/8/10 轮 `epochs=0`（共 7/30 轮）；
- 回放预测省 56 epoch，实际省 53 epoch，基本吻合；
- **模型裁决仍全是 `unavailable`**：`content`/`reasoning_content` 都没有文本，reason 如实写
  “无文本，按规则执行”；没有伪造模型判断，但“自主决策”的展示素材仍不足。

### 进行中

- **v3 试运行**（run `27143ad15abd6f1a4daf998200625545`，submission v3，镜像 0.3）
  已启动，A/B 完成、C 训练中；它带 C 题 145 上限，重点看 NASA 是否下降。

### 下一步

1. 等 v3 报告：重点看 C NASA、B 是否守住 251.70、A 是否回到 0.87 以上；
2. 若 A 仍低于 v1，把 A 的新阶段预算从 2 调回 3（只动 A）；
3. 用 v2 的“B 减半 + 资源减半”做技术报告/视频主线，C 等 v3；
4. 正式评测前再确认版本与记录链接的对应关系。

### 决策日志

| 日期 | 决策 | 理由 | 证据 |
|---|---|---|---|
| 2026-11-23 | 保留 v2 的停手规则，不回退到“每轮都训” | v2 实测省 53 epoch、B MAE 减半；C 没变但也没变差 | `reports/trial_runs.md`、v2 report |
| 2026-11-23 | A 的新阶段预算待 v3 后再调，不立刻改 | v3 已经在跑，先看同一套预算在 0.3 上的 A 结果，避免同时改两个变量 | v3 run `27143ad1…` |

---

## 2026-11-23：v1 试运行复盘 + v2 控制器与第二次提交准备

### 完成

- **读回 v1 试运行报告**（`_state/trial_report_completed.json`）：A/B/C 各 10 轮全部 `finish`
  - A：macro-F1 **0.8735** / accuracy 0.8795，CPU 993 s，wall 2662 s
  - B：MAE **506.67** 分钟 / RMSE 708.58，CPU 368 s，wall 1968 s
  - C：MAE **41.12** 循环 / RMSE 56.38，CPU 1868 s，wall 3530 s
  - 三个待查点：`/data` 按阶段变化（A 8 段 / B 3 段 / C 4 段）、父检查点路径稳定、10 轮都完成
- **定位 v1 四个缺陷**：回归 `best_val` 初值 0；Qwen 回复在 `reasoning_content` 而只读 `content`；
  C 第 4–10 轮 / B 第 3–10 轮候选被拒仍重训；漂移参考不跨轮、预测 JS 恒 0
- **v2 轮次控制器** `src/harness/round_controller.py`：跨轮参考画像、历史回读、模型裁决可读/可收窄、
  相对 gap、停手/探针、相对验收阈值；`harness/contest_submission.py` 接入 Gatekeeper 五件套
- **状态 schema 2**：`profile/baseline_profile/pred_profile`、`best/last/last_gain/stalled`、
  `decisions[≤10]`、`bounds/budget_steps/audit_events`
- **测试**：`tests/test_round_controller.py` 23 项全 PASS
- **彩排**：进程内 A 10 轮、B 3 轮、C 3 轮（含回归路径）、C 全量首轮 45k 行
  （RMSE 52.87→52.00，+1.65%，采纳）；容器内 `--network none` A 3 轮全 exit 0
- **回放**：v1 报告 119 epoch → v2 63 epoch（-47%），估计少 1291 CPU 秒（-40%）；
  A 不省（更新都有收益），B/C 主要省在“被拒还重训”的轮次
- **v2 镜像**：`zhixun-rsi-harness:0.3`（285 MB，tar 64 MB）已构建、断网 describe 通过；
  `dist/zhixun-rsi-harness-0.3.tar` 已导出
- **平台动作**：镜像已上传：v2 初版 archive `88f5afb…`、**v3 最终版** archive
  `880b0fd503abcc04b0d49433c15eeafb`（均 ready）；提交版本 v2（`8636a3d…`）与
  **v3 最终版**（`b318577fbc02cdfeb8504ac94c707c10`）均 `status=valid`。
  v3 = 0.3 镜像 + 含 C 题 145 上限的文档；两个版本都未消耗试运行次数。
  首次校验曾返回 `模型服务暂不可用`，退避约 2 分钟后重试成功。
- **材料**：`reports/技术报告_v2.md`、`reports/来源声明.md`、`docs/演示视频分镜.md`、
  `reports/v2_replay_v1.{md,json}`、`reports/trial_runs.md`、`reports/系统说明.md` 已更新

### 待办（第二次提交）

1. ~~上传 `dist/zhixun-rsi-harness-0.3.tar`~~ ✅（archive `880b0fd…`，ready）
2. ~~新建提交版本~~ ✅（v2 `8636a3d…` 中间版；**v3 `b318577…` 最终版**，均 valid）
3. **选 1 次试运行**（trial 已用 1/10；上传/建版本/校验都不扣次数）→ 看 A/B/C 的
   `decision`、C 是否停止白训、模型裁决是否仍有文本
4. 录演示视频（按 `docs/演示视频分镜.md`，≤5 分钟）
5. 确认试运行报告后再决定「选为正式版本 + 开始正式评测」
   （平台 `#teams` 显示 `formal_limit=3`，与交接单写的 2 次不一致，以平台为准）

### 新增问题 / 事实修正

- **Q21**：C 的 NASA 分数在 v1 报告中为 8.85e10，指数惩罚对严重高估极敏感；
  v2 已对 24 特征回归输出加 145 上限，效果待下一次试运行验证。
- **Q22**：v2 首次校验返回 `模型服务暂不可用，请稍后重试`；退避约 2 分钟后重试成功，
  状态变 `valid`。短时间重复请求会 409 `校验请求过于频繁`，不要连续点。
- 平台 `formal_limit=3`（`formal_remaining=3`），不是交接单里写的 2；试运行 `trial_used=1`。
- 第一次试运行（`41d6c1d6…`）平台侧失败，同版本重跑后完成；失败不占次数。
- **事实修正（SwanLab）**：`x5kz5fn7` 是本地 `simulate_a_rounds.py` 的 **static 模拟**
  （val 0.7140、test_mean 0.399、test_worst 0.185、**0 updates**），不是平台测评；
  `rynlo0ne`（trigger）在 `reports/sim_a_rounds.json` 里同样 **0 updates**。
  因此旧交接单里“trigger 隐藏-like 0.339 / 多训 = 负收益”的结论**没有当前 JSON 支撑**，
  且平台 A 实测 0.8735 说明 batch 3–10 不是可靠的 hidden 代理；v3 已不拿它设计平台预算。

### 下一步

- v2 平台试运行 1 次 → 按报告决定是否调 C 预算 → 正式评测。
- 若模型仍无文本：保持“规则执行 + `model_action=unavailable`”的诚实记录，不伪造裁决。

### 阻塞

- 上传/试运行/正式评测需要人工确认；无技术阻塞。

---

## 2026-11-20（第五段）：平台提交完成，首次试运行已排队

### 完成

- 镜像上传成功（`zhixun-rsi-harness:0.1`，64 MiB）；**提交版本 v1 已校验通过**
- 表单填写：基础模型 **Qwen/Qwen3.5-4B**、训练记录 = SwanLab run `x5kz5fn7`、
  源码 = `github.com/AliceJiuyuansi/rsi-zhixun-harness`（已 public）、PyTRiO key（新建）、
  作品文档 = `reports/系统说明.md`
- **首次试运行已提交**（trial，排在第 4 位；监视脚本 `_state/watch_run.py`，日志 `_state/run_watch.log`）
- SwanLab：项目 `rsi-zhixun` 两条真实 run（static / trigger 十轮模拟，56 条记录已上传）
- 新写 `docs/交接_2026-11-21.md`（明天一页上手）；`reports/系统说明.md`（提交三件套之一）

### 待办（明天从交接单开始）

1. 看试运行报告的 3 件事：每轮 `/data` 行数、父检查点路径、10 轮是否都 finish
2. 按报告调 `decide_training_budget()`
3. 技术报告 ≤10 页 / 演示视频 ≤5 分钟 / 来源声明
4. 确认后再考虑「选为正式版本 + 首次正式评测」（2 次机会）

### 阻塞

- 试运行排队中（平台侧）；正式评测需人工确认

---

## 2026-11-20（第四段）：平台开放、Harness 镜像、A 任务本地闭环

### 平台事实（登录后从页面/SDK 抓到的，可直接当已知条件）

- 平台 `rsi-index.115.zone` 已开放，账号 `Richard1037` 登录成功；队伍 1 支、提交版本 0、评测 0
- **每数据集 10 轮续训**；**试运行 10 次/队**（每次 = 一次完整"训练+评测"= 10 轮，累计不重置）；
  **正式评测 2 次**（第 1 次自助、第 2 次需审批）；上传镜像/新建版本**不扣次数**
- SDK 1.0.4：入口 `python3 /app/harness.py describe|train|predict`；`dataset_key` 只接受 A/B/C；
  `/data`、`/previous`、`/state` 只读，`/workspace` 可写；**每轮必须至少一次真实模型调用**
- 评分：客观 35（平均提升 15 / 最差阶段 5 / 恢复 5 / 稳定性 5 / 资源效率 5），
  专家 65（自主决策 10 / 漂移检测 8 / 安全 6 / 系统 6 / 技术报告 6 / 路线 6 / 受控实验 5 /
  实验记录 5 / 记录分析 5 / 工业约束 4 / 答辩 4），加分 10（扩展数据验证 5 + 技术创新 5）
- 硬要求新增一条：**至少 3 次"读历史—做实验—依据结果决策"的完整记录**

### 完成

- **A 任务数据核对**：`gas+sensor+array+drift+dataset (1)/Dataset/batch{1..10}.dat`
  与官方 `train.jsonl`(445)/`validation.jsonl`(1244) **逐位相同**（maxdiff=0.0），类别映射正确
  → batches 3–10（12,221 条）是合法的"伪隐藏集"，**只许考、不许练**
- **官方基线精确复现**：20 epoch → macro-F1 `0.7140297220788709`（与官方 training.json 16 位一致）
- **漂移曲线**（`reports/gas_baseline_drift.csv`）：batch1 0.956 → batch2 0.714 → batch9 0.185
- **续训适配器** `src/models/official_round.py`：`continue_round()` 从父检查点续训，
  记录父子摘要、逐 epoch 历史；新增**早停 + 最佳快照**
- **策略表** `src/policy/strategy.py`：PSI×掉分 → none / finetune(3/4/5 ep) / reweight，全部过白名单
- **十轮模拟** `scripts/simulate_a_rounds.py`：static 隐藏-like 均值 0.399 / 最差 0.185；
  periodic 验证刷到 0.7255 但隐藏-like 掉到 0.339 → **多训 = 负收益（过拟合旧分布）**
- **Harness 镜像** `zhixun-rsi-harness:0.1`（linux/amd64，285 MB，tar 64 MB）：
  `harness/harness.py` + `harness/contest_submission.py` + `vendor/contest_baselines/`
- **本地彩排** `scripts/test_harness_local.py`：进程内假宿主 + **容器内 `--network none`**
  跑完 describe/train(2 轮)/predict，全部 exit 0；不消耗平台次数
- `scripts/prepare_gas.py` → `data/processed/gas/gas_all.npz` + manifest（可追溯哈希）

### 未完成

- 镜像**未上传**、未跑试运行（会消耗 1/10 次，需本人确认）
- `reports/系统说明.md`、技术报告、演示视频、来源声明未写
- `docs/open_questions.md` Q20（10 轮数据是否变化）待问
- 训练配方（噪声增强 / 权重平均）未实验；`src/harness/loop.py` 的逐 step 内核尚未接进 `contest_submission.py`

### 下一步

1. 上传 tar → 新建提交版本 → **试运行 1 次**（同时验证 Q20）
2. 按试运行报告修 `decide_training_budget`
3. 写系统说明 + 技术报告（用漂移曲线和十轮模拟作图表）

### 阻塞

- 报名/推荐状态（人）；企业赛 PyTRIO 额度

---

## 2026-11-20（第三段）：官方三任务接入路径

### 完成

- 收到官方任务口径：A 气体识别（6 类 / Macro-F1）、B 轴承健康（RUL 分钟 / RMSE+MAE）、
  C 发动机寿命（RUL 循环 / RMSE + NASA 非对称评分）
- `docs/08_三任务落地路径.md`：三任务身份推断（UCI Gas Drift / FEMTO-ST PRONOSTIA / C-MAPSS）、
  复用度对照、看不懂数据时的侦察三步 + 通用特征四族、指标/阈值/loader 改造清单、三天计划
- 新增 Q16–Q19（三选一还是全做、指标口径与单位、公开集划分、能否用公开原版预训练）

### 关键判断

- 系统零重写：换的是**数据加载器 / 指标函数 / 阈值量纲**三样
- B 复用度最高（已有 XJTU-SY 全链路）；A 需 macro-F1；C 需新 loader + 非对称评分
- 阈值必须相对化（`k × std(y)`），否则 0.06 换到"分钟/循环"直接失效

### 下一步

1. 等 Q16 答复确定主线（推荐 B → A → C）
2. 数据到手先跑侦察脚本，不要先建模
3. 指标层加 `metric` 选择器（`macro_f1 / mae / rmse / nasa`），**全部转成越高越好**

---

## 2026-11-20（第二段）：镜像断网跑通 + 五件套 5/5

### 完成

- **Docker 三件套**：`Dockerfile` / `requirements.txt` / `.dockerignore` + `scripts/entrypoint.py`
  （`docker run <image>` 零参数、零网络、恒 exit 0；镜像自带 36 维特征缓存，不挂盘也能跑）
  - `docker pull python:3.11-slim` **直连成功** → registry mirrors 不必配，原来的"拉不到镜像"阻塞解除
  - 镜像 `zhixun-rsi:0.1`：Python 3.11.18 + numpy/sklearn/scipy/pyyaml，565 MB；**不装 swanlab**
  - `docker run --rm --network none` 断网 dry-run：static MAE 0.4364 / trigger 0.1292，
    7 次重训、6 次晋升、0 回滚，退出码 0；带 `-v` 时 `run.log`/`summary.json`/各 run 目录齐全
- **受控 RSI 五件套 5/5**：
  - `src/policy/whitelist.py`（白名单 + 参数范围 + 外部提案解析，永不抛异常）
  - `src/policy/bounds.py`（幅度上限：≤ 上版 2×，最多改 2 个参数）
  - `src/policy/budget.py`（次数上限：终身 12 次 + 窗口 8 次/40 步）
  - `src/harness/killswitch.py`（env / 配置 / STOP 文件 / 超时 / 连续异常，触发即锁死）
  - `src/utils/audit.py`（`<run>/audit.jsonl`，append-only、逐行 flush）
  - 统一闸口 `src/harness/guard.py`：kill-switch → 白名单 → 预算 → 幅度上限 → approve
- **故障注入验收** `scripts/check_guard.py`：9 类（F1 白名单 / F2 坏 JSON / F3 次数 / F4 幅度 /
  F5 kill-switch 多来源 / F6 断流 / F7 连续异常自动锁死 / F8 数据缺失 / 健康跑到尾）全部 PASS
- **回归未变**：4 轴承 × 3 策略 × 2 seed 聚合 static −0.4070 / periodic −0.1451 / trigger −0.1060；
  单流 `xjtu_3_4` trigger：MAE 0.1292、重训 7、晋升 6、回滚 0
- ⚠️ 成本口径提醒：`unit_gain_cost` 是 wall-clock 计时，**机器负载能改变 ±15%**
  （同一矩阵两次跑：periodic 114.1 → 121.9，trigger 174.7 → 178.6；MAE 不动）。
  结论"periodic 更便宜"不变，但最终报告里的成本数字要**空载机器复测**后再定稿。
- `ExternalPolicy`（`policy.type: external`）：LLM/外部 proposer 的接入位，
  输出必过 `parse_proposal` + Gatekeeper，proposer 抛异常也只是 `none`

### 新增产物字段

- `<run>/audit.jsonl`、`<run>/guard.json`；`summary.json` 增 `guard` / `audit` / `degraded_steps`
- `configs/base.yaml` policy 增：`max_retrains_per_window: 8`、`max_consecutive_failures: 3`、
  `max_wall_seconds: 0`、`amplitude_factor: 2.0`

### 未完成

- SwanLab 云端项目 / `swanlab login`、`trio login` 仍未做（**必须本人**）
- 报名状态、官方数据包、企业赛提交细节仍未确认
- 跨工况域偏移流、HPO 第二层、Decision Model（企业赛）未开始
- `reports/系统说明.md`（提交三件套之一）还没写

### 下一步

1. 本人：报名状态 + `swanlab login` + `trio login`（三条外部依赖）
2. 写 `reports/系统说明.md`（怎么跑、输入输出、资源占用、五件套怎么演示）
3. 跨工况域偏移流（35Hz 训 → 40Hz 部署）：区分"磨损"与"工况变了"
4. `docker save` 出 `.tar` 并复核体积

### 阻塞

- 登录类/报名类（等本人）；官方评测平台未开放（Q8）

---

## 2026-11-20：新手可操作性补齐（操作手册 + 三个教学工具）

### 完成

- `docs/07_操作手册.md`：harness 要不要训练（三种训练拆分）、数据五步流水线、
  本地自测三步、结果解读与排障表、动作白名单（none/retrain/finetune/swap_model/reweight
  + 自动保护 rollback/kill-switch）、五件套自查表
- `scripts/inspect_data.py`：原始 CSV → 36 特征 → npz → 流切分 → 误差增长，一条命令看完整条链
- `scripts/report_run.py`：把 `summary.json`/`metrics.json` 翻成人话（去掉会误导的
  unit_gain_cost_vs_ref，改为指向 matrix.md 与静态基线对比）
- `scripts/export_view.py`：导出 Excel 可读 CSV（`view/数据_*.csv`、`view/对比_*.csv`）
- Docker Desktop 29.9.3 已装并运行；**镜像加速仍未配置**，`Dockerfile`/`requirements.txt` 未写
- 按官方 README 装好 PyTRIO 体系：`pytrio` 0.2.9（可 import）、`trio` CLI、agent skill
  `pytrio-skill` v0.2.4（全局，`~/.agents/skills`）。**未登录**（需本人 `trio login`），校赛无额度不训练
- 记录到 `docs/07` 3.3b（环境）与 5.6（把动作选择做成 Decision Model 的思路）

### 未完成

- Docker `--network none` dry-run（阻塞于镜像加速配置）
- SwanLab 云端项目 `rsi-zhixun` 仍未创建
- 五件套：白名单枚举已有，校验器/kill-switch/审计 JSONL 仍缺
- `trio login` 未做（阻塞于本人操作）；PyTRIO 支持模型清单待登录后用 `get_supported_models()` 核对
  （规则 PDF 写 Qwen3.8-27B，skill 案例用 Qwen3.5-4B，仍未消解）

### 下一步

1. `swanlab login` → 重跑一条流 → 云端可见
2. 配 `registry-mirrors` → `docker pull python:3.11-slim`
3. 写 Dockerfile 三件套 → 断网 run 通
4. 审计日志 + kill-switch 最小实现

### 阻塞

- 官方评测平台未开放（Q8）；报名状态仍未核实

---

## 2026-11-11（当天第二段）：从零代码到真实数据闭环

### 完成

- 搭建完整可运行骨架：`src/`（数据/模型/漂移监控/策略/Harness 循环/成本账本/SwanLab 封装）、
  `configs/`（6 条合成流 + 4 条真实流）、`scripts/`（smoke / stream / matrix / 特征缓存）
- 合成漂移垂直切片已通过：static 在 concept/gradual 上掉到 0.62/0.68，
  trigger/periodic 恢复 +0.13~0.22，且 trigger 比 periodic 便宜约 33%
- 接入 **XJTU-SY 轴承全寿命数据集**（公开代理数据，非官方包）：
  - `scripts/build_xjtu_cache.py` 抽取 36 维时/频特征，9,216 个快照，8 进程 131 秒
  - 新增回归任务通路（HistGradientBoostingRegressor，perf = −MAE）
  - 4 条真实流（Bearing3_1/3_2/3_4/2_3）× 3 策略 × 2 seed 跑通
- 修复真实数据暴露的三个设计缺陷：
  1. PSI 只用单步 batch（6 行）计算 → 恒为 0，全部误判 concept；改为用最近 40 步窗口
  2. 晋升门控拿"不同分布上的验证分"比较 → 挑战者永远输；改为冠军/挑战者在**同一 holdout** 上比较
  3. 分组聚合把一步压成一行 → 训练历史不足；改为保留快照级样本、同一步多样本
- 真实数据结果：静态 MAE 0.34–0.45，periodic 0.11–0.20，trigger 0.09–0.12；
  单位增益成本 periodic 92 vs trigger 143 → **周期重训更划算**（详见 `docs/05`）

### 未完成

- 队伍报名状态仍未知；官方数据包仍未开放
- SwanLab 项目 `rsi-zhixun` 仍未在云端创建（本地 `swanlog/` 已验证）
- 跨工况域偏移流、超参数第二层、LLM Harness 均未开始

### 下一步

1. 跨工况流：35Hz/12kN 训练 → 40Hz/10kN 部署
2. 策略层自进化（超参搜索）接入同一门控与成本账本
3. 登录 SwanLab 后把这批 run 重放到云端，作"完整记录"演练
4. 官方数据包开放后优先重跑 `docs/05` 第 4 节

### 阻塞

- 报名状态、官方数据包与评测平台（均等外部）

---

## 2026-11-11（项目第 0 天）

### 完成

- 读完 `赛题规则.pdf` 全文，产出 `docs/00_赛题规则解读.md`
- 核实官方赛程，发现 `plan0.md` 时间轴整体晚约 3 周，产出 `docs/01` 与 `plan1.md`
- 建立工作区文档体系（README / AGENTS / docs 五篇 / plan1）
- 确认 SwanLab 团队空间存在：`@xiangxinnidexin`（"相信你的心"，中国人民大学，北京）

### 未完成

- 队伍报名状态未知（**最紧急**）
- 官方数据包未下载
- SwanLab 项目未创建、smoke run 未跑
- 答疑群未进、问题未发
- 代码为零

### 新增问题

- Q1–Q15 全部登记在 `docs/open_questions.md`，均待确认

### 下一步

1. 今天：确认报名 + 进群发问 + 建 SwanLab 项目
2. 11-12：数据包 + schema + smoke
3. 11-18：垂直切片（基线 + 流评测 + 成本 + 4 条合成流）

### 阻塞

- 报名状态（等教务处/组委会）
- 官方数据包与评测平台（等平台开放）

---

## 决策日志

| 日期 | 决策 | 理由 | 证据 |
|---|---|---|---|
| 2026-11-11 | 以 `plan1.md` 取代 `plan0.md` 作为执行计划 | plan0 的日期与官方赛程冲突，且缺少资格确认与分工 | `docs/01`、`docs/02` |
| 2026-11-11 | 保留 `plan0.md` 不修改 | 留痕，便于复盘；避免历史信息丢失 | — |
| 2026-11-11 | 一人一个 SwanLab 账号，共用一个团队项目 | 规则禁止账号转借；run 需可追溯到人 | `docs/00` 第十节 |
| 2026-11-11 | 官方包未开放期间用 XJTU-SY 作代理数据 | 全寿命退化天然对应 RUL 预测；结果标 `proxy`，不冒充官方成绩 | `data/VERSION.md` |
| 2026-11-11 | 任务默认做 RUL 回归，分类作为可切换分支 | 官方赛题写"剩余寿命预测"；回归与官方口径更接近 | `docs/05` 第 2 节 |
| 2026-11-11 | 晋升必须在**同一 holdout** 上比较冠军与挑战者 | 跨分布比较会让挑战者永远输，实测真实数据 0 次晋升 | `src/harness/loop.py` |
| 2026-11-20 | Harness 本身不"训练"：校赛阶段用规则+校验，PyTRIO 微调留到企业赛 | 校赛无 PyTRIO 额度；规则版可解释、可离线、成本为 0 | `docs/07` 第 1 节 |
| 2026-11-20 | 次数上限做成两层：终身 ≤12 + 窗口 ≤8/40 步 | 主限速仍是 `retrain_cooldown`，窗口上限只防突刺；设成 4/40 会误伤正常触发式（实测 MAE 0.129→0.173，多出 1 次回滚） | `scripts/check_guard.py` F3、`experiments/matrix-*` |
| 2026-11-20 | 未实现的动作（finetune/swap_model/reweight）在白名单里"声明但不放行" | 防止 LLM 提名后被静默当 retrain 执行；没有执行器就不许选 | `src/harness/guard.py`、F1 |
| 2026-11-20 | 镜像不装 swanlab，入口固定 `RSI_SWAN_MODE=disabled` | 断网是硬约束，装了也连不上，反而制造出网尝试；云端记录由宿主机跑的实验负责 | `Dockerfile`、`scripts/entrypoint.py` |
| 2026-11-20 | 镜像把 36 维特征缓存烤进去（1.3 MB），不要求挂盘 | 评测机挂载方式未知（Q8 未确认），自包含最稳；原始 16 GB 数据集不进镜像 | `Dockerfile`、`.dockerignore` |
| 2026-11-23 | v2 把“连续 2 轮无收益就停手”放在重训之前 | v1 报告显示 C 第 4–10 轮、B 第 3–10 轮候选全被拒却仍花 4/5 epoch；停手不改变已选模型，只省算力 | `reports/v2_replay_v1.md`、v1 trial report |
| 2026-11-23 | 回归 gap 改为相对 RMSE，`best` 不再用 0 初始化 | 绝对值 600/50 会把任何阈值冲垮，导致每轮无条件重训 | `src/harness/round_controller.py`、单测 |
| 2026-11-23 | 模型裁决只允许“收窄/保持/回滚”，不允许放大预算 | 4B 基础模型可能给出不可靠建议；规则保底、模型只能减少风险，符合“受控 RSI” | `apply_verdict()`、`docs/04` §4.6 |
| 2026-11-23 | 平台入口接入 Gatekeeper（白名单/幅度/次数/kill-switch） | v1 平台路径绕过了本地闸门；专家分里的“安全/受控实验”需要证据 | `harness/contest_submission.py`、`scripts/check_guard.py` |
| 2026-11-25 | v15 先缩短模型调用并收紧输出预算 | 第 12 版 30 次空回执均用满输出额度。三题配对验证显示空回执下训练结果一致 | [官网运行列表](https://rsi-index.115.zone/#runs)、本页 v15 记录 |

## 版本记录维护

每次创建提交版本时，在[版本演进记录](版本演进记录.md)追加方案、改进手段和理由；运行结束后在同一节回填结果与取舍。
