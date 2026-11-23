# 项目进度快照

> 每次会话/每周例会结束更新。最新的在最上面。
> 格式固定，方便 agent 与人快速接手。

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
