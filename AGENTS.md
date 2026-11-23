# AGENTS.md · 交接协议（给 dsh 及其他 AI agent）

> 你正在接手一个**有硬截止日期**的竞赛项目。
> 读完本文 + `README.md` + `docs/00` + `docs/01` + `plan1.md`，你就具备了继续工作的全部上下文。
> 最后更新：2026-11-23

---

## 0. 一句话任务

为"第二届北京市大学生人工智能+创新大赛"赛题 9「智训引擎」，
构建一个在**漂移数据流**上能**自动监测→归因→决策→更新→回滚**的工业模型自进化系统，
全部实验记录在 SwanLab 团队空间，最终交付**离线 Docker 镜像 + 实验记录链接 + 说明文档**。

**真正的截止不是 12 月底，是 11-25（校内推荐）/ 12-05（企业赛结束）。**

---

## 1. 工作区

- 根目录：`D:\Alice_AI\rsi`（Windows 11，Git Bash）
- 这里**就是**项目工作区，不要另建目录，不要动 `D:\Alice_AI` 下的其他文件夹
- `赛题规则.pdf` 是最高事实源。**注意：该 PDF 文本层是乱码**，用 pymupdf 提取会得到 mojibake。
  正确读法：`pymupdf` 渲染每页为 PNG（dpi≈170），再用图像方式阅读。
  已整理好的解读见 `docs/00_赛题规则解读.md`。

---

## 2. 阅读顺序（必须按序）

0. `docs/交接_2026-11-23.md` — **一页交接单（最新，先看这个）**；`docs/交接_2026-11-21.md`、`docs/交接_2026-11-20.md` 为历史版本
1. `README.md` — 全景与文件地图
2. `docs/00_赛题规则解读.md` — 规则原文 + 解读，**事实源**
3. `docs/01_赛程与关键节点.md` — **真实日期**，与 plan0 冲突处以本文为准
4. `plan1.md` — 当前执行计划（唯一有效的计划）
5. `docs/03_SwanLab使用与团队协作.md` — 记录即交付物
6. `docs/04_工程架构与规范.md` — 接口契约，改代码前必读
7. `docs/06_赛题任务与实现思路.md` — 任务全貌、微调 vs 重训、RSI 两层、20 个知识点（新手向）
8. `docs/08_三任务落地路径.md` — 官方 A/B/C 三任务的判定、可复用度、指标/阈值/loader 改造清单
9. `docs/07_操作手册.md` — 数据五步流水线、本地自测、结果解读、动作白名单、五件套自查
10. `docs/05_真实数据实验报告.md` — 当前真实数据结果、结论、局限
11. `docs/status.md` — 最新进度（每次会话结束必须更新）
12. `docs/02_plan0评析与plan1改动说明.md` — 只在你想知道"为什么改"时读
13. `plan0.md` — **历史存档，日期全错，不要照做**

---

## 3. 事实源优先级（冲突时从上到下）

1. `赛题规则.pdf` 原文
2. 组委会 / 企业答疑群的书面答复（记录在 `docs/open_questions.md`）
3. 学校教务处通知
4. `docs/00` `docs/01`
5. `plan1.md`
6. 其他任何文档、聊天记录、记忆

**发现冲突 → 改文档，不要默默按自己的理解做。**

---

## 4. 已确认的事实（截至 2026-11-20）

| 事实 | 来源 |
|---|---|
| 赛题：智训引擎，工业制造场景下的模型训练与超参数优化 | PDF |
| 命题企业：软通计算机、华为 | PDF |
| 团队 3–5 人，指导教师 ≤2，每人一队，每队一赛题 | PDF / 通知 |
| 校赛阶段**无** PyTRiO 额度；SwanLab 免费可用 | PDF p5 |
| 企业赛窗口 **11-26 ~ 12-05**；决赛 **12-26** 北工大 | 官方通知 |
| 校内推荐截止 **11-25**，每校每赛题 ≤5 队 | 官方通知 |
| 官网初始报名截止 **11-07 / 11-09**（各校转述不一）→ **已过期，待核实** | 通知 |
| 提交三件套：镜像、云端实验记录链接、系统运行说明 | PDF p3-4 |
| 三条硬门槛：无人工干预跑完 / 记录完整 / ≥静态基线 | PDF p4 |
| 客观分看"单位增益消耗的算力、标注、Token" | PDF p4 |
| 驱动 Harness 的 LLM 只能是 PyTRiO 模型或 Qwen3.8-27B | PDF p3 |
| 评测环境：指定设备、统一、**离线**、有资源限制 | PDF p4 |
| Python ≥3.10；数据小、有预提取特征、**CPU 可完成基础任务** | PDF p4 |
| 允许失败重跑一次，取两次较高 | PDF p4 |
| SwanLab 团队："相信你的心" / `@xiangxinnidexin` / 中国人民大学 / 北京 | SwanLab API |

### 尚未确认（不要当成事实）

- 企业赛线上提交的确切日期
- 镜像规范（基础镜像、入口、资源上限、超时、断网程度）
- 静态基线定义、综合性能公式、客观分公式
- 离线评测机是否预置 27B 权重
- 人大校内报名/推荐的具体流程

→ 全部在 `docs/open_questions.md`，**不要自己编答案**。

---

## 5. 当前状态（2026-11-20，代码可运行 + 新手手册齐备）

> **2026-11-23 更新**：v1 试运行已完成（A/B/C 各 10 轮），v2 控制器与第二次提交材料已就绪；
> 最新状态以 [`docs/交接_2026-11-23.md`](docs/交接_2026-11-23.md) 与 `docs/status.md` 顶部为准，
> 下面 2026-11-20 的段落保留为历史。

> 本次交接的**单页摘要**见 [`docs/交接_2026-11-20.md`](docs/交接_2026-11-20.md)。

### 已完成

- 规则 PDF 已读完并整理成 `docs/00`；官方赛程已核实并产出 `plan1`
- **完整可运行骨架**：`src/`（数据/模型/漂移监控/策略/Harness 循环/成本账本/SwanLab 封装）、
  `configs/`（6 条合成流 + 4 条真实流）、`scripts/`（smoke / stream / matrix / 特征缓存 / inspect / report / export）
- 合成漂移切片验证通过：static 明显退化，trigger/periodic 恢复 +0.13~0.22
- **真实数据闭环**：XJTU-SY 代理数据 9,216 快照 → 36 维特征 → RUL 回归 → 4 轴承 × 3 策略 × 2 seed
  - static 0.407 / periodic 0.145（单位增益成本 92.4）/ trigger 0.106（143.1）
  - 结论：trigger 更准、periodic 更便宜；见 `docs/05`
  - ⚠️ MAE 完全可复现；单位增益成本是 wall-clock 计时，机器负载能改变 ±15%（2026-11-20 两次复测得
    periodic 114~122 / trigger 175~179），**最终报告的成本数字要空载复测后定稿**
- **新手可操作性**：`docs/07_操作手册.md` + `scripts/inspect_data.py` / `report_run.py` / `export_view.py`
- SwanLab 本地模式已验证（`swanlog/`）；**云端项目尚未创建**
- PyTRIO 体系已装：`pytrio` 0.2.9（可 import）、`trio` CLI、agent skill `pytrio-skill` v0.2.4
  （全局 `~/.agents/skills/pytrio-skill`）；**未 `trio login`**
- **Docker 交付链已通**：`Dockerfile`/`requirements.txt`/`.dockerignore`/`scripts/entrypoint.py` 齐；
  `docker pull python:3.11-slim` 直连成功（**不再需要 registry mirrors**）；镜像 `zhixun-rsi:0.1`
  已构建，`docker run --network none` 断网 dry-run 通过（static 0.4364 / trigger 0.1292，exit 0）
- **五件套 5/5**：`policy/whitelist.py`+`bounds.py`+`budget.py`、`harness/killswitch.py`、
  `utils/audit.py`，统一闸口 `harness/guard.py`；`scripts/check_guard.py` 9 类故障注入全 PASS
- `<run>/audit.jsonl`（append-only）+ `<run>/guard.json`；`summary.json` 增 `guard`/`audit`/`degraded_steps`

### 未完成 / 不存在

- 官方数据包**尚未确认下载**（当前全部结果都是 `proxy`）
- 队伍报名状态**未知**
- SwanLab 项目 `rsi-zhixun` **未在云端创建**；`swanlab login` 未做
- `trio login` 未做（要本人交互）；PyTRIO 支持模型清单未核对
- `reports/系统说明.md`（提交三件套之一）未写；镜像 `.tar` 未导出
- `finetune` / `swap_model`(HPO) / `reweight` 只有白名单与参数范围，**执行器未实现**（闸门会拒绝）
- 跨工况域偏移、HPO 第二层、LLM Harness / Decision Model 均未开始
- 答疑群**尚未提问**

> 也就是说：机制骨架和真实数据闭环齐了，下一阶段是**工程化交付**（镜像 + 记录 + 五件套），
> 再往上才是机制加深（见第 8 节）。任何新结果必须区分 `proxy`（XJTU-SY）与官方数据。
> 任何新结果必须区分 `proxy`（XJTU-SY）与官方数据，不得混淆。

---

## 6. 硬约束（违反 = 作品无效）

1. **离线**：最终镜像不得依赖外网，不得调用外部 API
2. **无人工干预**：一条命令跑完全程，异常必须降级为"保持当前模型"，不能崩
3. **≥ 静态基线**：任何策略上线前必须过门控；没把握就 `none`
4. **记录完整**：所有实验进 SwanLab，禁止事后补录
5. **受控 RSI**：白名单 / 幅度上限 / 次数上限 / kill-switch / 审计日志，五件套缺一不可
6. **账号合规**：一人一号，不转借，key 不进仓库
7. **不碰隐藏集**：不写任何探测、攻击评测平台的代码

---

## 7. 绝对不要做

- ❌ 不要修改 `plan0.md`（历史存档）
- ❌ 不要把 `data/`、`experiments/`、权重、API Key 提交进 git
- ❌ 不要伪造、补录、美化 SwanLab 记录
- ❌ 不要用外部 LLM API 作为最终方案
- ❌ 不要追绝对精度而忽略成本
- ❌ 不要在 `src/` 之外写业务逻辑
- ❌ 不要在没有确认报名状态的情况下埋头写代码（先问人）
- ❌ 不要把猜测写成"已确认"

---

## 8. 下一步（按优先级，P0 先做）

| # | 任务 | 判据 | 依赖 |
|---|---|---|---|
| 1 | 确认报名/校内推荐状态 | 得到明确答复并写入 `status.md` | 人 |
| 2 | `swanlab login` → 建项目 `rsi-zhixun` → 跑 1 条 smoke run | 云端能看到 run | 账号 |
| 3 | `trio login` → `trio env show` → `get_supported_models()` | 拿到真实模型清单，消解 `Qwen3.8-27B` / `Qwen3.5-4B` 矛盾 | 账号 |
| 4 | 配 Docker `registry-mirrors`（可选） | 只在 `docker pull` 超时时才需要；2026-11-20 直连已成功 | 网络 |
| 5 | ✅ 镜像三件套 + `docker run --network none` dry-run | 已完成：断网一条命令出结果、exit 0 | — |
| 6 | ✅ 五件套补齐 + 故障注入验收 | 已完成：`scripts/check_guard.py` 9/9 PASS | — |
| 7 | 写 `reports/系统说明.md` + `docker save` 出 `.tar` | 提交三件套里的"运行说明"和"镜像"落地 | 5 |
| 8 | 跨工况域偏移流（35Hz 训练 → 40Hz 部署） | 触发式能区分"磨损"与"工况变了" | 现有缓存 |
| 9 | 策略层自进化（HPO 接入同一门控/成本账本） | 单位增益成本低于固定超参 | 8 |
| 10 | Decision Model 版动作选择（企业赛） | 见 `docs/07` 5.6 | PyTRIO 额度 |
| 11 | 官方数据到手后重跑 `docs/05` 第 4 节 | 结果替换 proxy 数字 | 平台 |

> 详细到天的排期见 `plan1.md`。1–3 是外部依赖（**登录类必须本人操作**）；7–9 现在就能做。

### 常用命令

```bash
export PYTHONIOENCODING=utf-8
python scripts/smoke.py                                  # 秒级自检
python scripts/check_guard.py                            # 五件套 9 类故障注入（全 PASS 才退出码 0）
python scripts/build_xjtu_cache.py                       # 重建特征缓存（约 2 分钟）
python scripts/run_stream.py --stream xjtu_3_4 --policy trigger --seed 0
python scripts/run_matrix.py --streams xjtu_3_1,xjtu_3_2,xjtu_3_4,xjtu_2_3 \
    --policies static,periodic,trigger --seeds 0,1
docker run --rm --network none zhixun-rsi:0.1             # 断网交付验证
```

### 改代码前必须知道的三件事

1. `data.task` 决定性能指标：`classification` → accuracy，`regression` → −MAE；
   所有 `perf` 都是**越高越好**。
2. 漂移统计用**最近 `policy.window` 步的窗口**，不是单步 batch（否则 PSI 样本量不足）。
3. 挑战者必须在**与冠军相同的 holdout** 上评分（`src/harness/loop.py` 中 `hold_X/hold_y`），
   不要退回"跨窗口比验证分"的写法，那会让晋升永远失败。

---

## 9. 工作方式（协议）

### 每次会话结束前

更新 `docs/status.md`：

```markdown
## 2026-11-11
- 完成：xxx
- 未完成：xxx（原因）
- 新增问题：Q16 xxx
- 下一步：xxx
- 阻塞：xxx（等谁）
```

### 每个实验

- 先写 `configs/exp/*.yaml`，再跑
- 跑完把 run 链接贴到 `docs/status.md`
- 失败也记，注明原因

### 每次决策

在 `docs/status.md` 的「决策日志」加一行：
`日期 | 选项 A vs B | 选了什么 | 理由 | 证据链接`

### 每次改接口

先改 `docs/04` 的接口契约，再改代码，并在 `status.md` 通知全队。

---

## 10. 环境备忘

- OS：Windows 11，shell 为 Git Bash
- 本机 Python：3.14（`C:\Users\久远寺\AppData\Local\Programs\Python\Python314`）
  → 赛题要求 ≥3.10，3.14 可用，但**依赖兼容性风险高**；建议锁 3.10/3.11 做正式环境
- 控制台是 GBK：Python 打印中文可能 `UnicodeEncodeError`，用 `PYTHONIOENCODING=utf-8` 或写文件
- 可用库：`pymupdf`（读 PDF）；`pypdf` 未安装
- 已装：`pytrio 0.2.9`（连带 modelscope 1.40.3 等）、`swanlab 0.10.5`、`numpy/scikit-learn`
  - `trio` CLI 在 `.../Python314/Scripts/trio`；登录态写 `~/.pytrio`；**未登录**
- Agent skill：`~/.agents/skills/pytrio-skill`（v0.2.4，全局装、已软链 Claude Code）
  - 最有用的是 `references/decision-model.md`：固定候选决策 ≈ 我们的动作白名单
- Docker：Desktop 29.9.3，context `desktop-linux`，Storage Driver overlayfs
  - 2026-11-20 实测：`docker pull python:3.11-slim` **直连可用**，registry mirrors 不必配
    （只在超时时才去 Settings → Docker Engine 加镜像加速）
  - 镜像 `zhixun-rsi:0.1` 已构建；入口 `scripts/entrypoint.py`，零参数、零网络、恒 exit 0
- 网络：`swanlab.cn` 可 curl；部分高校站点 WebFetch 被拦截，用 curl 兜底
- SwanLab 页面是 Next.js 客户端渲染，项目列表需登录；`/api/group/<username>` 可匿名读团队信息

---

## 11. 如果你（agent）只能做一件事

**让系统端到端跑通一次**：一条流 → 一个基线 → 一条 SwanLab 记录 → 一个成本数字。

其他所有优化，都建立在这条链路存在的前提上。
