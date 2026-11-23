# 智训引擎 · 工业制造场景下的模型训练与超参数优化

<!-- latest-trials -->
校赛只评 A，主包选 v6，隐藏 macro-F1 为 0.873469，高于 v5 的 0.867426。若 A/B/C 三题均评，推荐 v5，B、C 的隐藏 RMSE 分别为 352.083 和 54.961，优于 v6 的 569.046 和 56.315。 [提交材料与接手顺序](docs/交接_v6完成.md) · [平台试运行复盘](reports/v4-v6试运行复盘.md)
<!-- /latest-trials -->

> 第二届北京市大学生"人工智能+"创新大赛 · 赛题 9
> 工作区根目录：`D:\Alice_AI\rsi`
> 状态快照：2026-11-24（v4/v5 建版：C 无上限 + 工况校准；v2 仍是正式候选）

本目录是**唯一的项目工作区**。所有规则解读、计划、实验约定、代码、配置都放在这里；
不在微信/飞书/个人电脑桌面另起一份。

---

## 1. 这是什么比赛（一句话）

在**漂移的工业数据流**上，做一个**能自己发现问题、自己决定要不要更新、自己验证、自己回滚**的系统，
并且用 SwanLab 把全过程记录下来。评委既看你"最后多准"，也看你"**每提升 1 个单位性能花了多少算力/标注/Token**"，
还要看这个系统在**断网、限资源、无人干预**的情况下能不能活下来。

- 主办：北京市教育委员会；承办：北京工业大学
- 命题企业：软通计算机有限公司、华为技术有限公司
- 赛题全称："智训引擎"——工业制造场景下的模型训练与超参数优化
- 关键词：数据漂移（concept drift）、RSI（Recursive Self-Improvement）、Harness（LLM 驱动的工具/技能编排层）、受控自进化

---

## 2. 五分钟上手（新队员 / 新 agent 从这里开始）

1. 读 [`docs/00_赛题规则解读.md`](docs/00_赛题规则解读.md) —— 规则原文 + 逐条解读，**这是事实源**
2. 读 [`docs/01_赛程与关键节点.md`](docs/01_赛程与关键节点.md) —— 真实日期（**与 plan0 有重大出入**）
3. 读 [`plan1.md`](plan1.md) —— 当前执行计划（**以这份为准，plan0 仅作历史存档**）
4. 读 [`docs/03_SwanLab使用与团队协作.md`](docs/03_SwanLab使用与团队协作.md) —— 每个人必须配好
5. 读 [`docs/04_工程架构与规范.md`](docs/04_工程架构与规范.md) —— 目录、接口、命名、提交物
6. Agent（dsh 等）另读 [`AGENTS.md`](AGENTS.md) —— 交接说明、禁止事项、当前待办

---

## 3. 文件地图

| 路径 | 作用 | 谁维护 |
|---|---|---|
| `README.md` | 总入口 | 队长 |
| `AGENTS.md` | **给 AI agent 的交接协议**（dsh 必读） | 队长，变更需记录 |
| `plan0.md` | 最初的备赛清单（历史存档，**日期已过时**） | 不再修改 |
| `plan1.md` | 修正后的执行计划（当前有效） | 队长，每周更新 |
| `docs/00_赛题规则解读.md` | 规则逐条翻译成人话 + 设计含义 | 全员 |
| `docs/01_赛程与关键节点.md` | 官方赛程、校历、倒排 | 队长 |
| `docs/02_plan0评析与plan1改动说明.md` | 为什么改、改了什么 | 队长 |
| `docs/03_SwanLab使用与团队协作.md` | 安装、登录、团队空间、命名规范、合规 | 实验管理员 |
| `docs/04_工程架构与规范.md` | 系统架构、接口契约、目录、成本口径 | 技术负责人 |
| `docs/open_questions.md` | 待组委会/群内确认的问题台账 | 全员补充 |
| `docs/05_真实数据实验报告.md` | XJTU-SY 代理数据结果、结论、局限 | 技术负责人 |
| `docs/06_赛题任务与实现思路.md` | **新手向**：任务梳理、微调、RSI、知识点、5 天排期 | 全员必读 |
| `docs/08_三任务落地路径.md` | **官方三个任务（气体/轴承/发动机）怎么落地**、看不懂数据时的侦察法、指标与阈值改造清单 | 技术负责人 |
| `docs/07_操作手册.md` | **保姆级操作**：harness 要不要训练、数据处理五步、怎么测、怎么看结果、动作白名单 | 全员必读 |
| `docs/交接_2026-11-20.md` | **一页交接单**：现状、数字、环境、待办、约束（给 dsh / 新队员） | 队长 |
| `docs/交接_2026-11-24.md` | **最新一页交接单**：v4/v5 建版、C 校准、试运行顺序 | 队长 |
| `docs/交接_2026-11-23.md` | 上一版交接单：v1/v2 试运行、v3 上限证伪 | 队长 |
| `docs/status.md` | 每周进度快照（agent 交接也看这个） | 队长 |
| `src/` | 代码（含受控五件套：`policy/whitelist|bounds|budget.py`、`harness/killswitch|guard.py`、`utils/audit.py`） | — |
| `configs/` | YAML 配置（实验参数全在这，不写死在代码里） | — |
| `Dockerfile` / `requirements.txt` / `.dockerignore` | 离线镜像构建三件套 | — |
| `scripts/` | 一键脚本（smoke / stream / matrix / check_guard / entrypoint） | — |
| `data/` | 数据目录（当前为 XJTU-SY 代理数据），见 `data/README.md` / `data/VERSION.md` | — |
| `experiments/` | 本地实验产物（模型、指标、成本账本） | — |
| `reports/` | 答辩材料、图表、成本收益曲线 | — |
| `赛题规则.pdf` | 官方原文件，**最高事实源** | 只读 |

---

## 4. 三条硬门槛（不过线 = 不参与排名）

1. 封闭评测中**无人工干预**跑完全部阶段，并按时输出结果
2. 全部训练/微调/调整/评估实验按《实验记录规范》在 SwanLab 留下**完整记录**
3. 封闭评测**综合性能不低于组委会静态基线**（部署后不再更新的模型）

> 注意：门槛 3 意味着"花里胡哨但比基线差"= 直接出局。**先保底，再冲刺。**

---

## 5. 当前最紧急的三件事

| # | 事项 | 截止 | 负责人 |
|---|---|---|---|
| 1 | **确认队伍是否已完成官网初始报名**（官方通知写 11-07/11-09 截止，今天 11-11） | 今天 | 队长 |
| 2 | 进赛题答疑群，把 `docs/open_questions.md` 里的问题发出去 | 今天 | 队长 |
| 3 | 5 人各自注册 SwanLab 并加入团队空间 `@xiangxinnidexin` | 11-12 | 全员 |

---

## 6. 团队与账号

- SwanLab 团队空间：<https://swanlab.cn/@xiangxinnidexin>（团队名"相信你的心"，机构：中国人民大学）
- 规则明确：**账号不得泄露或转借**。所以每个人用自己的账号，加入同一个团队空间，而不是共用一个账号。
- 比赛的提交物里包含"云端实验记录链接/查询依据"，团队空间必须能让评委看到。

---

## 7. 现在能跑什么（2026-11-11 起）

代码已从"零"变成**可运行的系统骨架**，数据用公开的 XJTU-SY 轴承全寿命集（**非官方包**，
见 [`data/VERSION.md`](data/VERSION.md)）。

```bash
cd D:/Alice_AI/rsi
export PYTHONIOENCODING=utf-8      # Git Bash 下中文日志不乱码

# 1) 快速自检（几秒，不碰真实数据）
python scripts/smoke.py

# 2) 从原始振动 CSV 抽特征缓存（已有缓存则跳过；全量约 2 分钟 / 8 进程）
python scripts/build_xjtu_cache.py
#    -> data/processed/xjtu/*.npz + manifest.json（9,216 个快照）

# 3) 单条真实流：静态 / 周期 / 触发
python scripts/run_stream.py --stream xjtu_3_4 --policy trigger --seed 0

# 4) 策略矩阵（4 个轴承 × 3 策略 × 2 seed）
python scripts/run_matrix.py --streams xjtu_3_1,xjtu_3_2,xjtu_3_4,xjtu_2_3 \
    --policies static,periodic,trigger --seeds 0,1
#    -> experiments/matrix-<ts>/matrix.{csv,md,json}

# 5) 看不懂结果时：从头看一遍数据流水线 + 人话版成绩单
python scripts/inspect_data.py
python scripts/report_run.py

# 6) 受控 RSI 五件套：9 类故障注入（非法动作/坏 JSON/超预算/超幅度/kill-switch/断流/…）
python scripts/check_guard.py

# 7) 离线镜像：构建 + 断网跑（评测机形态）
docker build -t zhixun-rsi:0.1 .
docker run --rm --network none zhixun-rsi:0.1

# 8) v2 平台入口彩排（假宿主，不消耗平台次数）
python scripts/test_harness_local.py --dataset A --rounds 10 --grow
python scripts/test_harness_local.py --dataset C --rounds 2 --grow --max-train-rows 8000 --docker --image zhixun-rsi-harness:0.5
python scripts/replay_trial.py            # v1 报告 -> v2 预算回放
```

合成漂移流（`stable/gradual/sudden/recurring/concept/lshift`）仍然保留，
用于可控地压测单条机制；真实数据用于验证"自然退化"下的系统行为。

**已有结果**：静态模型 MAE 0.34–0.45，周期重训 0.11–0.20，触发式 0.09–0.12；
按单位增益成本算周期重训更划算。完整数据、结论与局限见
[`docs/05_真实数据实验报告.md`](docs/05_真实数据实验报告.md)
（MAE 可复现；成本是 wall-clock 计时，负载会让它差 ±15%，最终数字需空载复测）。

**交付链现状**：受控五件套 5/5（`scripts/check_guard.py` 全 PASS）；
离线镜像 `zhixun-rsi:0.1` 已构建，`docker run --network none` 断网跑通、exit 0。
**v4/v5 候选镜像** `zhixun-rsi-harness:0.4/0.5` 已构建（285 MB，tar 64 MB），
`docker run --rm --network none zhixun-rsi-harness:0.5 describe` 通过；
v1 平台试运行报告见 `docs/status.md` / `reports/技术报告_v2.md`。

> SwanLab 当前默认 `auto`：没登录时写本地 `swanlog/`（不阻塞），登录后自动上传团队空间。

---

## 8. 事实源优先级

`赛题规则.pdf`（官方原文） > 组委会/企业答疑群答复 > 学校教务处通知 > `docs/00` > `plan1.md` > 其它

冲突时永远以更靠前的为准，并在 `docs/open_questions.md` 记录冲突。
