# 03 · SwanLab 使用与团队协作

> 版本：2026-11-11
> 官方文档：<https://docs.swanlab.cn>
> 本文结合官方 Quick Start 与本赛题要求编写。**SDK 参数随版本变化，动手前用 `swanlab --help` 或 `python -c "import swanlab; help(swanlab.init)"` 核对一遍。**

---

## 一、为什么这个比赛必须用 SwanLab

不是"用了加分"，是**交付物的一部分**：

1. 赛题规则原文：全部训练、微调、调整与评估实验须在云端实验记录平台**完整记录**
2. 提交资料第二项：**完整的云端实验记录链接或查询依据**
3. 硬门槛第二条：记录不完整 → **不进排名**
4. 主观分"实验规范与分析深度"直接看你的看板

👉 所以：**没有进 SwanLab 的实验 = 没做过。** 本地跑出好结果但没记录，等于零。

---

## 二、先回答那个问题：每个队员的机器都要配吗？

### 结论

| 情况                      | 要不要装            | 说明                 |
| ----------------------- | --------------- | ------------------ |
| 跑训练 / 微调 / 评测的机器        | **必须装**         | 不装就没有记录，硬伤         |
| 跑 Docker 镜像 dry-run 的机器 | **必须装**         | 评测同样算"评估实验"，要记录    |
| 只写文档 / 做 PPT 的机器        | 可以不装            | 网页端能看板；但建议装，方便本地验证 |
| 多人共用一台机器                | **每人各自登录自己的账号** | 禁止共用账号（见第六节）       |

### 关键区分

- **"装在每台机器上" ≠ "每台机器建一个项目"**。
  所有 run 都进**同一个团队空间下的同一个项目**，机器只是上传端。
- **账号必须一人一个**。规则原文："妥善保管……账号，**不得泄露或转借**"。
  共用账号 = 违规 + 无法追溯谁做的实验（答辩时说不清）。
- **登录一次即可**，凭据会持久化到用户目录（`~/.swanlab/`），不用每次登录。

### 推荐配置

```
5 名队员 × 各自电脑：pip install swanlab + 各自账号 swanlab login
                    ↓ 全部加入
        团队空间 @xiangxinnidexin（相信你的心 / 中国人民大学）
                    ↓ 下辖
        项目 rsi-zhixun（所有 run 都在这）
```

- 团队空间：<https://swanlab.cn/@xiangxinnidexin>
- 项目名建议：`rsi-zhixun`（ASCII，避免中文路径/编码问题）
- 如果队伍已有项目，沿用现有项目名，**不要新建第二个**——记录分散会直接影响"完整性"判定。

---

## 三、安装与登录

### 3.1 安装

```bash
pip install swanlab
# 慢的话用国内源
pip install swanlab -i https://mirrors.cernet.edu.cn/pypi/web/simple
```

Windows 建议在**同一套 Python 环境**里装（和训练脚本同一个解释器），否则会出现"命令行能登录、脚本里 import 不到"。

### 3.2 登录（三种方式，任选）

```bash
# 方式 1：交互式（推荐第一次用）
swanlab login
# 提示时粘贴 API Key（在 https://swanlab.cn/settings 复制）

# 方式 2：命令行直接给 key（Windows CMD 粘贴有问题时用这个）
swanlab login -k 你的API_KEY

# 方式 3：Python 里登录并保存
python -c "import swanlab; swanlab.login(api_key='你的API_KEY', save=True)"
```

验证：

```bash
swanlab --version
```

> Windows 提示：Git Bash 里粘贴用 `Shift+Insert`；CMD 里右键粘贴。粘贴后**屏幕不会显示内容**，是正常的。

### 3.3 多账号 / 共用机器

```bash
swanlab logout          # 退出当前账号
swanlab login           # 换人登录
```

或者在脚本里用环境变量临时指定，避免改动全局登录态：

```bash
export SWANLAB_API_KEY=xxx      # Git Bash
set SWANLAB_API_KEY=xxx         # CMD
$env:SWANLAB_API_KEY="xxx"      # PowerShell
```

> ⚠️ 不要把 API Key 写进代码、配置文件、Git 仓库。CI 里用仓库 Secrets。

---

## 四、最小可用模板（直接抄）

```python
import swanlab

run = swanlab.init(
    project="rsi-zhixun",
    experiment_name="p1-static-baseline-s1",   # 命名规范见第五节
    description="静态基线：首段数据训练后冻结，评估于 synth_gradual",
    config={
        "stage": "p1",
        "method": "static",
        "seed": 1,
        "data_version": "v0.1",
        "model": "lightgbm",
        "window": 200,
        "stream": "synth_gradual",
    },
    tags=["p1", "baseline", "static"],
    group="G1-static",           # 对照实验分组
    job_type="train",
)

for step, batch in enumerate(stream):
    perf = evaluate(batch)
    cost = ledger.step_cost()
    swanlab.log({
        "perf_online": perf,
        "perf_delta_vs_static": perf - baseline_perf,
        "gpu_seconds": cost.gpu_seconds,
        "labels_used": cost.labels,
        "tokens_used": cost.tokens,
        "unit_gain_cost": cost.total() / max(perf - baseline_perf, 1e-6),
    }, step=step)

swanlab.finish()
```

**必须做的三件事**：`init` 带 `config`、循环里 `log(step=...)`、结束 `finish()`。
少任何一个都会出现"run 一直 running"或"曲线是断的"。

---

## 五、统一命名规范（全队强制）

> 目的：评委打开项目，一眼能看出哪条 run 是什么；我们自己也能批量筛选。

### 实验名 `experiment_name`

```
{阶段}-{方法}-{漂移类型}-s{种子}
```

| 字段 | 取值 |
|---|---|
| 阶段 | `p0` `p1` `p2` `p3` `p4` `p5` |
| 方法 | `static` `periodic` `trigger` `two-layer` `smoke` |
| 漂移 | `gradual` `sudden` `recurring` `lshift` `all` |
| 种子 | `s1` `s2` `s3` |

示例：`p2-trigger-gradual-s2`、`p3-two-layer-all-s1`

### 标签 `tags`

至少包含：阶段、漂移类型、方法族。例：`["p2","drift","trigger"]`

### 分组 `group`

对照实验必须同组，方便一键对比：
`G1-static` / `G2-periodic` / `G3-trigger` / `G4-two-layer`

### 配置 `config`（每次实验必须齐全）

```yaml
stage: p2
method: trigger
seed: 1
data_version: v0.1          # 数据包版本，必须写
stream: synth_gradual       # 哪条流
model: lightgbm
feature_set: pre_extracted  # 预提取特征
window: 200
retrain_threshold: 0.15
max_retrains_per_day: 3
llm_backend: rule           # rule / local_llm / pytrio_llm
prompt_version: v3          # 用了 LLM 才写
```

> **`data_version` 和 `llm_backend` 是最容易漏、也最影响复现的两项。**

---

## 六、团队空间怎么建、怎么加人

### 6.1 确认空间归属

1. 打开 <https://swanlab.cn/@xiangxinnidexin>，确认是团队（TEAM）空间
2. 空间主页 → 新建项目 `rsi-zhixun`
3. 项目设置里确认 **可见性**：评委要能打开链接 → 设为**公开**，或准备分享链接

> ⚠️ 私有项目会导致"提交了链接但评委打不开"，等于没提交。
> 提交前用**无痕窗口**打开一次链接验证。

### 6.2 加人

1. 团队空间 → 成员管理 → 邀请（按用户名/邮箱）
2. 权限：默认 **WRITE**（能记录、能建实验）；只有队长给 **ADMIN**
3. 每人接受邀请后，用自己的账号 `swanlab login`
4. 验证：5 个人各跑一次 `scripts/smoke.py`，在同一个项目下能看到 5 条 run

### 6.3 SDK 里的空间指定

新版 SwanLab 支持在 `init` 里指定 `workspace`：

```python
swanlab.init(workspace="xiangxinnidexin", project="rsi-zhixun", ...)
```

> 老版本没有 `workspace` 参数。若报 `unexpected keyword argument`，
> 就在网页端把项目建在团队空间下，脚本里只写 `project="rsi-zhixun"`，
> 并确保账号**只属于一个同名项目所在的空间**。有歧义时优先用网页端创建项目。

---

## 七、这个赛题该记录什么（比赛专用口径）

### 7.1 每次 run 的 config（复现的最小集合）

- 数据：`data_version`、`stream`、`split`、`window`
- 模型：`model`、`feature_set`、超参
- 策略：`retrain_threshold`、`max_retrains`、`decision_policy`
- 成本：`budget_total`、`label_quota`
- 环境：`python`、`torch/sklearn` 版本（`swanlab` 会自动采集一部分）

### 7.2 每次 step 的 metrics（客观分直接相关）

| 指标 | 含义 | 为什么必须 |
|---|---|---|
| `perf_online` | 当前窗口性能 | 主指标 |
| `perf_delta_vs_static` | 相对静态基线增量 | 门槛 3 看这个 |
| `gpu_seconds` | 累计 GPU 秒 | 客观分 |
| `labels_used` | 累计标注条数 | 客观分 |
| `tokens_used` | 累计 Token | 客观分 |
| `unit_gain_cost` | 成本 / 增益 | **核心优化目标** |
| `latency_p95` | 决策+推理延迟 | 镜像超时风险 |
| `retrain_count` | 重训次数 | 受控机制 |
| `rollback_count` | 回滚次数 | 受控机制 |

### 7.3 汇总与产物

- `swanlab.log({"cost_breakdown": ...})` 之外，把成本表存成 `experiments/cost_ledger.csv` 并用 `swanlab.save()` 上传
- 关键图（漂移曲线、对照柱状图）用 `swanlab.Image(...)` 记录，答辩直接取用
- 决策日志（JSONL）作为 run 的附件

---

## 八、离线 / 断网场景

评测环境断网，但**开发阶段**通常有网。两种做法：

1. **本地模式**：`swanlab.init(mode="local")`，数据落在本地 `swanlog/`，不联网
2. **离线模式**：`swanlab.init(mode="offline")`，之后在联网机器上 `swanlab sync <run目录>` 补传

> 这两个参数在部分版本里叫法/行为不同，**使用前先确认版本**：
> `python -c "import swanlab,inspect; print(inspect.signature(swanlab.init))"`
>
> 比赛要求的是"正式评测记录完整"，所以**镜像里不应依赖联网上传**；
> 记录应在开发/调试阶段就完成，正式评测的 run 用本地模式落盘、赛后补传并说明。

---

## 九、合规与安全（会被扣分/取消资格的点）

| 规则 | 做法 |
|---|---|
| 账号不得泄露或转借 | 一人一号；不把 key 发群里、不进仓库 |
| 禁止攻击评测平台/获取隐藏集 | 不写任何探测隐藏流的代码 |
| 不得未授权外部访问 | 镜像内删除所有出网调用 |
| 记录须完整 | 统一 wrapper，禁止事后手工补 |
| 数据安全 | `data/` 不进 Git；不把官方数据发给队外 |

`.gitignore` 至少包含：

```gitignore
.swanlab/
swanlog/
data/
experiments/
*.pt
*.pth
*.ckpt
.env
```

---

## 十、常见坑（都是真实会踩的）

| 现象 | 原因 | 解决 |
|---|---|---|
| run 一直显示 running | 忘了 `swanlab.finish()` | 用 `try/finally` 包住 |
| 曲线是断的 / 只有最后一个点 | `log` 没传 `step` | 显式 `step=` |
| 5 个人看不到彼此的 run | 项目建在个人空间 | 统一到团队空间 |
| `config` 里报序列化错误 | 塞了 DataFrame / 模型对象 | 只放基础类型，复杂对象转 JSON 字符串 |
| 中文实验名乱码 | Windows 编码 | 实验名用 ASCII |
| 上传很慢 | 每步 log 太频繁 | 按 batch/epoch 聚合后再 log |
| 换了机器就传不上去 | 没登录 / 环境不同 | 每台机器各自 `swanlab login` |
| 评审打不开链接 | 项目私有 | 改公开，无痕验证 |

---

## 十一、上线检查清单（每台机器）

- [ ] `pip install swanlab` 成功
- [ ] `swanlab login` 成功（用**本人**账号）
- [ ] `python scripts/smoke.py` 能生成一条 run
- [ ] 在团队项目页能看到这条 run
- [ ] `run.config` 里的字段齐全
- [ ] 知道怎么 `finish()`（脚本里有 try/finally）
- [ ] API Key 没有出现在任何提交文件里

---

## 十二、速查

```bash
swanlab login                 # 登录
swanlab login -k <key>        # 带 key 登录
swanlab logout                # 退出
swanlab --version             # 版本
swanlab sync <run_dir>        # 补传离线 run（若版本支持）
```

```python
swanlab.init(project=..., experiment_name=..., config=..., tags=..., group=...)
swanlab.log({...}, step=...)
swanlab.finish()
run.config["lr"]              # 读取配置
swanlab.Image(path)           # 记录图片
swanlab.Text("...")           # 记录文本
```

> 需要记录图片/音频/表格、或接入 PyTorch / Lightning / HuggingFace Trainer 时，
> 看官方文档对应章节，不要凭记忆写参数。
