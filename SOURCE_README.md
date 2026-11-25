# 智训引擎 v16 源码候选

v16 修复两份 v15 的模块与发布一致性。训练/预测策略沿用已验证的我方 v15；不包含新平台成绩、镜像 tar、数据、权重、SDK 或密钥。

- 比较依据与结果：`reports/v16_比较与交付.md`
- 平台入口：`harness/contest_submission.py`
- 发布预检：`scripts/check_release.py`；文件清单：`release_manifest.json`
- 安全导出：`scripts/build_source_bundle.py`（ZIP 解压后预检及完整单测）
- 92 项测试；SwanLab 历史记录不冒记为新版本实验。

先运行 `py -3.12 scripts/check_release.py --root .`，再运行 `py -3.12 -m unittest discover -s tests -v`。
预检通过但提示缺 SDK 时，只代表源码检查通过；完整协议仍需官方 SDK。

默认版本 0.16，C 为 res_mlp_v2；镜像构建仍需 `_sdk/contest-harness-sdk-1.0.4`。
Dockerfile 构建后会执行预检；实际镜像/断网协议未在当前机器完成验收。

训练代码仅依赖标准库和 NumPy；官方 SDK 由平台另行提供，其依赖以 SDK 清单为准。requirements.txt 还列有历史研究脚本依赖，不表示镜像安装了这些组件。源码整理参考队友的说明结构，算法仍来自完整我方 v15。
