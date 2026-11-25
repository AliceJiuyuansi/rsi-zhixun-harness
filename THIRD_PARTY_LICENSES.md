# 第三方组件说明

本项目训练与预测代码使用 Python 标准库和 NumPy；容器另需平台提供的官方 SDK。Python/NumPy 的许可文本由相应发行包提供，本文件是归属清单，不替代原始许可证。

requirements.txt 中另有历史研究脚本使用的 scikit-learn、SciPy、PyYAML；平台镜像的实际安装以 harness.Dockerfile 和 SDK 声明为准。SwanLab 仅用于此前开发实验的实时离线记录，本版没有新增训练实验或补录记录。

vendor/contest_baselines 为赛事公开基线副本及本项目改动，使用和分发范围应遵循赛事数据包及仓库中的原始说明；本文件不额外授予其权利。SDK、官方数据、检查点和密钥不随本源码包分发。
