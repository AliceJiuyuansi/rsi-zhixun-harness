# 智训引擎 · Harness 镜像（平台提交用）
#
# 与 zhixun-rsi:0.1 的区别：这个镜像不跑我们自己的实验循环，而是实现平台的
# 三合一入口 describe / train / predict，靠 SDK 与宿主通信。
# 平台调用方式：`python3 /app/harness.py describe|train|predict`
#
# v0.2（2026-11-23）：platform entry now uses the v2 round controller
# (cross-round drift reference, history read-back, model verdict, stall rule,
#  and the Gatekeeper's whitelist/budget/bounds/kill-switch).
#
#   docker build -f harness.Dockerfile -t zhixun-rsi-harness:0.6 .
#   docker run --rm --network none zhixun-rsi-harness:0.6 describe
FROM python:3.11-slim

ARG RSI_RELEASE=0.6
LABEL org.opencontainers.image.title="zhixun-rsi-harness" \
      org.opencontainers.image.version="${RSI_RELEASE}" \
      org.opencontainers.image.description="Contest 9 harness: offline continuation + drift-gated updates"

ARG PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
ARG RSI_C_CALIBRATION=0

ENV PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    CONTEST_APP=/app \
    CONTEST_SUBMISSION=contest_submission \
    PYTHONPATH=/app \
    RSI_SWAN_MODE=disabled \
    RSI_RELEASE=${RSI_RELEASE} \
    RSI_C_CALIBRATION=${RSI_C_CALIBRATION}

# 1) SDK（官方 1.0.4，随平台下载包，不打网络补丁）
COPY _sdk/contest-harness-sdk-1.0.4 /opt/contest-harness-sdk
RUN pip install --no-cache-dir --index-url "${PIP_INDEX_URL}" numpy \
    && pip install --no-cache-dir /opt/contest-harness-sdk

# 2) 我们自己的代码：官方基线副本 + 续训适配器 + 策略 + 入口
COPY vendor /app/vendor
COPY src /app/src
COPY harness/harness.py /app/harness.py
COPY harness/contest_submission.py /app/contest_submission.py

# 3) 平台约定的挂载点（运行时不写 /app）
RUN mkdir -p /app /data /previous /state /workspace && chmod 1777 /workspace /tmp

WORKDIR /app
USER 65534:65534

ENTRYPOINT ["python3", "/app/harness.py"]
CMD ["describe"]
