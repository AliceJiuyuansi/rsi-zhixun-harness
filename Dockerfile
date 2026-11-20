# 智训引擎 · 离线自进化镜像（second build: full runtime）
#
# Design constraints (docs/04 §6.1, docs/00):
#   - runtime is OFFLINE: no outbound calls, no external API, no network client
#   - one command, no arguments: `docker run --rm <image>`
#   - always exit 0: a failed stream degrades to "keep the current model"
#
# Build needs a package index; the run does not.
#   docker build -t zhixun-rsi:0.1 .
#   docker build --build-arg PIP_INDEX_URL=https://pypi.org/simple -t zhixun-rsi:0.1 .
FROM python:3.11-slim

ARG PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple

ENV PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    RSI_OUT=/app/out \
    RSI_SWAN_MODE=disabled

WORKDIR /app

# Dependency layer first so source edits do not invalidate it.
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --index-url "${PIP_INDEX_URL}" -r /app/requirements.txt

COPY src /app/src
COPY configs /app/configs
COPY scripts/entrypoint.py /app/scripts/entrypoint.py
# Proxy feature cache baked in: self-contained, works with the disk unmounted.
COPY data/processed/xjtu /app/data/processed/xjtu
COPY data/VERSION.md /app/data/VERSION.md

RUN mkdir -p /app/out

ENTRYPOINT ["python", "/app/scripts/entrypoint.py"]
