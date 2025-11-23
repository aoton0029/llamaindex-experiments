FROM ghcr.io/astral-sh/uv:python3.12-trixie-slim

WORKDIR /app

RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y \
    gcc \
    wget \
    git \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# COPY download_packages.py .
RUN uv venv .venv
RUN uv pip install -r requirements.txt
# RUN uv run ./download_packages.py

ENV TIKTOKEN_CACHE_DIR=/root/.cache/tiktoken
