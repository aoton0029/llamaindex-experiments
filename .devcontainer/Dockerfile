# Build stage
FROM ghcr.io/astral-sh/uv:python3.12-trixie-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and create virtual environment
COPY requirements.txt .
RUN uv venv .venv
RUN uv pip install -r requirements.txt

# Runtime stage
FROM python:3.12-slim-trixie

WORKDIR /app

# Install only runtime dependencies
RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y \
    wget \
    git \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /app/.venv /app/.venv

ENV PATH="/app/.venv/bin:$PATH"
ENV TIKTOKEN_CACHE_DIR=/root/.cache/tiktoken