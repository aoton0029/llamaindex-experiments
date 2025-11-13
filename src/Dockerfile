FROM ghcr.io/astral-sh/uv:python3.12-trixie-slim

WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen
