#!/usr/bin/env bash
docker run --rm \
    -v "$PWD":/project \
    -w /project \
    ghcr.io/astral-sh/uv:python3.12-trixie-slim \
    uv lock "$@"