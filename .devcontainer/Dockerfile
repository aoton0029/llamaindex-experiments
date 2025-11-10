FROM python:3.12-slim

WORKDIR /workspace

RUN apt-get update \
	&& apt-get install -y --no-install-recommends \
		build-essential \
		git \
		wget \
		ffmpeg \
		libsndfile1 \
		libgl1 \
	&& rm -rf /var/lib/apt/lists/*

COPY .devcontainer/requirements.txt /tmp/requirements.txt
RUN python -m pip install --upgrade pip setuptools wheel \
	&& pip install --no-cache-dir -r /tmp/requirements.txt || true
