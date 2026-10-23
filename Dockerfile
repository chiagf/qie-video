# qie-video - self-contained GPU container for Wan2.2 I2V.
#
# Build (on a machine WITH Docker, e.g. GitHub Actions or an NVIDIA box):
#   docker build -t qie-video .
#
# Run (needs NVIDIA Container Toolkit / a GPU host, e.g. Vast or RunPod):
#   docker run --gpus all -p 7860:7860 -v qie-data:/data -e HF_TOKEN=hf_xxx qie-video
#   # optional: -e QIE_AUTH=user:pass   (HTTP Basic gate over every route, for open ports)
#   # optional: -e QIE_WAN_OFFLOAD=model_cpu   (headroom on a tight card)
#
# Weights are NOT baked: the first boot pulls ~45 GB into /data/hf (Wan2.2 fp8 experts +
# official pipeline + Lightning LoRA, plus the ~22 MB RIFE archive). Give the instance
# 60 GB of disk.
# syntax=docker/dockerfile:1.7
FROM python:3.10-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        libgl1 \
        libglib2.0-0 \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# The engine is a git submodule with its own pyproject.toml; install it so the package lands in
# site-packages under its real name rather than nesting a second `qie_engine` directory.
COPY vendor/qie-engine ./vendor/qie-engine
RUN pip install --no-cache-dir ./vendor/qie-engine

COPY app.py config.py prompting.py store.py video_ui.py README.md ./

ENV PYTHONUNBUFFERED=1 \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860 \
    HF_HOME=/data/hf \
    QIE_OUT_DIR=/data/out \
    HF_XET_HIGH_PERFORMANCE=1

EXPOSE 7860

CMD ["python", "app.py"]
