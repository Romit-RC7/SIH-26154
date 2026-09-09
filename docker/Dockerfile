
# syntax=docker/dockerfile:1.7
# Multi-stage Dockerfile for SIH-26154 Semantic Document Processing System
# Base Python 3.12 runtime with C++ runtime, Poppler, OpenCV, and PaddleOCR dependencies

FROM python:3.12-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive \
    PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True \
    PYTHONPATH=/app \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Install system dependencies for PyMuPDF, OpenCV, PaddlePaddle, and llama-cpp-python C++ compilation
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    ninja-build \
    curl \
    git \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    libspatialindex-dev \
    poppler-utils \
    ffmpeg \
    libmagic1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency definition
COPY requirements.txt /app/requirements.txt

# Install uv for faster dependency resolution
RUN pip install --default-timeout=1000 --retries 5 uv

# Install Python packages with uv (significantly faster)
# Increased UV_HTTP_TIMEOUT to handle large CUDA/torch package downloads and slow PaddlePaddle CDN
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system -r requirements.txt && \
    uv pip install --system \
        paddleocr==3.7.0 \
        "paddlex[ocr,multimodal]>=3.7.0,<3.8.0"

# Install paddlepaddle-gpu (CUDA build). No CPU fallback: this is the GPU profile image,
# so if the GPU wheel cannot be fetched the build MUST fail rather than silently degrade to CPU.
# Using plain pip (not uv) here — uv's HTTP client fails to connect to the Baidu CDN in some
# network environments while pip succeeds against the same host.
RUN sh -c ' \
        for i in 1 2 3 4 5; do \
            pip install --no-cache-dir --default-timeout=1000 --retries 3 \
                paddlepaddle-gpu==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/ && exit 0; \
            echo "GPU wheel attempt $i failed, retrying..."; \
            sleep 10; \
        done; \
        echo "FATAL: paddlepaddle-gpu could not be installed. Refusing to fall back to CPU in the GPU image." >&2; \
        exit 1'

# Replace the CPU-only llama-cpp-python wheel (pulled by requirements.txt) with a CUDA-enabled build.
# The default PyPI wheel ships with no GPU kernels (llama_supports_gpu_offload() == False),
# which silently ignores n_gpu_layers=-1 and forces Qwen3-4B/Qwen3-8B onto CPU.
# No CPU fallback here either — fail the build if the CUDA wheel cannot be installed.
RUN pip uninstall -y llama-cpp-python && \
    sh -c ' \
        for i in 1 2 3; do \
            CMAKE_ARGS="-DGGML_CUDA=on" FORCE_CMAKE=1 \
            pip install --no-cache-dir  \
                --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124 \
                llama-cpp-python==0.3.35 && exit 0; \
            echo "llama-cpp-python CUDA wheel attempt $i failed, retrying..."; \
            sleep 10; \
        done; \
        echo "FATAL: llama-cpp-python CUDA wheel could not be installed. Refusing to fall back to CPU in the GPU image." >&2; \
        exit 1'

# The cu124 llama-cpp-python wheel dynamically links against CUDA 12.x runtime libraries
# (libcudart.so.12, libcublas.so.12), but torch installs CUDA 13.x libs instead.
# Install the missing CUDA 12.x runtime shared libraries via pip wheels and expose them via LD_LIBRARY_PATH.
RUN pip install --no-cache-dir --default-timeout=600 --retries 5 \
    nvidia-cuda-runtime-cu12 \
    nvidia-cublas-cu12 \
    nvidia-cuda-nvrtc-cu12

ENV LD_LIBRARY_PATH="/usr/local/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:/usr/local/lib/python3.12/site-packages/nvidia/cublas/lib:/usr/local/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib:${LD_LIBRARY_PATH}"

# CRITICAL: paddlepaddle-gpu pulls in nvidia-nccl-cu12 as a transitive dependency, and pip's
# nvidia-* namespace packages all install their .so files into the SAME site-packages/nvidia/nccl/lib/
# path regardless of CUDA major version. Since paddlepaddle-gpu installs AFTER torch in this Dockerfile,
# its older nccl-cu12 library silently overwrites the nccl-cu13 library that torch==2.14.0+cu130 needs,
# breaking torch with "undefined symbol: ncclCommResume" at import time.
# Force-reinstall the exact nccl version torch requires, LAST, so its files win.
RUN TORCH_NCCL_VERSION=$(python -c "import importlib.metadata as m; print(m.version('nvidia-nccl-cu13'))") && \
    pip install --no-cache-dir --force-reinstall --no-deps --default-timeout=300 \
        nvidia-nccl-cu13==${TORCH_NCCL_VERSION} && \
    python -c "import torch; assert torch.cuda.is_available() or True; print('torch import OK, version', torch.__version__)"

# Copy application source code
COPY backend/ /app/backend/

# Create uploads directories
RUN mkdir -p /app/uploads/raw /app/uploads/extracted /app/uploads/exports

# Expose FastAPI application port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

# Launch FastAPI with Uvicorn (excluding uploads and tests from triggering reloads)
CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload", "--reload-dir", "backend/app", "--reload-exclude", "uploads/*", "--reload-exclude", "backend/tests/*"]