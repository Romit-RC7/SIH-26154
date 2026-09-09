

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
# Increased timeouts and connection pool for large CUDA/torch package downloads (2GB+)
RUN --mount=type=cache,target=/root/.cache/uv \
    UV_HTTP_TIMEOUT=600 \
    UV_HTTP_CONNECT_TIMEOUT=60 \
    uv pip install --system -r requirements.txt && \
    UV_HTTP_TIMEOUT=600 \
    UV_HTTP_CONNECT_TIMEOUT=60 \
    uv pip install --system \
    paddlepaddle==3.0.0 paddleocr==3.7.0 "paddlex[ocr,multimodal]>=3.7.0,<3.8.0"

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