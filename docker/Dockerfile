# Multi-stage Dockerfile for SIH-26154 Semantic Document Processing System
# Base Python 3.12 runtime with C++ runtime, Poppler, OpenCV, and PaddleOCR dependencies

FROM python:3.12-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive \
    PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True \
    PYTHONPATH=/app

# Install system dependencies for PyMuPDF, OpenCV, and PaddlePaddle
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    libspatialindex-dev \
    poppler-utils \
    libmagic1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency definition
COPY requirements.txt /app/requirements.txt

# Install Python packages
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir paddlepaddle==3.0.0 paddleocr==3.7.0 "paddlex[ocr]>=3.7.0,<3.8.0"

# Copy application source code
COPY backend/ /app/backend/

# Create uploads directories
RUN mkdir -p /app/uploads/raw /app/uploads/extracted

# Expose FastAPI application port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

# Launch FastAPI with Uvicorn
CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
