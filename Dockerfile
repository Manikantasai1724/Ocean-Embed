# syntax=docker/dockerfile:1
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/src

WORKDIR /app

# Install system dependencies (build-essential, git, libnetcdf-dev)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    libnetcdf-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency requirements
COPY requirements.txt pyproject.toml ./

# Install CPU-only PyTorch and python packages
RUN pip install --upgrade pip && \
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install -r requirements.txt

# Copy source code, configs, tests
COPY src/ /app/src/
COPY configs/ /app/configs/
COPY tests/ /app/tests/
COPY README.md /app/

# Create data directories
RUN mkdir -p /app/data/raw /app/data/processed /app/data/interim /app/checkpoints /app/logs

# Default command: run unit tests
CMD ["python", "-m", "unittest", "discover", "-s", "tests"]
