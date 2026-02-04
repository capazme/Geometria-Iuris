# CLS Pipeline v2.0 Dockerfile
# Multi-stage build optimized for 16GB RAM

# ==============================================================================
# Stage 1: Builder
# ==============================================================================
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ==============================================================================
# Stage 2: Runtime
# ==============================================================================
FROM python:3.11-slim AS runtime

# Labels
LABEL maintainer="CLS Pipeline"
LABEL version="2.0.0"
LABEL description="Cross-Lingual Semantics Analysis Pipeline"

WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy application code
COPY config.yaml .
COPY pyproject.toml .
COPY src/ ./src/

# Create directories for volumes
RUN mkdir -p /app/data /app/output /app/cache /app/models

# Set Python path
ENV PYTHONPATH="/app:$PYTHONPATH"

# Environment variables for PyTorch
ENV PYTORCH_ENABLE_MPS_FALLBACK=1
ENV TOKENIZERS_PARALLELISM=false

# HuggingFace cache directory
ENV HF_HOME=/app/models
ENV TRANSFORMERS_CACHE=/app/models

# Default command
CMD ["python", "-m", "src.cli", "run"]

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import torch; print(torch.__version__)" || exit 1
