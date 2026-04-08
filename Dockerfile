# Multi-stage Dockerfile for clinical-triage-agent
# Optimized for Hugging Face Spaces (Standard port 7860)

ARG BASE_IMAGE=python:3.10-slim
FROM ${BASE_IMAGE} AS builder

WORKDIR /app

# Install build essentials
RUN apt-get update && \
    apt-get install -y --no-install-recommends git curl build-essential && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies from requirements.txt
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# Final runtime image
FROM python:3.10-slim

WORKDIR /app

# Install runtime system deps
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application code from current directory (root of clinical-triage-agent)
COPY . .

# Add current directory to PYTHONPATH to ensure relative imports work
ENV PYTHONPATH="/app:$PYTHONPATH"

# Expose port 7860 (HF Space standard)
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# Run the OpenEnv server entrypoint (matches openenv.yaml)
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
