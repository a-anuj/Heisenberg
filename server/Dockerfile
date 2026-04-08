# Multi-stage Dockerfile for clinical-triage-agent
# Follows the same pattern as the reference test_env environment.

ARG BASE_IMAGE=python:3.10-slim
FROM ${BASE_IMAGE} AS builder

WORKDIR /app

# Install build essentials
RUN apt-get update && \
    apt-get install -y --no-install-recommends git curl build-essential && \
    rm -rf /var/lib/apt/lists/*

# Install uv for fast dependency management
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    mv /root/.local/bin/uv /usr/local/bin/uv && \
    mv /root/.local/bin/uvx /usr/local/bin/uvx

# Copy environment code
COPY . /app/env

WORKDIR /app/env

# Install all dependencies into a virtual environment
RUN --mount=type=cache,target=/root/.cache/uv \
    uv venv /app/.venv && \
    uv pip install --python /app/.venv/bin/python \
        "openenv-core[core]>=0.2.2" \
        "fastapi>=0.115.0" \
        "uvicorn[standard]>=0.24.0" \
        "faker>=25.0.0" \
        "pydantic>=2.0.0" \
        -e .

# ---------------------------------------------------------------------------
# Final runtime image
# ---------------------------------------------------------------------------
FROM python:3.10-slim

WORKDIR /app

# Install runtime system deps
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

# Copy virtual environment and application code
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/env /app/env

# Add venv to PATH
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app/env:$PYTHONPATH"

# Expose port 7860 (HF Space standard)
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# Run the production server on port 7860
CMD ["sh", "-c", "cd /app/env && uvicorn server.app:app --host 0.0.0.0 --port 7860"]
