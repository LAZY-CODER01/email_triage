# Email Triage Environment — Production Dockerfile
#
# Build:  docker build -t email-triage-env .
# Run:    docker run -p 7860:7860 email-triage-env

FROM python:3.11-slim

# --------------------------------------------------------------------------
# System deps
# --------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
    && rm -rf /var/lib/apt/lists/*

# --------------------------------------------------------------------------
# Working directory
# --------------------------------------------------------------------------
WORKDIR /app

# --------------------------------------------------------------------------
# Install Python dependencies (separate layer for caching)
# --------------------------------------------------------------------------
COPY pyproject.toml .

# Upgrade pip and install from pyproject.toml
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir .

# --------------------------------------------------------------------------
# Copy application source
# --------------------------------------------------------------------------
COPY . .

# --------------------------------------------------------------------------
# Runtime configuration
# --------------------------------------------------------------------------
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# HF Spaces convention: port 7860
EXPOSE 7860

# --------------------------------------------------------------------------
# Health check — pings the /reset endpoint every 30 seconds
# --------------------------------------------------------------------------
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -sf -X POST http://localhost:7860/reset \
        -H "Content-Type: application/json" \
        -d '{}' || exit 1

# --------------------------------------------------------------------------
# Start
# --------------------------------------------------------------------------
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]