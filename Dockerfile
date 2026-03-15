# GroundTruth — Zero-Hallucination Enterprise Knowledge Agent
# Phase 24: Optimized Docker build for Google Cloud Run

FROM python:3.11-slim AS base

# Security: non-root user
RUN groupadd -r groundtruth && useradd -r -g groundtruth -d /app -s /sbin/nologin groundtruth

WORKDIR /app

# Install deps (cached layer)
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY backend/ ./backend/
COPY frontend/ ./frontend/

WORKDIR /app/backend

# Security: non-root
USER groundtruth

ENV PORT=8080
EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/api/health')" || exit 1

# Use the new modular app.py entry point
CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
