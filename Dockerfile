# Production Dockerfile for Google Cloud Run (FastAPI default; override CMD for Streamlit)
# Build context must exclude secrets (e.g. .dockerignore with .env, *.pem).

FROM python:3.11-slim

# Prevent Python from writing bytecode and buffer stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Cloud Run sets PORT (default 8080)
ENV PORT=8080

WORKDIR /app

# System deps only (no dev packages); clean apt cache in same layer
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Layer 1: install dependencies for better cache reuse
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Layer 2: application code (changes often)
COPY . .

# Non-root user for runtime
RUN groupadd --gid 1000 app \
    && useradd --uid 1000 --gid app --shell /bin/bash --create-home app \
    && mkdir -p /app/logs \
    && chown -R app:app /app

USER app

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://127.0.0.1:8080/health || exit 1

# FastAPI default; for Streamlit: CMD ["streamlit", "run", "frontend/app.py", "--server.port=8080", "--server.address=0.0.0.0"]
CMD ["sh", "-c", "exec uvicorn app.api.main:app --host 0.0.0.0 --port ${PORT:-8080}"]
