# Production Dockerfile for Google Cloud Run
# Build context: ensure data/ (FAISS indices) is included; secrets via env (OPENAI_API_KEY, etc.)

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

WORKDIR /app

# System deps for psycopg2/pgvector; clean apt cache in same layer
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy entire project (app, data/, scripts, etc.); FAISS index in data/ is included
COPY . .

EXPOSE 8080

# Cloud Run: OPENAI_API_KEY and other secrets from environment
CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8080"]
