#!/usr/bin/env sh
# Production entrypoint for Google Cloud Run.
# Backend (FastAPI): bind 0.0.0.0:$PORT, single uvicorn process for low cold start.
# For frontend-only deploy: override CMD with streamlit run frontend/app.py --server.port=$PORT --server.address=0.0.0.0

set -e
PORT="${PORT:-8080}"

# Single uvicorn process (no gunicorn) to minimize cold start; Cloud Run scales by instance count.
exec uvicorn app.api.main:app --host 0.0.0.0 --port "$PORT"
