#!/bin/bash
# Start FastAPI server with proper reload settings

uvicorn app.api.main:app \
  --reload \
  --reload-dir app \
  --port 8000 \
  --host 0.0.0.0