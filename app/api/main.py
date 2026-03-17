# Legal Chatbot - Main Application Entry Point
# Emit immediately so Cloud Run logs show something even if we crash during import
print("legal-chatbot-api: process starting", flush=True)

import asyncio
import logging
import os
import sys
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware

from app.api.routes import chat, documents, health
from app.core.config import (
    _validate_embedding_config,
    settings,
    validate_required_config,
)
from app.core.logging import setup_logging

logger = logging.getLogger(__name__)
print("legal-chatbot-api: imports done", flush=True)

# Background init (run_in_executor): server binds to PORT immediately for Cloud Run
_init_future: asyncio.Future | None = None


def _is_demo_mode() -> bool:
    """True if DEMO_MODE is set (env or settings). Cloud Run sets DEMO_MODE=true as string."""
    if getattr(settings, "DEMO_MODE", False):
        return True
    return os.getenv("DEMO_MODE", "").strip().lower() in ("true", "1", "yes")


def _run_startup_sync():
    """Run in thread: logging, validation (if not DEMO), RAG/Guardrails init. Never blocks server bind."""
    try:
        setup_logging()
    except Exception as e:
        print(f"legal-chatbot-api: setup_logging failed: {e}", flush=True)
    if _is_demo_mode():
        try:
            import logging as _log
            _log.getLogger(__name__).info("Running in DEMO_MODE - database disabled")
        except Exception:
            pass
    else:
        try:
            validate_required_config()
            _validate_embedding_config()
        except Exception as e:
            print(f"legal-chatbot-api: config validation failed: {e}", flush=True)
            raise
    from app.api.routes import chat as chat_routes
    try:
        chat_routes.init_chat_services()
        logger.info("✅ Background init complete (RAG and Guardrails initialized)")
    except Exception as e:
        logger.exception("❌ Background init failed: %s", e)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Yield immediately so the server binds to PORT; all startup runs in a background thread."""
    global _init_future
    print("legal-chatbot-api: lifespan yielding so server can bind", flush=True)
    loop = asyncio.get_event_loop()
    _init_future = loop.run_in_executor(None, _run_startup_sync)
    yield

    if _init_future and not _init_future.done():
        _init_future.cancel()
        try:
            await _init_future
        except asyncio.CancelledError:
            pass
    logger.info("Shutting down...")


# Create FastAPI application
app = FastAPI(
    title="Legal Chatbot API",
    description="AI-Powered Legal Assistant with RAG. **Note:** Most endpoints require authentication. Click the 'Authorize' button (lock icon) in Swagger UI and provide your Bearer token.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)
print("legal-chatbot-api: FastAPI app created", flush=True)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=settings.CORS_METHODS,
    allow_headers=settings.CORS_HEADERS,
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["localhost", "127.0.0.1", "testserver", "*"]
)

# Phase 4.2: Add monitoring middleware
from app.core.middleware import RequestResponseLoggingMiddleware, ErrorTrackingMiddleware

app.add_middleware(ErrorTrackingMiddleware)
app.add_middleware(RequestResponseLoggingMiddleware)

# Include routers
from app.api.routes import health, chat, documents, search, agentic_chat, metrics, auth, debug

app.include_router(health.router, prefix="/api/v1", tags=["health"])
app.include_router(auth.router, prefix="/api/v1", tags=["authentication"])
app.include_router(chat.router, prefix="/api/v1", tags=["chat"])
app.include_router(documents.router, prefix="/api/v1", tags=["documents"])
app.include_router(search.router, prefix="/api/v1", tags=["search"])
app.include_router(agentic_chat.router, prefix="/api/v1", tags=["agentic-chat"])
app.include_router(metrics.router, prefix="/api/v1", tags=["metrics"])
app.include_router(debug.router, prefix="/api/v1")


@app.get("/")
async def root():
    """Root endpoint; includes status for simple health checks."""
    return {
        "status": "running",
        "message": "Legal Chatbot API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/v1/health"
    }


@app.get("/health")
async def health_root():
    """Cloud Run / liveness: return 200 OK. Use /api/v1/health for detailed status."""
    return {"status": "ok"}


if __name__ == "__main__":
    port = int(os.environ.get("PORT", settings.API_PORT))
    uvicorn.run(
        "app.api.main:app",
        host=settings.API_HOST,
        port=port,
        reload=settings.API_RELOAD,
        workers=settings.API_WORKERS,
    )