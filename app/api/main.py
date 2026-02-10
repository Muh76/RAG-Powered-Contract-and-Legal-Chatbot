# Legal Chatbot - Main Application Entry Point

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.middleware.trustedhost import TrustedHostMiddleware
import uvicorn
import os
from contextlib import asynccontextmanager
import logging

from app.core.config import settings, _validate_embedding_config
from app.api.routes import chat, health, documents
from app.core.logging import setup_logging

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events"""
    # Startup
    setup_logging()
    _validate_embedding_config()
    from app.api.routes import chat as chat_routes
    chat_routes.init_chat_services()
    logger.info("✅ Application startup complete (RAG and Guardrails initialized)")
    yield
    
    # Shutdown
    logger.info("Shutting down...")
    pass


# Create FastAPI application
app = FastAPI(
    title="Legal Chatbot API",
    description="AI-Powered Legal Assistant with RAG. **Note:** Most endpoints require authentication. Click the 'Authorize' button (lock icon) in Swagger UI and provide your Bearer token.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

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
    """Root endpoint"""
    return {
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