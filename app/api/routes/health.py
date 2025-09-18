# Legal Chatbot - Health Check Route

from fastapi import APIRouter
from datetime import datetime
from app.models.schemas import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.utcnow(),
        version="1.0.0",
        services={
            "api": "healthy",
            "database": "healthy",
            "vector_db": "healthy"
        }
    )
