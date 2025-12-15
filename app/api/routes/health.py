# Legal Chatbot - Health Check Route
# Phase 4.2: Enhanced Health Checks with Dependency Monitoring

from fastapi import APIRouter, HTTPException
from datetime import datetime
from typing import Dict, Any
from app.models.schemas import HealthResponse
from app.core.health_checker import health_checker
from app.core.metrics import SystemMetrics

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Enhanced health check endpoint with dependency monitoring"""
    # Check all dependencies
    dependencies = await health_checker.check_all_dependencies()
    
    # Determine overall status
    all_healthy = all(
        dep.get("status") == "healthy" or dep.get("status") == "unknown"
        for dep in dependencies.values()
    )
    
    # Format service statuses
    services = {
        "api": "healthy",
        "database": dependencies["database"].get("status", "unknown"),
        "redis": dependencies["redis"].get("status", "unknown"),
        "vector_db": dependencies["vector_store"].get("status", "unknown"),
        "llm_api": dependencies["llm_api"].get("status", "unknown"),
    }
    
    return HealthResponse(
        status="healthy" if all_healthy else "degraded",
        timestamp=datetime.utcnow(),
        version="1.0.0",
        services=services,
    )


@router.get("/health/detailed")
async def detailed_health_check() -> Dict[str, Any]:
    """Detailed health check with system metrics"""
    # Check all dependencies
    dependencies = await health_checker.check_all_dependencies()
    
    # Get system metrics
    system_metrics = SystemMetrics.get_all_metrics()
    
    # Determine overall status
    critical_services = ["database", "vector_store", "llm_api"]
    critical_healthy = all(
        dependencies[service].get("status") == "healthy"
        for service in critical_services
    )
    
    status = "healthy" if critical_healthy else "degraded"
    
    return {
        "status": status,
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0",
        "dependencies": dependencies,
        "system_metrics": system_metrics,
    }


@router.get("/health/live")
async def liveness_probe():
    """Kubernetes liveness probe"""
    return {"status": "alive", "timestamp": datetime.utcnow().isoformat()}


@router.get("/health/ready")
async def readiness_probe():
    """Kubernetes readiness probe - only requires database (FAISS is in-memory)"""
    dependencies = await health_checker.check_all_dependencies()
    
    # Check if critical services are ready
    # Only database is critical - FAISS is in-memory and loads on-demand
    critical_services = ["database"]
    ready = all(
        dependencies[service].get("status") == "healthy"
        for service in critical_services
    )
    
    if ready:
        return {"status": "ready", "timestamp": datetime.utcnow().isoformat()}
    else:
        raise HTTPException(status_code=503, detail="Service not ready")
