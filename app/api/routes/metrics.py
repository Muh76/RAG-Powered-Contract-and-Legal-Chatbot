"""
Legal Chatbot - Metrics API Endpoints
Phase 4.2: Monitoring and Observability
"""

from fastapi import APIRouter, Depends
from typing import Dict, Any, Optional
from datetime import datetime
from app.core.metrics import metrics_collector, SystemMetrics
from app.auth.dependencies import require_admin
from app.auth.models import User

router = APIRouter()


@router.get("/metrics")
async def get_metrics(
    current_user: User = Depends(require_admin)
) -> Dict[str, Any]:
    """Get all metrics (requires Admin role)"""
    return {
        "summary": metrics_collector.get_summary_metrics(),
        "endpoints": metrics_collector.get_endpoint_metrics(),
        "tool_usage": metrics_collector.get_tool_usage_stats(),
        "timestamp": datetime.utcnow().isoformat(),
    }


@router.get("/metrics/summary")
async def get_summary_metrics(
    current_user: User = Depends(require_admin)
) -> Dict[str, Any]:
    """Get summary metrics (requires Admin role)"""
    return {
        **metrics_collector.get_summary_metrics(),
        "timestamp": datetime.utcnow().isoformat(),
    }


@router.get("/metrics/endpoints")
async def get_endpoint_metrics(
    endpoint: Optional[str] = None,
    current_user: User = Depends(require_admin)
) -> Dict[str, Any]:
    """Get endpoint-specific metrics (requires Admin role)"""
    return {
        "metrics": metrics_collector.get_endpoint_metrics(endpoint),
        "timestamp": datetime.utcnow().isoformat(),
    }


@router.get("/metrics/tools")
async def get_tool_metrics(
    tool_name: Optional[str] = None,
    current_user: User = Depends(require_admin)
) -> Dict[str, Any]:
    """Get tool usage statistics (requires Admin role)"""
    return {
        "metrics": metrics_collector.get_tool_usage_stats(tool_name),
        "timestamp": datetime.utcnow().isoformat(),
    }


@router.get("/metrics/system")
async def get_system_metrics(
    current_user: User = Depends(require_admin)
) -> Dict[str, Any]:
    """Get system metrics (CPU, memory, disk) - requires Admin role"""
    return SystemMetrics.get_all_metrics()


@router.post("/metrics/reset")
async def reset_metrics(
    current_user: User = Depends(require_admin)
) -> Dict[str, Any]:
    """Reset all metrics (for testing) - requires Admin role"""
    metrics_collector.reset_metrics()
    return {
        "message": "Metrics reset successfully",
        "timestamp": datetime.utcnow().isoformat(),
    }

