"""
Legal Chatbot - Request/Response Logging Middleware
Phase 4.2: Monitoring and Observability
"""

import time
import json
from typing import Callable
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp
from loguru import logger
import uuid
from datetime import datetime
from app.core.metrics import metrics_collector


class RequestResponseLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for logging HTTP requests and responses with metrics tracking"""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Generate request ID
        request_id = str(uuid.uuid4())
        
        # Start time
        start_time = time.time()
        
        # Log request
        request_body = None
        if request.method in ["POST", "PUT", "PATCH"]:
            try:
                body = await request.body()
                if body:
                    request_body = body.decode("utf-8")[:1000]  # Limit size
            except Exception:
                pass
        
        # Get client info
        client_ip = request.client.host if request.client else "unknown"
        user_agent = request.headers.get("user-agent", "unknown")
        
        logger.info(
            "Request received",
            extra={
                "request_id": request_id,
                "method": request.method,
                "url": str(request.url),
                "path": request.url.path,
                "query_params": str(request.query_params),
                "client_ip": client_ip,
                "user_agent": user_agent,
                "request_body": request_body,
                "headers": dict(request.headers),
                "type": "request",
            },
        )
        
        # Store request start time in request state for later access
        request.state.start_time = start_time
        request.state.request_id = request_id
        
        # Process request
        try:
            response = await call_next(request)
            
            # Calculate response time
            process_time = (time.time() - start_time) * 1000  # Convert to ms
            
            # Track API metrics
            endpoint_path = request.url.path
            metrics_collector.record_api_request(
                endpoint=endpoint_path,
                method=request.method,
                response_time_ms=process_time,
                status_code=response.status_code,
            )
            
            # Get response body size
            response_body_size = 0
            response_body_preview = None
            if hasattr(response, "body"):
                response_body_size = len(response.body)
                if response_body_size > 0:
                    try:
                        response_body_preview = response.body.decode("utf-8")[:500]
                    except Exception:
                        pass
            
            # Log response
            logger.info(
                "Response sent",
                extra={
                    "request_id": request_id,
                    "method": request.method,
                    "url": str(request.url),
                    "path": request.url.path,
                    "status_code": response.status_code,
                    "process_time_ms": round(process_time, 2),
                    "response_size": response_body_size,
                    "response_body_preview": response_body_preview,
                    "type": "response",
                },
            )
            
            # Add custom headers
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Process-Time"] = f"{process_time:.2f}ms"
            
            return response
            
        except Exception as e:
            # Log error
            process_time = (time.time() - start_time) * 1000
            
            # Track error metrics (500 status code for exceptions)
            endpoint_path = request.url.path
            metrics_collector.record_api_request(
                endpoint=endpoint_path,
                method=request.method,
                response_time_ms=process_time,
                status_code=500,
            )
            
            logger.error(
                "Request processing error",
                extra={
                    "request_id": request_id,
                    "method": request.method,
                    "url": str(request.url),
                    "path": request.url.path,
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "process_time_ms": round(process_time, 2),
                    "type": "error",
                },
                exc_info=True,
            )
            raise


class ErrorTrackingMiddleware(BaseHTTPMiddleware):
    """Middleware for tracking and logging errors"""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        try:
            response = await call_next(request)
            return response
        except Exception as e:
            # Log error details
            logger.error(
                "Unhandled exception",
                extra={
                    "method": request.method,
                    "url": str(request.url),
                    "path": request.url.path,
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "client_ip": request.client.host if request.client else "unknown",
                    "type": "exception",
                },
                exc_info=True,
            )
            raise

