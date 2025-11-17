# Phase 4.2: Monitoring and Observability - API Usage Guide

## Overview

This guide explains how to use the monitoring and observability features implemented in Phase 4.2.

## API Endpoints

### Health Check Endpoints

#### 1. Basic Health Check
```bash
GET /api/v1/health
```

Returns overall service health with dependency status.

**Response**:
```json
{
  "status": "healthy",
  "timestamp": "2025-11-17T12:00:00",
  "version": "1.0.0",
  "services": {
    "api": "healthy",
    "database": "healthy",
    "redis": "healthy",
    "vector_db": "healthy",
    "llm_api": "healthy"
  }
}
```

#### 2. Detailed Health Check
```bash
GET /api/v1/health/detailed
```

Returns comprehensive health information including dependency details and system metrics.

**Response**:
```json
{
  "status": "healthy",
  "timestamp": "2025-11-17T12:00:00",
  "version": "1.0.0",
  "dependencies": {
    "database": {
      "status": "healthy",
      "message": "Database connection successful",
      "response_time_ms": 2.5
    },
    "redis": {
      "status": "healthy",
      "message": "Redis connection successful",
      "response_time_ms": 1.2
    },
    "vector_store": {
      "status": "healthy",
      "message": "Vector store connection successful",
      "response_time_ms": 3.1,
      "collections_count": 1
    },
    "llm_api": {
      "status": "healthy",
      "message": "OpenAI API connection successful",
      "response_time_ms": 150.5
    }
  },
  "system_metrics": {
    "cpu": {
      "cpu_percent": 15.2,
      "cpu_count": 8,
      "cpu_freq_mhz": 2400.0
    },
    "memory": {
      "memory_total_gb": 16.0,
      "memory_available_gb": 8.5,
      "memory_used_gb": 7.5,
      "memory_percent": 46.9
    },
    "disk": {
      "disk_total_gb": 500.0,
      "disk_used_gb": 200.0,
      "disk_free_gb": 300.0,
      "disk_percent": 40.0
    },
    "timestamp": "2025-11-17T12:00:00"
  }
}
```

#### 3. Liveness Probe (Kubernetes)
```bash
GET /api/v1/health/live
```

Simple endpoint for Kubernetes liveness probe.

**Response**:
```json
{
  "status": "alive",
  "timestamp": "2025-11-17T12:00:00"
}
```

#### 4. Readiness Probe (Kubernetes)
```bash
GET /api/v1/health/ready
```

Checks if critical services are ready. Returns 503 if not ready.

**Response** (200 OK):
```json
{
  "status": "ready",
  "timestamp": "2025-11-17T12:00:00"
}
```

**Response** (503 Service Unavailable):
```json
{
  "detail": "Service not ready"
}
```

### Metrics Endpoints

#### 1. All Metrics
```bash
GET /api/v1/metrics
```

Returns all metrics including summary, endpoint metrics, and tool usage.

**Response**:
```json
{
  "summary": {
    "uptime_seconds": 3600.0,
    "total_requests": 150,
    "total_errors": 5,
    "overall_error_rate_percent": 3.33,
    "avg_response_time_ms": 125.5,
    "total_tool_usage": 25,
    "endpoints_tracked": 8,
    "tools_tracked": 3
  },
  "endpoints": {
    "POST:/api/v1/chat": {
      "endpoint": "/api/v1/chat",
      "method": "POST",
      "request_count": 50,
      "error_count": 2,
      "avg_response_time_ms": 150.2,
      "min_response_time_ms": 80.5,
      "max_response_time_ms": 450.0,
      "error_rate_percent": 4.0
    },
    "POST:/api/v1/agentic-chat": {
      "endpoint": "/api/v1/agentic-chat",
      "method": "POST",
      "request_count": 30,
      "error_count": 1,
      "avg_response_time_ms": 2500.0,
      "min_response_time_ms": 1200.0,
      "max_response_time_ms": 5000.0,
      "error_rate_percent": 3.33
    }
  },
  "tool_usage": {
    "search_legal_documents": {
      "tool_name": "search_legal_documents",
      "usage_count": 20,
      "success_count": 19,
      "failure_count": 1,
      "avg_execution_time_ms": 150.5,
      "success_rate_percent": 95.0
    },
    "get_specific_statute": {
      "tool_name": "get_specific_statute",
      "usage_count": 5,
      "success_count": 5,
      "failure_count": 0,
      "avg_execution_time_ms": 120.0,
      "success_rate_percent": 100.0
    }
  },
  "timestamp": "2025-11-17T12:00:00"
}
```

#### 2. Summary Metrics
```bash
GET /api/v1/metrics/summary
```

Returns high-level summary metrics.

**Response**:
```json
{
  "uptime_seconds": 3600.0,
  "total_requests": 150,
  "total_errors": 5,
  "overall_error_rate_percent": 3.33,
  "avg_response_time_ms": 125.5,
  "total_tool_usage": 25,
  "endpoints_tracked": 8,
  "tools_tracked": 3,
  "timestamp": "2025-11-17T12:00:00"
}
```

#### 3. Endpoint Metrics
```bash
GET /api/v1/metrics/endpoints?endpoint=/api/v1/chat
```

Returns metrics for specific endpoint(s).

**Without parameter**: Returns all endpoint metrics
**With endpoint parameter**: Returns metrics for specific endpoint

**Response**:
```json
{
  "metrics": {
    "POST:/api/v1/chat": {
      "endpoint": "/api/v1/chat",
      "method": "POST",
      "request_count": 50,
      "error_count": 2,
      "avg_response_time_ms": 150.2,
      "min_response_time_ms": 80.5,
      "max_response_time_ms": 450.0,
      "error_rate_percent": 4.0
    }
  },
  "timestamp": "2025-11-17T12:00:00"
}
```

#### 4. Tool Usage Statistics
```bash
GET /api/v1/metrics/tools?tool_name=search_legal_documents
```

Returns tool usage statistics.

**Without parameter**: Returns all tool statistics
**With tool_name parameter**: Returns statistics for specific tool

**Response**:
```json
{
  "metrics": {
    "search_legal_documents": {
      "tool_name": "search_legal_documents",
      "usage_count": 20,
      "success_count": 19,
      "failure_count": 1,
      "avg_execution_time_ms": 150.5,
      "success_rate_percent": 95.0
    }
  },
  "timestamp": "2025-11-17T12:00:00"
}
```

#### 5. System Metrics
```bash
GET /api/v1/metrics/system
```

Returns current system metrics (CPU, memory, disk).

**Response**:
```json
{
  "cpu": {
    "cpu_percent": 15.2,
    "cpu_count": 8,
    "cpu_freq_mhz": 2400.0
  },
  "memory": {
    "memory_total_gb": 16.0,
    "memory_available_gb": 8.5,
    "memory_used_gb": 7.5,
    "memory_percent": 46.9
  },
  "disk": {
    "disk_total_gb": 500.0,
    "disk_used_gb": 200.0,
    "disk_free_gb": 300.0,
    "disk_percent": 40.0
  },
  "timestamp": "2025-11-17T12:00:00"
}
```

#### 6. Reset Metrics
```bash
POST /api/v1/metrics/reset
```

Resets all metrics (for testing).

**Response**:
```json
{
  "message": "Metrics reset successfully",
  "timestamp": "2025-11-17T12:00:00"
}
```

## Logging

### Log Format

When `LOG_FORMAT=json`, logs are written in JSON format:

```json
{
  "timestamp": "2025-11-17T12:00:00.123456",
  "level": "INFO",
  "message": "Request received",
  "module": "app.core.middleware",
  "function": "dispatch",
  "line": 42,
  "file": "/path/to/app/core/middleware.py",
  "request_id": "550e8400-e29b-41d4-a716-446655440000",
  "method": "POST",
  "path": "/api/v1/chat",
  "type": "request"
}
```

### Log Levels

- **DEBUG**: Detailed debugging information
- **INFO**: General informational messages (requests, responses)
- **WARNING**: Warning messages for potential issues
- **ERROR**: Error messages with full exception details

### Log Files

- **Location**: `logs/legal_chatbot.log`
- **Rotation**: 10 MB
- **Retention**: 7 days
- **Compression**: ZIP

### Viewing Logs

```bash
# View logs in real-time
tail -f logs/legal_chatbot.log | jq

# View errors only
grep '"level":"ERROR"' logs/legal_chatbot.log | jq

# View requests only
grep '"type":"request"' logs/legal_chatbot.log | jq

# View responses only
grep '"type":"response"' logs/legal_chatbot.log | jq
```

## Request/Response Tracking

All requests and responses are automatically logged with:
- **Request ID**: Unique UUID for tracking
- **Request Details**: Method, URL, path, query params, body (limited)
- **Response Details**: Status code, process time, response size
- **Custom Headers**: `X-Request-ID` and `X-Process-Time` added to responses

### Example Log Entry

**Request**:
```json
{
  "timestamp": "2025-11-17T12:00:00",
  "level": "INFO",
  "message": "Request received",
  "request_id": "550e8400-e29b-41d4-a716-446655440000",
  "method": "POST",
  "path": "/api/v1/chat",
  "type": "request"
}
```

**Response**:
```json
{
  "timestamp": "2025-11-17T12:00:05",
  "level": "INFO",
  "message": "Response sent",
  "request_id": "550e8400-e29b-41d4-a716-446655440000",
  "method": "POST",
  "path": "/api/v1/chat",
  "status_code": 200,
  "process_time_ms": 5123.45,
  "type": "response"
}
```

## Error Tracking

All errors are automatically logged with:
- **Error Type**: Exception class name
- **Error Message**: Exception message
- **Traceback**: Full stack trace
- **Request Context**: Method, URL, path
- **Process Time**: Time before error occurred

### Example Error Log

```json
{
  "timestamp": "2025-11-17T12:00:00",
  "level": "ERROR",
  "message": "Request processing error",
  "error": "Connection refused",
  "error_type": "ConnectionError",
  "type": "error",
  "exception": {
    "type": "ConnectionError",
    "value": "Connection refused",
    "traceback": "..."
  }
}
```

## Tool Usage Tracking

Tool usage is automatically tracked in agentic chat:
- **Tool Name**: Name of the tool used
- **Execution Time**: Time taken to execute (milliseconds)
- **Success/Failure**: Whether tool execution succeeded
- **Usage Count**: Total number of times tool was used
- **Success Rate**: Percentage of successful executions

### Viewing Tool Usage

```bash
# Get all tool usage statistics
curl http://localhost:8000/api/v1/metrics/tools

# Get specific tool statistics
curl http://localhost:8000/api/v1/metrics/tools?tool_name=search_legal_documents
```

## Best Practices

### Health Checks
- Use `/health/live` for Kubernetes liveness probes
- Use `/health/ready` for Kubernetes readiness probes
- Use `/health/detailed` for comprehensive health monitoring
- Check health periodically (every 30-60 seconds)

### Metrics
- Monitor `/metrics/summary` for overall system health
- Track `/metrics/endpoints` for API performance
- Monitor `/metrics/tools` for agentic chat tool usage
- Use `/metrics/system` for resource monitoring

### Logging
- Set `LOG_LEVEL=INFO` for production
- Set `LOG_LEVEL=DEBUG` for debugging
- Use `LOG_FORMAT=json` for structured logging
- Monitor error logs regularly

### Alerts
- Set up alerts for error rates > 5%
- Set up alerts for response times > 5 seconds
- Set up alerts for system resource usage > 80%
- Monitor tool success rates

## Kubernetes Integration

### Liveness Probe
```yaml
livenessProbe:
  httpGet:
    path: /api/v1/health/live
    port: 8000
  initialDelaySeconds: 30
  periodSeconds: 10
```

### Readiness Probe
```yaml
readinessProbe:
  httpGet:
    path: /api/v1/health/ready
    port: 8000
  initialDelaySeconds: 10
  periodSeconds: 5
```

## Monitoring Dashboard

For visualization, you can:
1. Export metrics to Prometheus format
2. Create Grafana dashboards
3. Set up alerts based on metrics
4. Visualize logs using log aggregation tools (ELK, Splunk, etc.)

## Conclusion

Phase 4.2 provides comprehensive monitoring and observability:
- ✅ Structured JSON logging
- ✅ Request/response logging
- ✅ Error tracking
- ✅ Health checks with dependency monitoring
- ✅ System metrics (CPU, memory, disk)
- ✅ API metrics (response times, error rates, request volumes)
- ✅ Tool usage statistics

**Status**: ✅ **READY FOR PRODUCTION MONITORING**

