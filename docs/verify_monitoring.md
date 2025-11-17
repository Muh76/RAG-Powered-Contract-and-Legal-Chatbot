# How to Verify Phase 4.2: Monitoring and Observability

This guide explains how to verify that all monitoring and observability features are implemented and working seamlessly.

## Quick Verification Steps

### 1. Run Unit Tests

```bash
# Activate virtual environment
source venv/bin/activate

# Run monitoring unit tests
pytest tests/unit/test_monitoring.py -v
```

**Expected Result**: All 20 tests should pass ✅

### 2. Verify Components Import Successfully

```bash
# Test imports
python -c "from app.core.metrics import metrics_collector, SystemMetrics; print('✅ Metrics OK')"
python -c "from app.core.health_checker import health_checker; print('✅ Health Checker OK')"
python -c "from app.core.middleware import RequestResponseLoggingMiddleware; print('✅ Middleware OK')"
python -c "from app.core.logging import setup_logging; print('✅ Logging OK')"
python -c "from app.api.routes import metrics, health; print('✅ Routes OK')"
```

### 3. Start the Server and Test Endpoints

```bash
# Start the server
uvicorn app.api.main:app --reload --port 8000
```

In another terminal:

```bash
# Test health endpoints
curl http://localhost:8000/api/v1/health
curl http://localhost:8000/api/v1/health/detailed
curl http://localhost:8000/api/v1/health/live
curl http://localhost:8000/api/v1/health/ready

# Test metrics endpoints
curl http://localhost:8000/api/v1/metrics
curl http://localhost:8000/api/v1/metrics/summary
curl http://localhost:8000/api/v1/metrics/endpoints
curl http://localhost:8000/api/v1/metrics/tools
curl http://localhost:8000/api/v1/metrics/system
```

### 4. Run Comprehensive Verification Script

```bash
# Make sure server is running first, then:
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
python scripts/verify_monitoring.py --url http://localhost:8000
```

## Detailed Verification Checklist

### ✅ Component Verification

#### 1. Logging System
- [ ] Logging module can be imported: `from app.core.logging import setup_logging`
- [ ] Logs directory exists: `logs/`
- [ ] Log file can be created: `logs/legal_chatbot.log`
- [ ] JSON format logs are structured correctly
- [ ] Log levels work (DEBUG, INFO, WARNING, ERROR)

**Test**:
```python
from app.core.logging import setup_logging
logger = setup_logging()
logger.info("Test message")
logger.error("Test error")
```

#### 2. Metrics Collection
- [ ] Metrics collector can be imported: `from app.core.metrics import metrics_collector`
- [ ] API requests can be recorded
- [ ] Tool usage can be recorded
- [ ] Summary metrics can be retrieved
- [ ] System metrics can be collected

**Test**:
```python
from app.core.metrics import metrics_collector, SystemMetrics

# Record API request
metrics_collector.record_api_request("/api/v1/test", "GET", 100.0, 200)

# Record tool usage
metrics_collector.record_tool_usage("test_tool", 50.0, True)

# Get summary
summary = metrics_collector.get_summary_metrics()
print(summary)

# Get system metrics
system = SystemMetrics.get_all_metrics()
print(system)
```

#### 3. Health Checker
- [ ] Health checker can be imported: `from app.core.health_checker import health_checker`
- [ ] All dependencies can be checked
- [ ] Health checks return proper status format

**Test**:
```python
from app.core.health_checker import health_checker
import asyncio

async def test():
    deps = await health_checker.check_all_dependencies()
    print(deps)

asyncio.run(test())
```

#### 4. Middleware
- [ ] Middleware can be imported: `from app.core.middleware import RequestResponseLoggingMiddleware`
- [ ] Middleware is registered in `app/api/main.py`
- [ ] Request/response logging works

**Test**: Check `app/api/main.py` for:
```python
from app.core.middleware import RequestResponseLoggingMiddleware, ErrorTrackingMiddleware
app.add_middleware(ErrorTrackingMiddleware)
app.add_middleware(RequestResponseLoggingMiddleware)
```

### ✅ API Endpoint Verification

#### Health Endpoints

1. **`GET /api/v1/health`**
   - [ ] Returns 200 status code
   - [ ] Contains `status` field
   - [ ] Contains `services` dict with dependency statuses
   - [ ] Timestamp is present

2. **`GET /api/v1/health/detailed`**
   - [ ] Returns 200 status code
   - [ ] Contains `dependencies` with detailed info
   - [ ] Contains `system_metrics` (CPU, memory, disk)
   - [ ] Each dependency has `status` and `response_time_ms`

3. **`GET /api/v1/health/live`**
   - [ ] Returns 200 status code
   - [ ] Returns `{"status": "alive", "timestamp": "..."}`

4. **`GET /api/v1/health/ready`**
   - [ ] Returns 200 if ready, 503 if not ready
   - [ ] Returns `{"status": "ready", "timestamp": "..."}`

#### Metrics Endpoints

1. **`GET /api/v1/metrics`**
   - [ ] Returns 200 status code
   - [ ] Contains `summary`, `endpoints`, `tool_usage`, `timestamp`

2. **`GET /api/v1/metrics/summary`**
   - [ ] Returns 200 status code
   - [ ] Contains `uptime_seconds`, `total_requests`, `total_errors`, etc.

3. **`GET /api/v1/metrics/endpoints`**
   - [ ] Returns 200 status code
   - [ ] Contains endpoint metrics
   - [ ] Can filter by endpoint parameter

4. **`GET /api/v1/metrics/tools`**
   - [ ] Returns 200 status code
   - [ ] Contains tool usage statistics
   - [ ] Can filter by tool_name parameter

5. **`GET /api/v1/metrics/system`**
   - [ ] Returns 200 status code
   - [ ] Contains `cpu`, `memory`, `disk` metrics
   - [ ] All metrics have valid values or None

### ✅ Functional Verification

#### Request/Response Logging
1. **Make a test API call**:
   ```bash
   curl -X POST http://localhost:8000/api/v1/chat \
     -H "Content-Type: application/json" \
     -d '{"query": "test query"}'
   ```

2. **Check logs**:
   ```bash
   tail -f logs/legal_chatbot.log | jq
   ```

3. **Verify**:
   - [ ] Request is logged with `type: "request"`
   - [ ] Response is logged with `type: "response"`
   - [ ] Request ID is present and consistent
   - [ ] Process time is logged
   - [ ] Custom headers `X-Request-ID` and `X-Process-Time` are present in response

#### Metrics Collection
1. **Make multiple API calls**:
   ```bash
   for i in {1..5}; do
     curl http://localhost:8000/api/v1/health
   done
   ```

2. **Check metrics**:
   ```bash
   curl http://localhost:8000/api/v1/metrics/summary | jq
   ```

3. **Verify**:
   - [ ] `total_requests` increased
   - [ ] Response times are tracked
   - [ ] Error rates are calculated correctly

#### Tool Usage Tracking
1. **Make agentic chat request**:
   ```bash
   curl -X POST http://localhost:8000/api/v1/agentic-chat \
     -H "Content-Type: application/json" \
     -d '{"query": "What are the UK consumer rights?"}'
   ```

2. **Check tool metrics**:
   ```bash
   curl http://localhost:8000/api/v1/metrics/tools | jq
   ```

3. **Verify**:
   - [ ] Tools used are tracked
   - [ ] Execution times are recorded
   - [ ] Success/failure rates are calculated

#### System Metrics
1. **Check system metrics**:
   ```bash
   curl http://localhost:8000/api/v1/metrics/system | jq
   ```

2. **Verify**:
   - [ ] CPU metrics are present (or None if unavailable)
   - [ ] Memory metrics are present
   - [ ] Disk metrics are present
   - [ ] Values are reasonable (0-100 for percentages)

### ✅ Integration Testing

#### End-to-End Test
1. **Start server**:
   ```bash
   uvicorn app.api.main:app --reload --port 8000
   ```

2. **Run verification script**:
   ```bash
   export PYTHONPATH="${PYTHONPATH}:$(pwd)"
   python scripts/verify_monitoring.py --url http://localhost:8000 --report verification_report.json
   ```

3. **Expected Result**: All tests should pass ✅

## Automated Verification Script

The verification script (`scripts/verify_monitoring.py`) automatically tests:

- ✅ Health checker initialization
- ✅ Middleware initialization
- ✅ Logging functionality
- ✅ Metrics collection
- ✅ Health endpoints (4 endpoints)
- ✅ Metrics endpoints (5 endpoints)
- ✅ Request/response headers
- ✅ Full integration

**Usage**:
```bash
# With server running
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
python scripts/verify_monitoring.py --url http://localhost:8000

# Save report
python scripts/verify_monitoring.py --url http://localhost:8000 --report report.json
```

## Troubleshooting

### Issue: Module import errors
**Solution**: Ensure you're in the project root and virtual environment is activated:
```bash
cd "/Users/javadbeni/Desktop/Legal Chatbot"
source venv/bin/activate
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### Issue: Server connection errors
**Solution**: Start the server first:
```bash
uvicorn app.api.main:app --reload --port 8000
```

### Issue: Logs directory doesn't exist
**Solution**: It will be created automatically on first log write. Or create manually:
```bash
mkdir -p logs
```

### Issue: Metrics not updating
**Solution**: Metrics are updated on each request. Make some API calls and check again.

## Success Criteria

Phase 4.2 is working seamlessly when:

✅ **All unit tests pass** (20/20)
✅ **All components can be imported**
✅ **All health endpoints return valid responses**
✅ **All metrics endpoints return valid data**
✅ **Request/response logging works**
✅ **Metrics are collected automatically**
✅ **System metrics are available**
✅ **Tool usage is tracked**
✅ **Logs are written in JSON format**
✅ **Custom headers are present in responses**

## Summary

To verify monitoring is working:

1. ✅ Run unit tests: `pytest tests/unit/test_monitoring.py -v`
2. ✅ Start server: `uvicorn app.api.main:app --reload`
3. ✅ Test endpoints: Use curl commands above
4. ✅ Check logs: `tail -f logs/legal_chatbot.log | jq`
5. ✅ Run verification script: `python scripts/verify_monitoring.py`

If all steps pass, Phase 4.2 is implemented and working seamlessly! 🎉

