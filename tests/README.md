# Comprehensive Test Suite - Phase 4.1

## Overview

This directory contains comprehensive test suites for the Legal Chatbot project, covering:
- End-to-end API testing
- Error scenario testing
- Load testing
- Integration testing
- Regression testing
- Performance benchmarking

## Test Structure

```
tests/
├── e2e/                          # End-to-end tests
│   ├── test_all_endpoints.py     # All API endpoints coverage
│   ├── test_error_scenarios.py   # Error handling tests
│   ├── test_load.py              # Load testing
│   └── test_regression.py        # Regression tests (Phase 1, 2, 3)
├── integration/                  # Integration tests
│   └── test_service_integration.py  # Service-to-service communication
├── unit/                         # Unit tests
│   └── test_basic.py             # Basic unit tests
└── test_api_integration.py       # API integration tests
```

## Running Tests

### Run All Tests

```bash
# Using the test runner script
python scripts/run_all_tests.py

# Using pytest directly
pytest tests/ -v

# Run specific test suite
pytest tests/e2e/test_all_endpoints.py -v
```

### Run Specific Test Suites

```bash
# E2E Tests
pytest tests/e2e/ -v

# Integration Tests
pytest tests/integration/ -v

# Unit Tests
pytest tests/unit/ -v

# Error Scenarios
pytest tests/e2e/test_error_scenarios.py -v

# Load Testing
pytest tests/e2e/test_load.py -v

# Regression Tests
pytest tests/e2e/test_regression.py -v
```

### Performance Benchmarking

```bash
# Run performance benchmarks
python scripts/test_performance_benchmark.py

# With custom iterations
python scripts/test_performance_benchmark.py --iterations 10

# Save results to file
python scripts/test_performance_benchmark.py --output results.json
```

## Test Categories

### 1. End-to-End Tests (`tests/e2e/`)

**test_all_endpoints.py**
- Tests all API endpoints:
  - `/api/v1/health` - Health check
  - `/api/v1/chat` - Traditional RAG chat
  - `/api/v1/search/hybrid` - Hybrid search
  - `/api/v1/agentic-chat` - Agentic chat
  - `/api/v1/documents` - Document management
- Validates request/response formats
- Tests different modes (public, solicitor)
- Performance validation (response times)

**test_error_scenarios.py**
- Network failures (timeouts, connection errors)
- Invalid inputs (malformed JSON, wrong types)
- Missing required fields
- SQL injection attempts
- XSS attempts
- Large payloads
- Concurrent requests

**test_load.py**
- Light load (5 concurrent users)
- Medium load (10 concurrent users)
- Heavy load (20 concurrent users)
- Sustained load over time
- Mixed endpoints load

**test_regression.py**
- Phase 1 features still work
- Phase 2 features still work
- Phase 3 features still work
- Backward compatibility verification

### 2. Integration Tests (`tests/integration/`)

**test_service_integration.py**
- Service-to-service communication
- RAG service integration
- Guardrails service integration
- LLM service integration
- Agentic service integration
- Vector store (FAISS) integration
- LangChain integration

### 3. Unit Tests (`tests/unit/`)

**test_basic.py**
- Model validation
- Configuration loading
- Health endpoint
- Request validation

## Performance Targets

| Endpoint | Target Response Time |
|----------|---------------------|
| Simple Chat Query | < 3 seconds |
| Complex Chat Query | < 10 seconds |
| Hybrid Search | < 5 seconds |
| Agentic Chat | < 30 seconds |

## Test Execution Requirements

### Prerequisites

1. **API Server Running**
   ```bash
   uvicorn app.api.main:app --reload --port 8000
   ```

2. **Environment Variables**
   - `OPENAI_API_KEY` - Required for LLM service
   - Other config variables as needed

3. **Dependencies Installed**
   ```bash
   pip install -r requirements.txt
   pip install pytest pytest-cov
   ```

### Server Health Check

Before running tests, ensure the server is running:

```bash
curl http://localhost:8000/api/v1/health
```

## Test Results

### Expected Output

```
🧪 Comprehensive Test Suite Runner
============================================================
Project Root: /path/to/legal-chatbot

============================================================
🧪 Running: E2E: All Endpoints
============================================================
✅ Health endpoint test passed
✅ Chat endpoint (solicitor) test passed (response time: 2.34s)
...

📊 Test Summary
============================================================
Total Test Suites: 8
✅ Passed: 8
❌ Failed: 0
⏱️ Total Duration: 45.67s

🎯 Success Rate: 100.0%
🎉 All tests passed!
```

## Continuous Integration

Tests are designed to run in CI/CD pipelines:

```yaml
# .github/workflows/ci-cd.yml
- name: Run Tests
  run: |
    pytest tests/ -v --cov=app --cov-report=xml
```

## Troubleshooting

### Tests Failing Due to Server Not Running

**Error**: `ConnectionError: Failed to establish connection`

**Solution**: Ensure the API server is running:
```bash
uvicorn app.api.main:app --reload --port 8000
```

### Tests Failing Due to Missing API Key

**Error**: `LLM service error: OPENAI_API_KEY not found`

**Solution**: Set the environment variable:
```bash
export OPENAI_API_KEY="your-key-here"
```

### Performance Tests Failing

**Error**: `Response time exceeds threshold`

**Solution**: 
- Check server resources (CPU, memory)
- Ensure no other processes are using resources
- Consider running tests with fewer iterations

## Adding New Tests

### Adding E2E Test

1. Create test file in `tests/e2e/`
2. Import pytest and requests
3. Use `BASE_URL = "http://localhost:8000"`
4. Add test methods with `test_` prefix
5. Use assertions for validation

Example:
```python
def test_new_endpoint(self):
    response = requests.get(f"{self.BASE_URL}/api/v1/new-endpoint")
    assert response.status_code == 200
```

### Adding Integration Test

1. Create test file in `tests/integration/`
2. Import services directly
3. Test service-to-service communication
4. Use mocks where appropriate

## Best Practices

1. **Test Independence**: Each test should be independent and not rely on other tests
2. **Clean State**: Tests should clean up after themselves
3. **Realistic Data**: Use realistic test queries and data
4. **Error Handling**: Test both success and error scenarios
5. **Performance**: Monitor response times and resource usage
6. **Documentation**: Document test purpose and expected behavior

## Coverage Goals

- **Line Coverage**: > 80%
- **Branch Coverage**: > 75%
- **Endpoint Coverage**: 100%
- **Error Scenario Coverage**: > 90%

## Maintenance

- Review and update tests when API changes
- Update performance targets as system improves
- Add new tests for new features
- Remove obsolete tests
- Keep test data up to date

## References

- [pytest Documentation](https://docs.pytest.org/)
- [FastAPI Testing Guide](https://fastapi.tiangolo.com/tutorial/testing/)
- [Performance Testing Best Practices](https://docs.pytest.org/en/stable/how-to/parametrize.html)

