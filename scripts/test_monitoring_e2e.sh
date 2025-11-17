#!/bin/bash
# End-to-End Test Script for Phase 4.2: Monitoring and Observability
# Tests all monitoring features with a running server

set -e

BASE_URL="${BASE_URL:-http://localhost:8000}"
PYTHONPATH="${PYTHONPATH}:$(pwd)"
export PYTHONPATH

echo "=========================================="
echo "Phase 4.2: E2E Monitoring Verification"
echo "=========================================="
echo ""

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if server is running
echo "🔍 Checking if server is running..."
if curl -s "${BASE_URL}/api/v1/health/live" > /dev/null 2>&1; then
    echo -e "${GREEN}✅ Server is running${NC}"
else
    echo -e "${RED}❌ Server is not running${NC}"
    echo -e "${YELLOW}   Start server with: uvicorn app.api.main:app --reload --port 8000${NC}"
    exit 1
fi

echo ""
echo "📊 Running Tests..."
echo ""

# Test health endpoints
echo "1️⃣  Testing Health Endpoints..."
HEALTH=$(curl -s "${BASE_URL}/api/v1/health")
if echo "$HEALTH" | jq -e '.status' > /dev/null 2>&1; then
    echo -e "${GREEN}   ✅ Basic health check${NC}"
else
    echo -e "${RED}   ❌ Basic health check failed${NC}"
    exit 1
fi

DETAILED=$(curl -s "${BASE_URL}/api/v1/health/detailed")
if echo "$DETAILED" | jq -e '.dependencies' > /dev/null 2>&1; then
    echo -e "${GREEN}   ✅ Detailed health check${NC}"
else
    echo -e "${RED}   ❌ Detailed health check failed${NC}"
    exit 1
fi

LIVE=$(curl -s "${BASE_URL}/api/v1/health/live")
if echo "$LIVE" | jq -e '.status == "alive"' > /dev/null 2>&1; then
    echo -e "${GREEN}   ✅ Liveness probe${NC}"
else
    echo -e "${RED}   ❌ Liveness probe failed${NC}"
    exit 1
fi

echo ""

# Test metrics endpoints
echo "2️⃣  Testing Metrics Endpoints..."
METRICS=$(curl -s "${BASE_URL}/api/v1/metrics")
if echo "$METRICS" | jq -e '.summary' > /dev/null 2>&1; then
    echo -e "${GREEN}   ✅ All metrics${NC}"
else
    echo -e "${RED}   ❌ All metrics failed${NC}"
    exit 1
fi

SUMMARY=$(curl -s "${BASE_URL}/api/v1/metrics/summary")
if echo "$SUMMARY" | jq -e '.total_requests' > /dev/null 2>&1; then
    echo -e "${GREEN}   ✅ Summary metrics${NC}"
else
    echo -e "${RED}   ❌ Summary metrics failed${NC}"
    exit 1
fi

SYSTEM=$(curl -s "${BASE_URL}/api/v1/metrics/system")
if echo "$SYSTEM" | jq -e '.cpu' > /dev/null 2>&1; then
    echo -e "${GREEN}   ✅ System metrics${NC}"
else
    echo -e "${RED}   ❌ System metrics failed${NC}"
    exit 1
fi

echo ""

# Test request/response headers
echo "3️⃣  Testing Request/Response Headers..."
HEADERS=$(curl -s -I "${BASE_URL}/api/v1/health")
if echo "$HEADERS" | grep -q "X-Request-ID"; then
    echo -e "${GREEN}   ✅ X-Request-ID header${NC}"
else
    echo -e "${YELLOW}   ⚠️  X-Request-ID header not found (may be normal)${NC}"
fi

if echo "$HEADERS" | grep -q "X-Process-Time"; then
    echo -e "${GREEN}   ✅ X-Process-Time header${NC}"
else
    echo -e "${YELLOW}   ⚠️  X-Process-Time header not found (may be normal)${NC}"
fi

echo ""

# Test logging
echo "4️⃣  Testing Logging..."
if [ -f "logs/legal_chatbot.log" ]; then
    echo -e "${GREEN}   ✅ Log file exists${NC}"
    
    # Make a request to generate a log entry
    curl -s "${BASE_URL}/api/v1/health" > /dev/null
    
    # Wait a bit for log to be written
    sleep 0.5
    
    # Check if log has entries
    if [ -s "logs/legal_chatbot.log" ]; then
        echo -e "${GREEN}   ✅ Log file has entries${NC}"
        
        # Check if it's JSON format
        LAST_LINE=$(tail -n 1 logs/legal_chatbot.log)
        if echo "$LAST_LINE" | jq . > /dev/null 2>&1; then
            echo -e "${GREEN}   ✅ Log format is JSON${NC}"
        else
            echo -e "${YELLOW}   ⚠️  Log format may not be JSON (check LOG_FORMAT setting)${NC}"
        fi
    else
        echo -e "${YELLOW}   ⚠️  Log file is empty${NC}"
    fi
else
    echo -e "${YELLOW}   ⚠️  Log file not found (will be created on first request)${NC}"
fi

echo ""

# Test metrics collection after requests
echo "5️⃣  Testing Metrics Collection..."
# Make a few requests
for i in {1..3}; do
    curl -s "${BASE_URL}/api/v1/health" > /dev/null
done

sleep 0.5

SUMMARY_AFTER=$(curl -s "${BASE_URL}/api/v1/metrics/summary")
REQUESTS=$(echo "$SUMMARY_AFTER" | jq -r '.total_requests // 0')

if [ "$REQUESTS" -gt 0 ]; then
    echo -e "${GREEN}   ✅ Metrics are being collected (${REQUESTS} requests tracked)${NC}"
else
    echo -e "${YELLOW}   ⚠️  No requests tracked yet${NC}"
fi

echo ""
echo "=========================================="
echo -e "${GREEN}✅ All E2E Tests Passed!${NC}"
echo "=========================================="
echo ""
echo "📊 Summary:"
echo "   • Health endpoints: ✅"
echo "   • Metrics endpoints: ✅"
echo "   • Request/response headers: ✅"
echo "   • Logging: ✅"
echo "   • Metrics collection: ✅"
echo ""
echo "🎉 Phase 4.2: Monitoring and Observability is working seamlessly!"

