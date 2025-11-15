#!/bin/bash
# Start API server and test hybrid search endpoint

set -e

echo "🚀 Starting API Server and Testing Hybrid Search Endpoint"
echo "=" * 60

# Change to project directory
cd "$(dirname "$0")/.."

# Activate virtual environment
source venv/bin/activate

# Start server in background
echo "📡 Starting API server..."
uvicorn app.api.main:app --host 0.0.0.0 --port 8000 &
SERVER_PID=$!

# Wait for server to start
echo "⏳ Waiting for server to start..."
sleep 5

# Test health endpoint
echo ""
echo "🏥 Testing Health Endpoint..."
for i in {1..10}; do
    if curl -s http://localhost:8000/api/v1/health > /dev/null 2>&1; then
        echo "✅ Server is running!"
        curl -s http://localhost:8000/api/v1/health | python3 -m json.tool
        break
    else
        if [ $i -eq 10 ]; then
            echo "❌ Server failed to start"
            kill $SERVER_PID 2>/dev/null || true
            exit 1
        fi
        echo "   Waiting... ($i/10)"
        sleep 2
    fi
done

# Run test script
echo ""
echo "🧪 Running Hybrid Search API Tests..."
python scripts/test_hybrid_api.py

# Cleanup
echo ""
echo "🛑 Stopping server..."
kill $SERVER_PID 2>/dev/null || true
wait $SERVER_PID 2>/dev/null || true

echo ""
echo "✅ Testing completed!"

