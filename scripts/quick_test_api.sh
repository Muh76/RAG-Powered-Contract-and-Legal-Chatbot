#!/bin/bash
# Quick test script for hybrid search API

echo "🔍 Testing Hybrid Search API Endpoint"
echo "======================================"

# Check if server is running
if curl -s http://localhost:8000/api/v1/health > /dev/null 2>&1; then
    echo "✅ API server is running"
    
    echo ""
    echo "📤 Testing POST /api/v1/search/hybrid"
    curl -X POST http://localhost:8000/api/v1/search/hybrid \
        -H "Content-Type: application/json" \
        -d '{"query": "contract law", "top_k": 3, "fusion_strategy": "rrf"}' \
        -s | python3 -m json.tool
    
    echo ""
    echo "📥 Testing GET /api/v1/search/hybrid"
    curl -X GET "http://localhost:8000/api/v1/search/hybrid?query=employee%20rights&top_k=3&fusion_strategy=weighted" \
        -s | python3 -m json.tool
    
    echo ""
    echo "✅ API endpoint tests completed!"
else
    echo "❌ API server is not running"
    echo "Please start the server first:"
    echo "   uvicorn app.api.main:app --reload"
fi
