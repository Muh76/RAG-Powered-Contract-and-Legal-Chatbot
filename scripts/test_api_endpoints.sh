#!/bin/bash
# Test API endpoints manually
# Usage: ./scripts/test_api_endpoints.sh

echo "🧪 Testing API Endpoints"
echo "=" * 60
echo ""
echo "Prerequisites:"
echo "1. Server must be running: uvicorn app.api.main:app --reload"
echo "2. FAISS index must exist: data/faiss_index.bin"
echo "3. Chunk metadata must exist: data/chunk_metadata.pkl"
echo ""

# Test health endpoint
echo "1️⃣ Testing Health Endpoint..."
curl -s http://localhost:8000/api/v1/health | python3 -m json.tool
echo ""

# Test POST hybrid search
echo "2️⃣ Testing POST /api/v1/search/hybrid..."
curl -X POST http://localhost:8000/api/v1/search/hybrid \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the implied conditions in a contract of sale?",
    "top_k": 5,
    "fusion_strategy": "rrf"
  }' | python3 -m json.tool
echo ""

# Test GET hybrid search
echo "3️⃣ Testing GET /api/v1/search/hybrid..."
curl -X GET "http://localhost:8000/api/v1/search/hybrid?query=employee%20rights&top_k=3&fusion_strategy=weighted" \
  | python3 -m json.tool
echo ""

# Test with metadata filter
echo "4️⃣ Testing Metadata Filtering..."
curl -X POST http://localhost:8000/api/v1/search/hybrid \
  -H "Content-Type: application/json" \
  -d '{
    "query": "contract law",
    "top_k": 5,
    "fusion_strategy": "rrf",
    "metadata_filters": [
      {
        "field": "jurisdiction",
        "value": "UK",
        "operator": "eq"
      }
    ]
  }' | python3 -m json.tool
echo ""

echo "✅ Testing completed!"
echo ""
echo "📋 API Documentation:"
echo "   Swagger UI: http://localhost:8000/docs"
echo "   ReDoc: http://localhost:8000/redoc"

