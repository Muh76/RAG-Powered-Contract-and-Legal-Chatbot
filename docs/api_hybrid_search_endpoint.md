# Hybrid Search API Endpoint Documentation

## Overview

The Hybrid Search API endpoint (`/api/v1/search/hybrid`) provides access to the advanced hybrid retrieval system combining BM25 keyword search and semantic search with metadata filtering.

## Endpoints

### POST /api/v1/search/hybrid

Perform hybrid search with POST request and JSON body.

**Request Body:**
```json
{
  "query": "What are the implied conditions in a contract of sale?",
  "top_k": 10,
  "fusion_strategy": "rrf",
  "similarity_threshold": 0.0,
  "metadata_filters": [
    {
      "field": "jurisdiction",
      "value": "UK",
      "operator": "eq"
    }
  ],
  "pre_filter": true
}
```

**Response:**
```json
{
  "query": "What are the implied conditions in a contract of sale?",
  "results": [
    {
      "chunk_id": "chunk_0",
      "text": "Section 13 - Sale by description...",
      "similarity_score": 0.856,
      "bm25_score": 12.45,
      "semantic_score": 0.743,
      "bm25_rank": 2,
      "semantic_rank": 1,
      "rank": 1,
      "section": "Section 13",
      "title": "Sale of Goods Act 1979",
      "source": "legislation.gov.uk",
      "jurisdiction": "UK",
      "metadata": {
        "document_type": "statute",
        "section": "Section 13"
      }
    }
  ],
  "total_results": 10,
  "fusion_strategy": "rrf",
  "search_time_ms": 45.2,
  "bm25_results_count": 10,
  "semantic_results_count": 10,
  "timestamp": "2024-11-15T14:30:00.000Z"
}
```

### GET /api/v1/search/hybrid

Perform hybrid search with GET request and query parameters.

**Query Parameters:**
- `query` (required): Search query text
- `top_k` (optional, default=10): Number of results to return (1-50)
- `fusion_strategy` (optional, default="rrf"): Fusion strategy ("rrf" or "weighted")
- `similarity_threshold` (optional, default=0.0): Minimum similarity score (0.0-1.0)
- `jurisdiction` (optional): Filter by jurisdiction
- `document_type` (optional): Filter by document type
- `source` (optional): Filter by source

**Example:**
```
GET /api/v1/search/hybrid?query=contract%20law&top_k=5&fusion_strategy=weighted&jurisdiction=UK
```

## Request Parameters

### HybridSearchRequest

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `query` | string | Yes | - | Search query text (1-1000 characters) |
| `top_k` | integer | No | 10 | Number of results to return (1-50) |
| `fusion_strategy` | string | No | "rrf" | Fusion strategy: "rrf" or "weighted" |
| `bm25_weight` | float | No | null | BM25 weight for weighted fusion (0.0-1.0) |
| `semantic_weight` | float | No | null | Semantic weight for weighted fusion (0.0-1.0) |
| `similarity_threshold` | float | No | 0.0 | Minimum similarity score (0.0-1.0) |
| `metadata_filters` | array | No | [] | List of metadata filters |
| `pre_filter` | boolean | No | true | Apply filters before search |

### MetadataFilterRequest

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `field` | string | Yes | - | Metadata field name (e.g., "jurisdiction") |
| `value` | any | Yes | - | Filter value (string, list, etc.) |
| `operator` | string | No | "eq" | Filter operator: "eq", "in", "not_in", "contains" |

### Fusion Strategies

#### RRF (Reciprocal Rank Fusion)
- Combines rankings from BM25 and semantic search
- Formula: `score = sum(1 / (k + rank))`
- Better for diverse result sets
- Default strategy

#### Weighted
- Combines normalized scores from BM25 and semantic search
- Formula: `score = bm25_weight * normalized_bm25 + semantic_weight * semantic_score`
- Better for controlled blending
- Requires weights to sum to 1.0

## Response Format

### HybridSearchResponse

| Field | Type | Description |
|-------|------|-------------|
| `query` | string | Original search query |
| `results` | array | List of search results |
| `total_results` | integer | Total number of results |
| `fusion_strategy` | string | Fusion strategy used |
| `search_time_ms` | float | Search execution time in milliseconds |
| `bm25_results_count` | integer | Number of results with BM25 scores |
| `semantic_results_count` | integer | Number of results with semantic scores |
| `timestamp` | string | Response timestamp (ISO 8601) |

### HybridSearchResult

| Field | Type | Description |
|-------|------|-------------|
| `chunk_id` | string | Unique chunk identifier |
| `text` | string | Chunk text content |
| `similarity_score` | float | Final fused similarity score |
| `bm25_score` | float | BM25 score (if available) |
| `semantic_score` | float | Semantic similarity score (if available) |
| `bm25_rank` | integer | BM25 rank (if available) |
| `semantic_rank` | integer | Semantic rank (if available) |
| `rank` | integer | Final result rank (1-indexed) |
| `section` | string | Section name |
| `title` | string | Document title (if available) |
| `source` | string | Document source (if available) |
| `jurisdiction` | string | Jurisdiction (if available) |
| `metadata` | object | Additional metadata |

## Examples

### Example 1: Basic Hybrid Search

**Request:**
```bash
curl -X POST http://localhost:8000/api/v1/search/hybrid \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are employee rights in the UK?",
    "top_k": 5,
    "fusion_strategy": "rrf"
  }'
```

**Response:**
```json
{
  "query": "What are employee rights in the UK?",
  "results": [...],
  "total_results": 5,
  "fusion_strategy": "rrf",
  "search_time_ms": 42.5,
  "bm25_results_count": 5,
  "semantic_results_count": 5,
  "timestamp": "2024-11-15T14:30:00.000Z"
}
```

### Example 2: Hybrid Search with Metadata Filtering

**Request:**
```bash
curl -X POST http://localhost:8000/api/v1/search/hybrid \
  -H "Content-Type: application/json" \
  -d '{
    "query": "contract law",
    "top_k": 10,
    "fusion_strategy": "weighted",
    "metadata_filters": [
      {
        "field": "jurisdiction",
        "value": "UK",
        "operator": "eq"
      },
      {
        "field": "document_type",
        "value": "statute",
        "operator": "eq"
      }
    ]
  }'
```

### Example 3: GET Request with Query Parameters

**Request:**
```bash
curl -X GET "http://localhost:8000/api/v1/search/hybrid?query=discrimination%20law&top_k=5&fusion_strategy=rrf&jurisdiction=UK"
```

### Example 4: IN Filter

**Request:**
```bash
curl -X POST http://localhost:8000/api/v1/search/hybrid \
  -H "Content-Type: application/json" \
  -d '{
    "query": "contract law",
    "top_k": 10,
    "metadata_filters": [
      {
        "field": "document_type",
        "value": ["statute", "contract"],
        "operator": "in"
      }
    ]
  }'
```

## Error Responses

### 400 Bad Request
```json
{
  "detail": "Validation error: query must be between 1 and 1000 characters"
}
```

### 503 Service Unavailable
```json
{
  "detail": "Hybrid retriever not available. Ensure data ingestion is complete and FAISS index exists."
}
```

### 500 Internal Server Error
```json
{
  "detail": "Internal server error: [error message]"
}
```

## Testing

### Start the API Server
```bash
uvicorn app.api.main:app --reload
```

### Run Test Script
```bash
python scripts/test_hybrid_api.py
```

### Quick Test (Bash)
```bash
./scripts/quick_test_api.sh
```

## Performance

Typical performance metrics:
- **BM25 search**: ~10-50ms per query
- **Semantic search**: ~20-50ms per query
- **Hybrid search**: ~50-150ms per query (both methods + fusion)
- **Metadata filtering**: <5ms overhead

## Notes

1. The endpoint requires hybrid search to be enabled in RAGService
2. FAISS index and chunk metadata must be available
3. BM25 and semantic retrievers must be initialized
4. Metadata filters are applied as AND conditions (all must match)
5. Pre-filtering is recommended for large corpora (>1000 chunks)

## Integration

The hybrid search endpoint can be integrated into the frontend:

```javascript
// JavaScript example
const response = await fetch('http://localhost:8000/api/v1/search/hybrid', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    query: 'contract law',
    top_k: 10,
    fusion_strategy: 'rrf',
    metadata_filters: [
      { field: 'jurisdiction', value: 'UK', operator: 'eq' }
    ]
  })
});

const data = await response.json();
console.log(data.results);
```

## API Documentation

Interactive API documentation is available when the server is running:
- **Swagger UI**: `http://localhost:8000/docs` - Interactive API explorer
- **ReDoc**: `http://localhost:8000/redoc` - Alternative API documentation

**Note**: These links only work when the API server is running. Start the server with:
```bash
uvicorn app.api.main:app --reload
```

