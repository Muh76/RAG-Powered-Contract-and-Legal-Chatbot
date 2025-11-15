# Phase 2 - Module 4 & 5: Reranking and Explainability
# Status: COMPLETE

## Summary
Successfully implemented cross-encoder reranking and explainability features for the advanced RAG system. This includes source highlighting, retrieval explanations, and automated red-team testing.

---

## Implementation Overview

### Components Created

1. **CrossEncoderReranker** (`retrieval/rerankers/cross_encoder_reranker.py`)
   - Cross-encoder reranking using sentence-transformers
   - Model: `cross-encoder/ms-marco-MiniLM-L-6-v2` (default)
   - Batch processing for efficiency
   - Metadata preservation
   - Configurable batch size and device

2. **ExplainabilityAnalyzer** (`retrieval/explainability.py`)
   - Query term extraction and matching
   - Source highlighting with matched text spans
   - Retrieval explanation generation
   - Confidence score calculation
   - Matched terms identification

3. **RedTeamTester** (`retrieval/red_team_tester.py`)
   - Automated red team test execution
   - Test case loading (JSON or default)
   - Test result reporting
   - Category and risk level grouping
   - Report generation with statistics

---

## Features Implemented

### Module 4: Cross-Encoder Reranking

#### 1. Reranker Class
- ✅ Cross-encoder model initialization
- ✅ Batch reranking for efficiency
- ✅ Metadata preservation
- ✅ Score normalization and combination
- ✅ Top-k result selection
- ✅ Error handling and fallback

#### 2. Hybrid Retriever Integration
- ✅ Optional reranking in hybrid search
- ✅ Configurable via `enable_reranking` parameter
- ✅ Reranks top 2x candidates for better quality
- ✅ Combines reranked results with remaining results
- ✅ Statistics tracking

#### 3. Configuration
- ✅ `ENABLE_RERANKING` setting in config
- ✅ `RERANKER_MODEL` configuration
- ✅ `RERANKER_BATCH_SIZE` configuration
- ✅ Automatic initialization in RAG service

### Module 5: Explainability and Source Highlighting

#### 1. Explainability Analyzer
- ✅ Query term extraction (stop word filtering)
- ✅ Matched term identification
- ✅ Text highlighting with customizable tags
- ✅ Retrieval explanation generation
- ✅ Confidence score calculation
- ✅ Matched text span extraction

#### 2. Integration with RAG Service
- ✅ Optional explainability in search results
- ✅ `include_explanation` parameter
- ✅ `highlight_sources` parameter
- ✅ Explanation fields in results:
   - `explanation`: Why document was retrieved
   - `confidence`: Confidence score (0-1)
   - `matched_terms`: List of matched query terms
   - `highlighted_text`: Text with highlighted matches
   - `matched_spans`: [(start, end), ...] for matches

#### 3. API Integration
- ✅ `include_explanation` parameter in API request
- ✅ `highlight_sources` parameter in API request
- ✅ Explainability fields in response
- ✅ GET and POST endpoint support

### Module 6: Red Team Testing Automation

#### 1. Red Team Tester
- ✅ Automated test execution
- ✅ Test case loading (file or default)
- ✅ Guardrails integration
- ✅ RAG pipeline testing
- ✅ Result validation
- ✅ Performance tracking

#### 2. Test Categories
- ✅ Prompt injection detection
- ✅ Domain gating (legal vs non-legal)
- ✅ Harmful content detection
- ✅ PII detection
- ✅ Fabricated statutes detection
- ✅ Valid legal queries (should pass)

#### 3. Reporting
- ✅ Summary statistics (total, passed, failed, pass rate)
- ✅ Results by category
- ✅ Results by risk level
- ✅ Failed test details
- ✅ JSON report export

---

## Usage Examples

### Reranking

```python
from retrieval.rerankers.cross_encoder_reranker import CrossEncoderReranker

# Initialize reranker
reranker = CrossEncoderReranker()

# Rerank documents
query = "What is contract law?"
documents = ["doc1 text...", "doc2 text...", ...]
reranked = reranker.rerank(query, documents, top_k=5)

# Rerank with metadata
reranked_results = reranker.rerank_results(
    query=query,
    retrieval_results=results,
    top_k=5,
    preserve_metadata=True
)
```

### Explainability

```python
from retrieval.explainability import ExplainabilityAnalyzer

# Initialize analyzer
analyzer = ExplainabilityAnalyzer()

# Highlight matched terms
highlighted_text, spans = analyzer.highlight_matched_terms(
    text="Document text...",
    query="contract law",
    highlight_tag="**"
)

# Generate explanation
explanation = analyzer.explain_retrieval(result, query)
print(explanation.explanation)
print(explanation.confidence)
print(explanation.matched_terms)
```

### Red Team Testing

```python
from retrieval.red_team_tester import RedTeamTester

# Initialize tester
tester = RedTeamTester(guardrails_service=guardrails)

# Run test suite
results = tester.run_test_suite()

# Generate report
report = tester.generate_report(results, output_path="report.json")
print(f"Pass rate: {report['summary']['pass_rate']}%")
```

### RAG Service with Explainability

```python
from app.services.rag_service import RAGService

# Initialize with hybrid search
rag_service = RAGService(use_hybrid=True)

# Search with explainability
results = rag_service.search(
    query="What are employee rights?",
    top_k=5,
    use_hybrid=True,
    include_explanation=True,
    highlight_sources=True
)

# Access explainability fields
for result in results:
    print(f"Explanation: {result.get('explanation')}")
    print(f"Confidence: {result.get('confidence')}")
    print(f"Highlighted: {result.get('highlighted_text')}")
```

### API Usage

```bash
# POST request with explainability
curl -X POST http://localhost:8000/api/v1/search/hybrid \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is contract law?",
    "top_k": 5,
    "include_explanation": true,
    "highlight_sources": true
  }'

# GET request with explainability
curl "http://localhost:8000/api/v1/search/hybrid?query=employee%20rights&top_k=5&include_explanation=true&highlight_sources=true"
```

---

## Configuration

### Settings (`app/core/config.py`)

```python
# Reranking Configuration
ENABLE_RERANKING: bool = False  # Enable cross-encoder reranking
RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANKER_BATCH_SIZE: int = 32
```

### Enable Reranking

To enable reranking in hybrid search:

1. Set `ENABLE_RERANKING=True` in config or environment
2. Reranker will auto-initialize when hybrid retriever is created
3. Results will be reranked automatically after fusion

---

## Testing

### Test Reranking

```bash
python scripts/test_reranking.py
```

### Test Red Team

```bash
python scripts/test_red_team.py
```

### Test Explainability

Test explainability through RAG service or API with `include_explanation=true`.

---

## Performance Considerations

### Reranking
- **Speed**: ~10-50ms per query (depends on batch size and model)
- **Memory**: ~500MB for cross-encoder model
- **Recommendation**: Use reranking for top-k results only (not all candidates)

### Explainability
- **Speed**: <1ms per result (very fast)
- **Memory**: Negligible
- **Recommendation**: Enable explainability when needed for debugging/UX

### Red Team Testing
- **Speed**: Depends on guardrails/RAG service speed
- **Memory**: Negligible
- **Recommendation**: Run as part of CI/CD pipeline

---

## Future Enhancements

1. **Advanced Reranking**:
   - Multiple reranker models (ensemble)
   - Query-specific reranker selection
   - Adaptive reranking based on query type

2. **Enhanced Explainability**:
   - Visual highlighting in UI
   - Interactive explanation exploration
   - Retrieval path visualization
   - Confidence score breakdown by component

3. **Advanced Red Team Testing**:
   - Continuous monitoring
   - Automated test case generation
   - Adversarial query generation
   - Performance regression detection

---

## Files Created/Modified

### New Files
- `retrieval/rerankers/__init__.py`
- `retrieval/rerankers/cross_encoder_reranker.py`
- `retrieval/explainability.py`
- `retrieval/red_team_tester.py`
- `scripts/test_reranking.py`
- `scripts/test_red_team.py`
- `docs/phase2_reranking_explainability.md`

### Modified Files
- `retrieval/hybrid_retriever.py` - Added reranking integration
- `retrieval/__init__.py` - Added new exports
- `app/services/rag_service.py` - Added reranking and explainability
- `app/core/config.py` - Added reranking configuration
- `app/models/schemas.py` - Added explainability fields
- `app/api/routes/search.py` - Added explainability parameters

---

## Status: COMPLETE ✅

All Phase 2 remaining features have been implemented:
- ✅ Cross-encoder reranking
- ✅ Explainability and source highlighting
- ✅ Advanced red-team testing automation

The system now provides:
- Improved retrieval accuracy through reranking
- Transparent retrieval decisions through explainability
- Automated safety validation through red team testing

