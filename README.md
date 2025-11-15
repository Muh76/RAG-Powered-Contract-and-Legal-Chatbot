# Legal Chatbot - AI-Powered Legal Assistant

A production-ready legal chatbot built with RAG (Retrieval-Augmented Generation) for UK legal system with future localization for Iranian market.

## 🎯 Project Overview

This project demonstrates end-to-end AI system development with:
- **Phase 1**: ✅ MVP chatbot with UK legal corpus (CUAD + Legislation.gov.uk) - **COMPLETE**
- **Phase 2**: ✅ Advanced RAG with hybrid retrieval, reranking, and explainability - **COMPLETE**
- **Phase 3**: 📋 Multilingual support (English + Farsi) and role-based responses
- **Phase 4**: 📋 Enterprise features with authentication and monetization

### 📊 Phase 2 Status (Advanced RAG) - ✅ **COMPLETE**
- ✅ **Dataset Preparation**: 1,411 chunks (1,389 CUAD + 22 UK statutes)
- ✅ **Gold Evaluation Set**: 150 Q&A pairs with professional methodology
- ✅ **Safety Testing**: 50 red-team test cases for guardrail validation
- ✅ **Visualization**: Comprehensive dataset analysis dashboard
- ✅ **Hybrid Retrieval**: BM25 + Semantic search with fusion strategies (RRF, weighted)
- ✅ **Metadata Filtering**: Structured metadata filtering with multiple operators
- ✅ **Cross-Encoder Reranking**: Improved retrieval accuracy through reranking
- ✅ **Explainability**: Source highlighting and retrieval explanations
- ✅ **Red Team Testing Automation**: Automated adversarial testing framework
- ✅ **API Integration**: Hybrid search endpoints with explainability support

## 🏗️ Architecture

```
/app           # FastAPI services
/frontend      # Streamlit UI
/ingestion     # Document loaders and parsers
/retrieval     # Embeddings and vector store
/guardrails    # Safety policies and validators
/eval          # RAG evaluation framework
/infra         # Docker and deployment configs
/tests         # Comprehensive test suite
/docs          # Architecture and security documentation
/notebooks     # Phase-specific development notebooks
```

## 🚀 Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/Muh76/legal-chatbot.git
   cd legal-chatbot
   ```

2. **Set up environment**
   ```bash
   pip install -r requirements.txt
   export OPENAI_API_KEY="your-api-key-here"
   ```

3. **Run the application**
   ```bash
   # Option 1: Quick start script
   python scripts/quick_start.py
   
   # Option 2: Manual setup
   uvicorn app.api.main:app --reload --port 8000 &
   streamlit run frontend/app.py --server.port 8501
   
   # Option 3: Docker
   docker-compose -f docker-compose.phase1.yml up
   ```

4. **Access the UI**
   - Streamlit UI: http://localhost:8501
   - FastAPI docs: http://localhost:8000/docs (Swagger UI)
   - ReDoc: http://localhost:8000/redoc
   - API Health: http://localhost:8000/api/v1/health
   - Hybrid Search API: http://localhost:8000/api/v1/search/hybrid

5. **Ingest Data (if needed)**
   ```bash
   # Create FAISS index and embeddings
   python scripts/ingest_data.py
   ```

## 📊 Features

### Phase 1 (MVP) - ✅ **COMPLETED**
- ✅ UK legal corpus with comprehensive legal documents
- ✅ Vector-based retrieval with FAISS and TF-IDF embeddings
- ✅ Safety guardrails and domain gating
- ✅ FastAPI backend with Streamlit UI
- ✅ Docker containerization
- ✅ End-to-end RAG pipeline with citations
- ✅ Dual-mode responses (Solicitor/Public)
- ✅ Comprehensive testing and validation

### Phase 2 (Advanced RAG) - ✅ **COMPLETED**
- ✅ **Hybrid Retrieval System**
  - BM25 keyword-based search
  - Semantic search with embeddings (FAISS)
  - Fusion strategies (Reciprocal Rank Fusion, Weighted combination)
  - Configurable BM25 and semantic weights
  - Top-k retrieval with metadata filtering
  
- ✅ **Cross-Encoder Reranking**
  - Cross-encoder model for improved accuracy
  - Batch processing for efficiency
  - Automatic reranking integration with hybrid search
  - Configurable via settings (`ENABLE_RERANKING`)
  
- ✅ **Explainability & Source Highlighting**
  - Query term extraction and matching
  - Source text highlighting with matched spans
  - Retrieval explanation generation
  - Confidence score calculation
  - Matched terms identification
  - API support for explainability features
  
- ✅ **Advanced Red Team Testing**
  - Automated adversarial test execution
  - Test case categories: prompt injection, domain gating, harmful content, PII detection
  - Automated report generation with statistics
  - Integration with guardrails service
  - Test validation and performance tracking
  
- ✅ **API Enhancements**
  - POST/GET `/api/v1/search/hybrid` endpoints
  - Metadata filtering support
  - Explainability parameters (`include_explanation`, `highlight_sources`)
  - BM25, semantic, and rerank scores in response
  - Backward compatibility maintained

### Phase 3 (Localization)
- 🔄 Multilingual support (English + Farsi)
- 🔄 Role-based responses (Solicitor vs Public)
- 🔄 Document upload and private corpora
- 🔄 Privacy compliance (GDPR/UK GDPR)

### Phase 4 (Enterprise)
- 🔄 OAuth2 authentication and RBAC
- 🔄 Multi-tenant architecture
- 🔄 Audit logging and compliance
- 🔄 Billing and monetization hooks

## 🛠️ Tech Stack

- **Backend**: FastAPI, PostgreSQL, pgvector
- **Frontend**: Streamlit
- **ML/AI**: 
  - OpenAI API (GPT-4)
  - SentenceTransformers (embeddings, cross-encoder reranking)
  - PyTorch (model inference)
  - RAGAS (evaluation)
- **Retrieval**: 
  - FAISS (vector similarity search)
  - BM25 (keyword search)
  - Cross-encoder reranking
- **Vector DB**: FAISS (in-memory), Qdrant/pgvector (optional)
- **Infrastructure**: Docker, Docker Compose
- **Monitoring**: Prometheus, Grafana, OpenTelemetry
- **Testing**: pytest, automated red-team testing

## 📈 Business Value

- **For Law Firms**: 
  - Reliable, cited legal answers with audit trails
  - Advanced hybrid retrieval for accurate document finding
  - Explainable AI with confidence scores and source highlighting
  - Automated safety testing for compliance

- **For Public**: 
  - Accessible legal information with clear disclaimers
  - Transparent retrieval explanations
  - Highlighted source text for easy verification

- **For Developers**: 
  - Production-ready RAG system with enterprise features
  - Advanced retrieval techniques (BM25 + Semantic + Reranking)
  - Comprehensive testing framework
  - Well-documented API with explainability support

## 🔒 Security & Compliance

- Privacy-first design with PII redaction
- Automated red-team testing for adversarial scenarios
- Prompt injection detection and prevention
- Domain gating (legal vs non-legal queries)
- Harmful content detection
- Audit logging for compliance
- Multi-tenant data isolation (planned)
- Security scanning and dependency checks

## 📝 Documentation

- [Architecture Overview](docs/architecture/README.md)
- [Security Guidelines](docs/security/README.md)
- [Privacy Policy](docs/privacy/README.md)
- [Evaluation Metrics](docs/eval/README.md)
- [Phase 2 Implementation](docs/phase2_hybrid_retrieval.md)
- [Reranking & Explainability](docs/phase2_reranking_explainability.md)
- [API Documentation](docs/api_hybrid_search_endpoint.md)
- [Phase 2 Completion Summary](docs/phase2_completion_summary.md)

## 🔍 Phase 2 Features in Detail

### Hybrid Search API

**POST `/api/v1/search/hybrid`**
```bash
curl -X POST http://localhost:8000/api/v1/search/hybrid \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are employee rights in the UK?",
    "top_k": 5,
    "fusion_strategy": "rrf",
    "include_explanation": true,
    "highlight_sources": true,
    "metadata_filters": [
      {"field": "jurisdiction", "value": "UK", "operator": "eq"}
    ]
  }'
```

**Features:**
- BM25 + Semantic fusion (RRF or weighted)
- Metadata filtering (pre-filter or post-filter)
- Cross-encoder reranking (optional)
- Explainability (retrieval explanations)
- Source highlighting (matched terms)
- Returns: BM25, semantic, rerank, and fused scores

### Configuration

**Enable Reranking:**
```python
# In app/core/config.py or environment
ENABLE_RERANKING = True
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANKER_BATCH_SIZE = 32
```

**Hybrid Search Settings:**
```python
HYBRID_SEARCH_FUSION_STRATEGY = "rrf"  # or "weighted"
HYBRID_SEARCH_BM25_WEIGHT = 0.4
HYBRID_SEARCH_SEMANTIC_WEIGHT = 0.6
HYBRID_SEARCH_TOP_K_FINAL = 10
```

### Testing

```bash
# Test reranking
python scripts/test_reranking.py

# Test red team testing
python scripts/test_red_team.py

# Test hybrid API
python scripts/test_hybrid_api.py

# Test explainability (through API)
curl "http://localhost:8000/api/v1/search/hybrid?query=contract%20law&include_explanation=true&highlight_sources=true"
```

## 📋 Project Status

- **Phase 1**: ✅ **COMPLETED** - MVP with RAG pipeline, guardrails, and web interface
- **Phase 2**: ✅ **COMPLETED** - Advanced RAG with hybrid retrieval, reranking, explainability, and red-team testing  
- **Phase 3**: 📋 **PLANNED** - Multilingual support and role-based responses
- **Phase 4**: 📋 **PLANNED** - Enterprise features and monetization

### Phase 2 Completion Summary

**Core Features:**
- ✅ Hybrid retrieval combining BM25 + Semantic search
- ✅ Metadata filtering with multiple operators
- ✅ Cross-encoder reranking for improved accuracy
- ✅ Explainability with source highlighting and explanations
- ✅ Automated red-team testing framework

**Integration:**
- ✅ RAG service integration with hybrid search
- ✅ API endpoints for hybrid search (`/api/v1/search/hybrid`)
- ✅ Backward compatibility maintained
- ✅ Configuration via settings and environment variables

**Documentation:**
- ✅ Phase 2 implementation documentation
- ✅ API endpoint documentation
- ✅ Testing guides and examples
- ✅ Notebook cells demonstrating all features

## 🤝 Contributing

This is a portfolio project demonstrating production-ready AI system development with a focus on legal domain applications.

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 📧 Contact

- **Email**: mj.babaie@gmail.com
- **LinkedIn**: https://www.linkedin.com/in/mohammadbabaie/
- **GitHub**: https://github.com/Muh76

## 🎓 Learning Resources

### Phase 2 Notebooks

Explore the implementation details in the development notebooks:

- **Phase 1**: `notebooks/phase1/Foundation_MVP_Chatbot.ipynb` - MVP implementation
- **Phase 2**: `notebooks/phase2/Advanced_RAG.ipynb` - Advanced RAG features
  - Cell 13: Hybrid Retrieval Implementation
  - Cell 14: BM25 vs Semantic vs Hybrid Comparison
  - Cell 15: Metadata Filtering Examples
  - Cell 16: Performance Benchmarks
  - Cell 17: RAG Service Integration
  - Cell 18: Cross-Encoder Reranking
  - Cell 19: Explainability and Source Highlighting
  - Cell 20: Red Team Testing Automation

### Key Concepts Demonstrated

1. **Hybrid Retrieval**: Combining keyword (BM25) and semantic search for better results
2. **Fusion Strategies**: RRF (Reciprocal Rank Fusion) and weighted combination
3. **Reranking**: Using cross-encoders to improve retrieval accuracy
4. **Explainability**: Understanding why documents were retrieved
5. **Source Highlighting**: Visual indication of matched terms in documents
6. **Metadata Filtering**: Structured filtering for precise retrieval
7. **Red Team Testing**: Automated adversarial testing for safety validation

---

*This chatbot provides educational information only and does not constitute legal advice. Always consult with qualified legal professionals for specific legal matters.*
