# Legal Chatbot - AI-Powered Legal Assistant

A production-ready legal chatbot built with RAG (Retrieval-Augmented Generation) for UK legal system with future localization for Iranian market.

## 🎯 Project Overview

This project demonstrates end-to-end AI system development with:
- **Phase 1**: ✅ MVP chatbot with UK legal corpus (CUAD + Legislation.gov.uk) - **COMPLETE**
- **Phase 2**: ✅ Advanced RAG with hybrid retrieval, reranking, and explainability - **COMPLETE**
- **Phase 3**: ✅ Agentic RAG with LangChain agents and tool calling - **COMPLETE**
- **Phase 4.1**: ✅ Comprehensive testing and validation - **COMPLETE**
- **Phase 4.2**: ✅ Monitoring and observability - **COMPLETE**
- **Phase 5.1**: ✅ Database setup & migrations - **COMPLETE**
- **Phase 5.2**: ✅ Route protection with authentication & RBAC - **COMPLETE**
- **Phase 5.3**: ✅ Document upload system - **COMPLETE**
- **Phase 5.4**: ✅ Frontend role-based UI with authentication - **COMPLETE**
- **Phase 5**: 📋 Enterprise features and deployment (in progress)

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
   
   # Set up database (PostgreSQL required for authentication)
   export DATABASE_URL="postgresql://user:password@localhost:5432/legal_chatbot"
   export JWT_SECRET_KEY="your-jwt-secret-key"
   export SECRET_KEY="your-secret-key"
   
   # Run database migrations
   python -m alembic upgrade head
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
   - Streamlit UI: http://localhost:8501 (Requires authentication - login/register)
   - FastAPI docs: http://localhost:8000/docs (Swagger UI) - Public
   - ReDoc: http://localhost:8000/redoc - Public
   - API Health: http://localhost:8000/api/v1/health - Public (no auth required)
   - Detailed Health: http://localhost:8000/api/v1/health/detailed - Public
   - Metrics Summary: http://localhost:8000/api/v1/metrics/summary - Admin only
   - System Metrics: http://localhost:8000/api/v1/metrics/system - Admin only
   - Hybrid Search API: http://localhost:8000/api/v1/search/hybrid - Auth required
   - Agentic Chat API: http://localhost:8000/api/v1/agentic-chat - Auth required
   - Authentication: http://localhost:8000/api/v1/auth - Register, Login, OAuth

**Note**: The Streamlit frontend now requires authentication. When you first access http://localhost:8501, you'll see a login/register page. Create an account or login to access the chat interface.

5. **Ingest Data (if needed)**
   ```bash
   # Create FAISS index and embeddings
   python scripts/ingest_data.py
   ```

6. **Verify Authentication (Recommended)**
   ```bash
   # Quick verification - no API server required
   export DATABASE_URL="postgresql://javadbeni@localhost:5432/legal_chatbot"
   export JWT_SECRET_KEY="test-secret-key-for-testing"
   export SECRET_KEY="test-secret-key"
   python scripts/test_route_protection.py
   
   # Or use quick verification script
   chmod +x scripts/quick_verify_auth.sh
   ./scripts/quick_verify_auth.sh
   ```

7. **Test Document Upload System (New!)**
   ```bash
   # Test document upload functionality
   export DATABASE_URL="postgresql://javadbeni@localhost:5432/legal_chatbot"
   export JWT_SECRET_KEY="test-secret-key-for-testing"
   export SECRET_KEY="test-secret-key"
   python scripts/test_document_upload.py
   ```

## ✅ Phase 5: Authentication & Authorization - **COMPLETE**

### Phase 5.1: Database Setup & Migrations - ✅ **COMPLETE**
- ✅ PostgreSQL database setup
- ✅ Alembic migrations configured
- ✅ Authentication tables created (users, oauth_accounts, refresh_tokens)
- ✅ Database connection verified
- ✅ Migration scripts tested

### Phase 5.2: Route Protection with RBAC - ✅ **COMPLETE**
- ✅ All API routes protected with authentication
- ✅ Role-based access control (RBAC) implemented
- ✅ Public, Solicitor, Admin roles with appropriate permissions
- ✅ JWT token authentication
- ✅ Token refresh mechanism
- ✅ User activity logging

**Protected Endpoints:**
- ✅ `/api/v1/chat` - All authenticated users
- ✅ `/api/v1/search/hybrid` - All authenticated users
- ✅ `/api/v1/documents/*` - Solicitor/Admin only
- ✅ `/api/v1/agentic-chat` - Mode-based (Public mode: all users, Solicitor mode: Solicitor/Admin)
- ✅ `/api/v1/metrics/*` - Admin only

**Public Endpoints (No Auth Required):**
- ✅ `/api/v1/health` - Health checks
- ✅ `/docs` - API documentation (Swagger UI)
- ✅ `/redoc` - API documentation (ReDoc)

### Phase 5.3: Document Upload System - ✅ **COMPLETE**
- ✅ **Document Management**
  - PDF, DOCX, TXT document parsing and processing
  - User-specific document storage
  - Automatic chunking with overlap
  - Embedding generation and indexing
  - Status tracking (uploaded → parsing → processing → indexing → completed)
  
- ✅ **Private Corpus Search**
  - User-scoped document retrieval
  - Semantic search over private documents
  - Cosine similarity scoring
  - Combined public + private corpus search using RRF
  
- ✅ **API Endpoints**
  - `POST /api/v1/documents/upload` - Upload document (Solicitor/Admin)
  - `GET /api/v1/documents` - List documents with pagination
  - `GET /api/v1/documents/{id}` - Get document details with chunks
  - `PUT /api/v1/documents/{id}` - Update document metadata
  - `DELETE /api/v1/documents/{id}` - Delete document
  - `POST /api/v1/documents/{id}/reprocess` - Reprocess document
  
- ✅ **Integration**
  - Chat endpoint includes private corpus by default
  - Hybrid search combines public + private results
  - Metadata tagging (public/private corpus)
  - Source attribution with corpus information

### Phase 5.4: Frontend Role-Based UI - ✅ **COMPLETE**
- ✅ **Authentication UI Components**
  - Login/Register forms with validation
  - Email/password authentication
  - OAuth integration (Google, GitHub, Microsoft)
  - Token storage in Streamlit session state
  - Automatic token refresh handling
  
- ✅ **Protected Routes**
  - Authentication guards for all protected pages
  - Automatic redirect to login if not authenticated
  - Role-based access control in UI
  - Protected pages: Chat, Documents, Settings
  
- ✅ **Role-Based UI Rendering**
  - User profile display in sidebar
  - Role badges (Admin, Solicitor, Public)
  - Mode selection based on role:
    - Public users: Only "public" mode
    - Solicitor/Admin: Both "solicitor" and "public" modes
  - Conditional UI elements based on user permissions
  
- ✅ **User Management**
  - User profile display with role badge
  - Profile update in Settings page
  - Password change functionality
  - Logout with token revocation
  
- ✅ **Document Management UI**
  - Document list with metadata
  - Document upload form (Solicitor/Admin only)
  - Role-based access warnings
  - File uploader (PDF, DOCX, TXT)
  
- ✅ **Features**
  - Token expiration tracking (60-second buffer)
  - Automatic token refresh before expiration
  - Token refresh on API 401 errors
  - OAuth callback handling
  - Error handling and user feedback

**Document Upload Usage:**
```bash
# 1. Register/Login to get JWT token
curl -X POST http://localhost:8000/api/v1/auth/register \
  -H "Content-Type: application/json" \
  -d '{
    "email": "solicitor@example.com",
    "password": "securepassword123",
    "full_name": "Test Solicitor",
    "role": "solicitor"
  }'

# 2. Login to get access token
curl -X POST http://localhost:8000/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{
    "email": "solicitor@example.com",
    "password": "securepassword123"
  }'

# 3. Upload document (requires Solicitor/Admin role)
curl -X POST http://localhost:8000/api/v1/documents/upload \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN" \
  -F "file=@/path/to/document.pdf" \
  -F "title=My Legal Document" \
  -F "description=Test document" \
  -F "jurisdiction=UK" \
  -F "tags=contract,employment"

# 4. List documents
curl -X GET "http://localhost:8000/api/v1/documents?skip=0&limit=10" \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN"

# 5. Get document details
curl -X GET "http://localhost:8000/api/v1/documents/{document_id}" \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN"

# 6. Chat with private corpus (automatic inclusion)
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are my rights according to my uploaded documents?",
    "top_k": 10
  }'
```

**Testing:**
```bash
# Test document upload system
python scripts/test_document_upload.py

# Expected output:
# ✅ Database connection successful
# ✅ Document tables exist
# ✅ Storage service working
# ✅ Document parsing working
# ✅ Document upload and processing working
# ✅ Document retrieval working
# ✅ Private corpus search working
```

**Verification:**
```bash
# Quick verification test (recommended first)
export DATABASE_URL="postgresql://javadbeni@localhost:5432/legal_chatbot"
export JWT_SECRET_KEY="test-secret-key-for-testing"
export SECRET_KEY="test-secret-key"
python scripts/test_route_protection.py

# Full HTTP endpoint test (requires API server running)
uvicorn app.api.main:app --reload &
export API_BASE_URL="http://localhost:8000"
python scripts/test_api_endpoints.py

# Or use quick verification script
chmod +x scripts/quick_verify_auth.sh
./scripts/quick_verify_auth.sh
```

**Verification Results**: ✅ All tests passing
- ✅ Authentication dependencies working correctly
- ✅ All routes properly protected
- ✅ Role-based access control enforced
- ✅ Token creation and verification working
- ✅ Token refresh working

See [Verification Guide](docs/verification_guide.md) for detailed testing instructions.

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

### Phase 3 (Agentic RAG) - ✅ **COMPLETED**
- ✅ **LangChain Agent Integration**
  - OpenAI Functions Agent for tool calling
  - Autonomous tool selection based on query
  - Multi-step reasoning and information gathering
  - Iterative refinement until sufficient information
  - ReAct (Reasoning + Acting) pattern implementation
  
- ✅ **Tool System**
  - `search_legal_documents`: Hybrid search tool with metadata filtering
  - `get_specific_statute`: Statute lookup by name
  - `analyze_document`: Document analysis and summarization
  - Extensible tool architecture for future capabilities
  
- ✅ **Agentic Chat API**
  - POST `/api/v1/agentic-chat` endpoint
  - Tool call tracking and reporting
  - Conversation history support
  - Solicitor and public modes
  - Safety validation with guardrails
  
- ✅ **Benefits of Agentic RAG**
  - Handles complex, multi-part queries automatically
  - Autonomous decision-making for tool selection
  - Multi-step problem solving (e.g., "Compare X with Y")
  - Better reasoning through iterative refinement
  - Extensible architecture for adding new tools

### Phase 4.1 (Comprehensive Testing) - ✅ **COMPLETED**
- ✅ End-to-end test suite (all API endpoints)
- ✅ Integration testing (service-to-service communication)
- ✅ Regression testing (backward compatibility)
- ✅ Performance benchmarking
- ✅ Error handling enhancement
- ✅ Load testing framework

### Phase 4.2 (Monitoring and Observability) - ✅ **COMPLETED**
- ✅ **Application Logging**
  - Structured JSON logging (configurable)
  - Log levels (DEBUG, INFO, WARNING, ERROR)
  - Request/response logging with unique IDs
  - Error tracking with full exception details
  - Thread-safe logging with rotation
  
- ✅ **Health Checks**
  - Service health endpoints (`/api/v1/health`)
  - Detailed health check with dependencies (`/api/v1/health/detailed`)
  - Kubernetes liveness probe (`/api/v1/health/live`)
  - Kubernetes readiness probe (`/api/v1/health/ready`)
  - Dependency health monitoring (database, redis, vector store, LLM API)
  - System metrics (CPU, memory, disk)
  
- ✅ **Metrics and Alerts**
  - API response times tracking (min, max, avg)
  - Error rates per endpoint
  - Request volumes tracking
  - Tool usage statistics for agentic chat
  - System metrics collection (CPU, memory, disk)
  - Summary metrics endpoint
  
- ✅ **Monitoring API Endpoints**
  - `GET /api/v1/metrics` - All metrics
  - `GET /api/v1/metrics/summary` - Summary metrics
  - `GET /api/v1/metrics/endpoints` - Endpoint-specific metrics
  - `GET /api/v1/metrics/tools` - Tool usage statistics
  - `GET /api/v1/metrics/system` - System metrics

### Phase 5 (Enterprise Features) - 🔄 **IN PROGRESS**
- ✅ OAuth2 authentication and RBAC - **COMPLETE**
- ✅ Document upload and private corpora - **COMPLETE**
- 🔄 Multi-tenant architecture - **PLANNED**
- 🔄 Multilingual support (English + Farsi) - **PLANNED**
- 🔄 Billing and monetization hooks - **PLANNED**

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
  - BM25 (keyword-based retrieval)
- **Agentic**: 
  - LangChain (agent framework)
  - LangChain OpenAI (function calling)
  - LangGraph (workflow orchestration)
  - BM25 (keyword search)
  - Cross-encoder reranking
- **Vector DB**: FAISS (in-memory), Qdrant/pgvector (optional)
- **Infrastructure**: Docker, Docker Compose
- **Monitoring**: 
  - Structured JSON logging (Loguru)
  - Health checks with dependency monitoring
  - Metrics collection (API response times, error rates, request volumes)
  - System metrics (CPU, memory, disk via psutil)
  - Tool usage statistics (agentic chat)
  - Prometheus, Grafana, OpenTelemetry (planned integration)
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
- [Phase 3 Agentic RAG Summary](docs/phase3_agentic_rag_summary.md)
- [Phase 4.1 Testing Summary](docs/phase4_1_testing_summary.md)
- [Phase 4.2 Monitoring Summary](docs/phase4_2_monitoring_summary.md)
- [Monitoring Verification Guide](docs/verify_monitoring.md)
- [Monitoring API Usage Guide](MONITORING_API_USAGE.md)
- [Phase 5.1 Authentication Summary](docs/phase5_1_authentication_summary.md)
- [Phase 5.2 Route Protection](docs/phase5_2_route_protection_complete.md)
- [Phase 5.3 Document Upload System](docs/phase5_3_document_upload_complete.md)
- [Phase 5.4 Frontend Role-Based UI](docs/phase5_frontend_auth_complete.md)

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

## 🤖 Phase 3: Agentic RAG Features in Detail

### Agentic Chat API

**POST `/api/v1/agentic-chat`**

Simple query example:
```bash
curl -X POST http://localhost:8000/api/v1/agentic-chat \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the Sale of Goods Act 1979?",
    "mode": "public"
  }'
```

Complex multi-tool query example:
```bash
curl -X POST http://localhost:8000/api/v1/agentic-chat \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Compare the Sale of Goods Act 1979 with the Consumer Rights Act 2015",
    "mode": "public",
    "chat_history": []
  }'
```

With conversation history:
```bash
curl -X POST http://localhost:8000/api/v1/agentic-chat \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What about the Consumer Rights Act 2015?",
    "mode": "public",
    "chat_history": [
      {"role": "user", "content": "Tell me about the Sale of Goods Act 1979"},
      {"role": "assistant", "content": "The Sale of Goods Act 1979 is..."}
    ]
  }'
```

**Response includes:**
- `answer`: Generated answer with citations
- `tool_calls`: List of tools used by the agent (e.g., `search_legal_documents`, `get_specific_statute`)
- `iterations`: Number of agent reasoning steps (typically 1-3)
- `intermediate_steps_count`: Total number of reasoning steps
- `confidence_score`: Confidence in the answer (0.0-1.0)
- `safety`: Safety validation results
- `metrics`: Performance metrics (retrieval time, generation time, etc.)

**Example Response:**
```json
{
  "answer": "The Sale of Goods Act 1979 and Consumer Rights Act 2015 both govern contracts for the sale of goods...",
  "tool_calls": [
    {
      "tool": "get_specific_statute",
      "input": {"statute_name": "Sale of Goods Act 1979"},
      "result": "Found 5 sections from 'Sale of Goods Act 1979'..."
    },
    {
      "tool": "get_specific_statute",
      "input": {"statute_name": "Consumer Rights Act 2015"},
      "result": "Found 8 sections from 'Consumer Rights Act 2015'..."
    }
  ],
  "iterations": 2,
  "intermediate_steps_count": 2,
  "confidence_score": 0.9,
  "mode": "public",
  "safety": {
    "is_safe": true,
    "flags": [],
    "confidence": 0.95,
    "reasoning": "Query passed validation"
  },
  "metrics": {
    "retrieval_time_ms": 1200.5,
    "generation_time_ms": 3500.2,
    "total_time_ms": 4700.7,
    "retrieval_score": 0.85,
    "answer_relevance_score": 0.9
  }
}
```

### Agentic Chat vs Traditional Chat

| Feature | Traditional (`/api/v1/chat`) | Agentic (`/api/v1/agentic-chat`) |
|---------|------------------------------|----------------------------------|
| **Tool calling** | ❌ No | ✅ Yes (autonomous) |
| **Multi-step reasoning** | ❌ No | ✅ Yes (iterative) |
| **Complex queries** | ⚠️ Limited | ✅ Excellent |
| **Tool selection** | ❌ Hardcoded | ✅ Autonomous |
| **Iteration tracking** | ❌ No | ✅ Yes |
| **Conversation history** | ✅ Basic | ✅ Full support |
| **Multi-tool queries** | ❌ No | ✅ Yes (e.g., "Compare X with Y") |
| **Query breakdown** | ❌ No | ✅ Yes (breaks down complex queries) |

### Available Tools

The agentic chatbot has access to the following tools:

1. **`search_legal_documents`**
   - Purpose: Search UK legal documents using hybrid search
   - Parameters: `query`, `jurisdiction` (optional), `document_type` (optional), `top_k`
   - Returns: Relevant legal chunks with citations and relevance scores

2. **`get_specific_statute`**
   - Purpose: Look up a specific UK statute by name
   - Parameters: `statute_name`
   - Returns: Relevant sections from the specified statute

3. **`analyze_document`**
   - Purpose: Analyze legal documents (placeholder for future features)
   - Parameters: `document_text`, `analysis_type` (optional)
   - Returns: Document summary or analysis

### How Agentic RAG Works

1. **User Query**: User submits a query
2. **Agent Analysis**: LLM analyzes the query and decides which tools to use
3. **Tool Execution**: Agent calls appropriate tools (e.g., `get_specific_statute`)
4. **Result Observation**: Agent observes tool results
5. **Reasoning**: Agent decides if more information is needed
6. **Iteration**: If needed, agent calls additional tools
7. **Final Answer**: Agent generates comprehensive answer with citations

**Example Flow:**
```
Query: "Compare the Sale of Goods Act 1979 with the Consumer Rights Act 2015"

Step 1: Agent reasons → "I need both statutes"
Step 2: Agent acts → Calls get_specific_statute("Sale of Goods Act 1979")
Step 3: Agent observes → Receives first statute sections
Step 4: Agent reasons → "I need the second statute"
Step 5: Agent acts → Calls get_specific_statute("Consumer Rights Act 2015")
Step 6: Agent observes → Receives second statute sections
Step 7: Agent reasons → "I have both, can now compare"
Step 8: Agent generates → Comprehensive comparison with citations
```

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

**Phase 2 Tests:**
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

**Phase 3 Tests (Agentic RAG):**
```bash
# Test agentic chat imports and registration
python scripts/test_agentic_import.py

# Test agentic chat API (requires server running)
python scripts/test_agentic_chat.py

# Test via curl (simple query)
curl -X POST http://localhost:8000/api/v1/agentic-chat \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the Sale of Goods Act 1979?", "mode": "public"}'

# Test agent stats endpoint
curl http://localhost:8000/api/v1/agentic-chat/stats
```

**Phase 4.1 Tests (Comprehensive Testing):**
```bash
# Run all E2E tests
pytest tests/e2e/ -v

# Run integration tests
pytest tests/integration/ -v

# Run performance benchmarks
python scripts/test_performance_benchmark.py

# Run all tests
python scripts/run_all_tests.py
```

**Phase 4.2 Tests (Monitoring):**
```bash
# Run monitoring unit tests
pytest tests/unit/test_monitoring.py -v

# Verify monitoring (requires server running)
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
python scripts/verify_monitoring.py --url http://localhost:8000

# E2E monitoring verification
bash scripts/test_monitoring_e2e.sh

# Test health endpoints
curl http://localhost:8000/api/v1/health
curl http://localhost:8000/api/v1/health/detailed
curl http://localhost:8000/api/v1/health/live
curl http://localhost:8000/api/v1/health/ready

# Test metrics endpoints
curl http://localhost:8000/api/v1/metrics/summary
curl http://localhost:8000/api/v1/metrics/system
curl http://localhost:8000/api/v1/metrics/endpoints
curl http://localhost:8000/api/v1/metrics/tools
```

**Phase 5 Tests (Enterprise Features):**
```bash
# Test authentication (requires database)
python scripts/test_auth_database.py

# Test route protection
python scripts/test_route_protection.py

# Test document upload system
python scripts/test_document_upload.py

# Test frontend authentication components
python scripts/test_frontend_auth.py

# Test frontend integration (requires API server)
python scripts/test_frontend_integration.py
```

## 📋 Project Status

- **Phase 1**: ✅ **COMPLETED** - MVP with RAG pipeline, guardrails, and web interface
- **Phase 2**: ✅ **COMPLETED** - Advanced RAG with hybrid retrieval, reranking, explainability, and red-team testing  
- **Phase 3**: ✅ **COMPLETED** - Agentic RAG with LangChain agents, tool calling, and multi-step reasoning
- **Phase 4.1**: ✅ **COMPLETED** - Comprehensive testing and validation (end-to-end, integration, regression, performance)
- **Phase 4.2**: ✅ **COMPLETED** - Monitoring and observability (structured logging, health checks, metrics collection)
- **Phase 5**: 🔄 **IN PROGRESS** - Enterprise features and deployment
  - ✅ Phase 5.1: Database setup & migrations
  - ✅ Phase 5.2: Route protection with authentication & RBAC
  - ✅ Phase 5.3: Document upload system
  - ✅ Phase 5.4: Frontend role-based UI with authentication

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

### Phase 3 Completion Summary (Agentic RAG)

**Core Features:**
- ✅ LangChain agent integration with LangGraph (ReAct pattern)
- ✅ Tool system wrapping Phase 2 RAG capabilities
- ✅ Autonomous tool selection and multi-step reasoning
- ✅ Iterative refinement until sufficient information
- ✅ Conversation history support
- ✅ Safety validation with guardrails integration
- ✅ LangChain 1.0+ compatibility with fallback support

**Tools Available:**
- ✅ `search_legal_documents`: Hybrid search with metadata filtering
- ✅ `get_specific_statute`: Look up specific UK statutes by name
- ✅ `analyze_document`: Document analysis and summarization (placeholder)

**Integration:**
- ✅ Agentic chat API endpoint (`/api/v1/agentic-chat`)
- ✅ Agent stats endpoint (`/api/v1/agentic-chat/stats`)
- ✅ Tool call tracking and reporting in responses
- ✅ Solicitor and public modes
- ✅ Backward compatible with existing RAG system (traditional `/chat` endpoint still works)

**Key Improvements Over Traditional RAG:**
- ✅ Handles complex, multi-part queries automatically
- ✅ Autonomous decision-making for tool selection (not hardcoded)
- ✅ Multi-step problem solving (e.g., "Compare X with Y" requires multiple tool calls)
- ✅ Better reasoning through iterative refinement
- ✅ Extensible architecture for adding new tools

**Documentation:**
- ✅ Agentic service implementation (`app/services/agent_service.py`)
- ✅ Tool system architecture (`app/tools/`)
- ✅ API endpoint documentation (`app/api/routes/agentic_chat.py`)
- ✅ Test scripts for agentic chat (`scripts/test_agentic_chat.py`)
- ✅ Phase 3 implementation summary (`docs/phase3_agentic_rag_summary.md`)

### Phase 4.1 Completion Summary (Comprehensive Testing)

**Core Features:**
- ✅ End-to-end test suite for all API endpoints
- ✅ Integration testing for service-to-service communication
- ✅ Regression testing for backward compatibility
- ✅ Performance benchmarking with targets
- ✅ Error handling enhancement
- ✅ Load testing framework

**Test Coverage:**
- ✅ E2E tests: 108+ test cases
- ✅ Integration tests: 15+ test cases
- ✅ Regression tests: 20+ test cases
- ✅ Performance benchmarks: Response time tracking
- ✅ Error scenario tests: Network failures, invalid inputs, timeouts

**Integration:**
- ✅ Test runner scripts (`scripts/run_all_tests.py`)
- ✅ Server management for tests (`scripts/run_tests_with_server.py`)
- ✅ Performance benchmark script (`scripts/test_performance_benchmark.py`)
- ✅ Comprehensive test documentation

**Documentation:**
- ✅ Phase 4.1 testing summary (`docs/phase4_1_testing_summary.md`)
- ✅ Test documentation (`tests/README.md`)
- ✅ Test execution reports

### Phase 4.2 Completion Summary (Monitoring and Observability)

**Core Features:**
- ✅ Structured JSON logging (configurable)
- ✅ Request/response logging with unique IDs
- ✅ Error tracking with full exception details
- ✅ Health checks with dependency monitoring (4 endpoints)
- ✅ System metrics (CPU, memory, disk)
- ✅ API metrics (response times, error rates, request volumes)
- ✅ Tool usage statistics for agentic chat (5 endpoints)

**Health Checks:**
- ✅ Basic health check (`GET /api/v1/health`)
- ✅ Detailed health check with dependencies (`GET /api/v1/health/detailed`)
- ✅ Kubernetes liveness probe (`GET /api/v1/health/live`)
- ✅ Kubernetes readiness probe (`GET /api/v1/health/ready`)
- ✅ Dependency monitoring (database, redis, vector store, LLM API)
- ✅ Parallel dependency checking with caching

**Metrics Collection:**
- ✅ API request tracking (automatic via middleware)
- ✅ Response time statistics (min, max, avg per endpoint)
- ✅ Error rate calculation per endpoint
- ✅ Request volume tracking
- ✅ Tool usage statistics (execution time, success/failure rates)
- ✅ System metrics (CPU percentage, memory usage, disk usage)

**Logging:**
- ✅ Structured JSON format (configurable)
- ✅ Log levels (DEBUG, INFO, WARNING, ERROR)
- ✅ Request/response logging with unique request IDs
- ✅ Error tracking with full exception details
- ✅ Thread-safe logging with rotation (10 MB, 7 days retention)
- ✅ Custom headers (X-Request-ID, X-Process-Time)

**Integration:**
- ✅ Monitoring middleware integrated (`app/core/middleware.py`)
- ✅ Metrics collector (`app/core/metrics.py`)
- ✅ Health checker (`app/core/health_checker.py`)
- ✅ Enhanced logging (`app/core/logging.py`)
- ✅ Metrics API endpoints (`app/api/routes/metrics.py`)
- ✅ Enhanced health endpoints (`app/api/routes/health.py`)

**Documentation:**
- ✅ Phase 4.2 monitoring summary (`docs/phase4_2_monitoring_summary.md`)
- ✅ Monitoring verification guide (`docs/verify_monitoring.md`)
- ✅ Monitoring API usage guide (`MONITORING_API_USAGE.md`)
- ✅ Verification checklist (`docs/MONITORING_VERIFICATION_CHECKLIST.md`)
- ✅ E2E verification results (`MONITORING_E2E_VERIFICATION_RESULTS.md`)

**Verification:**
- ✅ Unit tests: 20/20 passed
- ✅ All 9 API endpoints working
- ✅ Metrics collection verified (42+ requests tracked)
- ✅ System metrics verified (CPU, memory, disk)
- ✅ Request/response logging verified
- ✅ Custom headers verified

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

**Phase 2 Concepts:**
1. **Hybrid Retrieval**: Combining keyword (BM25) and semantic search for better results
2. **Fusion Strategies**: RRF (Reciprocal Rank Fusion) and weighted combination
3. **Reranking**: Using cross-encoders to improve retrieval accuracy
4. **Explainability**: Understanding why documents were retrieved
5. **Source Highlighting**: Visual indication of matched terms in documents
6. **Metadata Filtering**: Structured filtering for precise retrieval
7. **Red Team Testing**: Automated adversarial testing for safety validation

**Phase 3 Concepts:**
1. **Agentic RAG**: LLM agents with tool calling for autonomous information gathering
2. **ReAct Pattern**: Reasoning + Acting loop for multi-step problem solving
3. **Tool System**: LangChain tools wrapping existing RAG capabilities
4. **Multi-Step Reasoning**: Iterative refinement until sufficient information
5. **Autonomous Tool Selection**: Agent decides which tools to use based on query analysis
6. **Conversation History**: Context-aware responses across multiple turns

**Phase 4.2 Concepts:**
1. **Structured Logging**: JSON format logs for machine readability and log aggregation
2. **Request Tracking**: Unique request IDs for tracing requests through the system
3. **Health Checks**: Dependency monitoring for database, cache, vector store, and LLM API
4. **Metrics Collection**: Automatic tracking of API response times, error rates, and request volumes
5. **System Metrics**: Real-time monitoring of CPU, memory, and disk usage
6. **Tool Usage Statistics**: Tracking tool execution times and success/failure rates in agentic chat
7. **Observability**: Comprehensive visibility into system behavior for debugging and optimization

---

*This chatbot provides educational information only and does not constitute legal advice. Always consult with qualified legal professionals for specific legal matters.*
