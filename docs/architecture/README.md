# Legal Chatbot - Architecture Documentation

## System Architecture

### Overview
The Legal Chatbot is built as a microservices architecture with the following components:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Streamlit UI  │────│   FastAPI API   │────│   Vector DB     │
│   (Port 8501)   │    │   (Port 8000)   │    │   (Qdrant)      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                       ┌─────────────────┐
                       │   PostgreSQL    │
                       │   + pgvector    │
                       └─────────────────┘
```

### Components

#### 1. Frontend (Streamlit)
- **Location**: `frontend/streamlit/`
- **Purpose**: User interface for legal Q&A
- **Features**: Chat interface, source display, settings panel
- **Port**: 8501

#### 2. Backend API (FastAPI)
- **Location**: `app/api/`
- **Purpose**: RESTful API for chatbot functionality
- **Endpoints**:
  - `GET /api/v1/health` - Health check
  - `POST /api/v1/chat` - Chat with legal assistant
  - `POST /api/v1/documents/upload` - Upload documents
- **Port**: 8000

#### 3. Data Layer
- **PostgreSQL**: Main application data
- **pgvector**: Vector embeddings storage
- **Qdrant**: Alternative vector database
- **Redis**: Caching layer

#### 4. ML Pipeline
- **Embeddings**: SentenceTransformers (all-MiniLM-L6-v2)
- **LLM**: OpenAI GPT-4
- **RAG**: Retrieval-Augmented Generation
- **Guardrails**: Safety and domain validation

### Data Flow

```
User Query → Streamlit UI → FastAPI API → RAG Pipeline → Vector DB
                ↓              ↓              ↓
            Response ← JSON Response ← Generated Answer ← Retrieved Docs
```

### Security Architecture

#### Authentication & Authorization
- JWT tokens for API access
- Role-based access control (RBAC)
- Multi-tenant data isolation

#### Data Protection
- PII redaction using Presidio
- GDPR/UK GDPR compliance
- Audit logging for all operations
- Data retention policies

#### API Security
- CORS configuration
- Rate limiting
- Input validation
- SQL injection prevention

### Monitoring & Observability

#### Metrics
- Response time (P50, P95, P99)
- Retrieval accuracy
- Safety flag rates
- User engagement

#### Logging
- Structured JSON logs
- Correlation IDs
- Error tracking
- Performance monitoring

#### Alerting
- Prometheus metrics
- Grafana dashboards
- Slack notifications
- Health check monitoring

### Deployment Architecture

#### Development
- Docker Compose for local development
- Hot reload for API and frontend
- Local databases (PostgreSQL, Redis, Qdrant)

#### Production
- Kubernetes deployment
- Auto-scaling based on load
- Load balancing
- Database clustering
- CDN for static assets

### Phase Progression

#### Phase 1: MVP
- Basic RAG with UK legal corpus
- Simple safety guardrails
- Single-tenant architecture

#### Phase 2: Advanced RAG
- Hybrid retrieval (dense + BM25)
- Cross-encoder reranking
- Explainability features

#### Phase 3: Localization
- Multilingual support (English + Farsi)
- Role-based responses
- Document upload functionality

#### Phase 4: Enterprise
- Multi-tenant architecture
- OAuth2 authentication
- Advanced monitoring
- Billing integration
