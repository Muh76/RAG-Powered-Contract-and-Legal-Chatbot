# Legal Chatbot — Retrieval-Augmented Legal Assistant for UK Law

## 1. Project Overview

This repository contains a **production-oriented RAG (Retrieval-Augmented Generation) system** that answers UK legal questions, with an emphasis on:

- **Grounded answers**: responses are based on retrieved legal documents, not model memorization.
- **Citations**: every sentence is expected to end with citation tokens like `[1]`, `[2]` that map back to specific chunks.
- **Safety and observability**: guardrails, red-team tests, health checks, metrics, and structured logging.

The system is implemented as a FastAPI backend with a Streamlit frontend, a hybrid retrieval layer (BM25 + FAISS + OpenAI embeddings), and an OpenAI-powered LLM layer for answer generation. Authentication, role-based access, and document upload are implemented for a realistic production setup.

---

## 2. Key Features

- **Hybrid RAG pipeline**: BM25 lexical search + FAISS vector search with OpenAI embeddings, fused via Reciprocal Rank Fusion.
- **Citation-enforced LLM answers**: strict prompts plus post-hoc validation and a repair loop ensure every sentence ends with valid `[n]` citations.
- **Agentic RAG (optional path)**: LangChain/LangGraph-based agent with tools (legal search, statute lookup, document analyzer) for multi-step queries.
- **Private + public corpus**: user document upload and retrieval combined with a shared legal corpus.
- **Authentication & RBAC**: JWT-based auth, OAuth provider flow, and role-based access for public vs solicitor/admin features.
- **Monitoring & health**: request/response logging middleware, metrics collector, dependency health checks (DB, Redis, Qdrant/OpenAI).
- **Dockerized backend**: container built around uvicorn/FastAPI, ready for environments such as Google Cloud Run.

All of these are implemented in code under `app/`, `retrieval/`, `ingestion/`, `frontend/`, and `scripts/`.

---

## 3. System Architecture

At a high level:

1. A user interacts with the **Streamlit frontend**.
2. The frontend calls the **FastAPI backend**.
3. The backend:
   - runs **guardrails** on the query,
   - executes the **RAG retrieval pipeline**,
   - constructs a prompt with numbered sources,
   - calls the **OpenAI Chat API** to generate an answer,
   - validates citations and applies additional guardrails,
   - returns a structured response containing answer, sources, safety info, and metrics.

Document ingestion and indexing are handled offline via scripts using loaders and chunkers, OpenAI embeddings, and FAISS.

### Architecture Diagram

The diagram below represents the implemented architecture:

```mermaid
flowchart TD
    user(User) --> fe[Streamlit Frontend]
    fe --> api[FastAPI Backend<br/>app/api/main.py]

    subgraph RAG[RAG Pipeline]
      direction TB
      qp[Query Processing & Guardrails<br/>app/services/guardrails_service.py] --> emb[Embedding (OpenAI)<br/>retrieval/embeddings/openai_embedding_generator.py]
      emb --> hr[Hybrid Retrieval<br/>BM25 + FAISS<br/>app/services/rag_service.py<br/>retrieval/hybrid_retriever.py]
      hr --> rr[Reranking (optional)<br/>retrieval/rerankers/cross_encoder_reranker.py]
      rr --> llm[LLM Generation + Citation Validation<br/>app/services/llm_service.py]
    end

    api --> qp
    llm --> api

    subgraph DOC[Document Processing & Indexing]
      direction TB
      load[Loaders (PDF/TXT/JSON/CUAD)<br/>ingestion/loaders/document_loaders.py] --> chunk[Chunking<br/>ingestion/chunkers/document_chunker.py]
      chunk --> demb[Embeddings (OpenAI)<br/>retrieval/embeddings/openai_embedding_generator.py]
      demb --> faiss[FAISS Index + Metadata<br/>data/faiss_index.bin<br/>data/chunk_metadata.pkl]
    end

    subgraph SEC[Security & Guardrails]
      direction TB
      auth[JWT Auth + OAuth + RBAC<br/>app/auth/*] --> guards[Domain & Safety Guardrails<br/>app/services/guardrails_service.py]
    end

    subgraph MON[Evaluation & Monitoring]
      direction TB
      tests[E2E & Red-team Tests<br/>tests/*, retrieval/red_team_tester.py] --> metrics[Metrics & Health Checks<br/>app/core/metrics.py<br/>app/core/health_checker.py]
    end

    subgraph INFRA[Infrastructure]
      direction TB
      docker[Docker Container<br/>Dockerfile] --> run[Cloud Run-ready Deployment Script<br/>scripts/deploy.sh]
    end
```

> Note: A rendered PNG diagram can be generated from this Mermaid spec and saved as `docs/architecture.png`. The README references this path for GitHub display.

![Architecture](docs/architecture.png)

---

## 4. Tech Stack

**Language & Core**
- Python 3.11

**Backend**
- FastAPI (REST API, routing)
- Uvicorn (ASGI server)
- SQLAlchemy + Alembic (database ORM + migrations)
- PostgreSQL (primary database; connection string via `DATABASE_URL`)
- Optional Redis integration (caching/health checks)

**Frontend**
- Streamlit (`frontend/app.py`, `frontend/streamlit/app.py`) for chat UI and basic dashboards.

**Retrieval & ML / LLM**
- OpenAI Chat API (GPT-style models) for answer generation.
- OpenAI embeddings (`text-embedding-3-large`) via `retrieval/embeddings/openai_embedding_generator.py`.
- FAISS CPU for in-process vector search.
- `rank-bm25` for BM25 lexical search.
- Optional cross-encoder reranker (sentence-transformers) via `retrieval/rerankers/cross_encoder_reranker.py`.
- LangChain / LangGraph for agentic workflows (`app/services/agent_service.py`).

**Security & Auth**
- `python-jose` and `passlib` for JWTs and password hashing.
- OAuth provider integration via `app/auth/oauth.py` and `app/api/routes/auth.py`.
- Role-based access via `UserRole` in `app/auth/models.py` and dependency helpers.

**Monitoring & Observability**
- Loguru-based structured logging (`app/core/logging.py`, `app/core/middleware.py`).
- Prometheus-style metrics (`app/core/metrics.py`).
- Health checks for DB, Redis, Qdrant/OpenAI (`app/core/health_checker.py`).

**Evaluation & Safety**
- Red-team testing (`retrieval/red_team_tester.py`, `scripts/test_red_team.py`).
- Extensive tests under `tests/` and various `scripts/test_*.py`.
- RAGAS and TruLens are included as dependencies and available for use, but are not directly wired into runtime code.

**Infrastructure**
- Dockerfile for containerizing the backend.
- `scripts/deploy.sh` for building and deploying to Google Cloud Run using Cloud Build and Artifact Registry.

---

## 5. Project Structure

Key directories and files:

```text
app/
  api/
    main.py               # FastAPI app and router registration
    routes/
      chat.py             # /api/v1/chat endpoint (RAG pipeline orchestration)
      search.py           # /api/v1/search (hybrid search API)
      documents.py        # /api/v1/documents (upload, list)
      auth.py             # /api/v1/auth (register, login, OAuth, profile)
      health.py           # /api/v1/health* (liveness/readiness)
      metrics.py          # /api/v1/metrics (exposes metrics)
      agentic_chat.py     # /api/v1/agentic-chat (agentic RAG, LangChain)
      debug.py            # Debug and diagnostic routes

  services/
    rag_service.py        # RAGService: loads FAISS + metadata, runs hybrid/semantic/TF-IDF search
    llm_service.py        # LLMService: constructs prompts, calls OpenAI, enforces citations
    guardrails_service.py # GuardrailsService: domain gating, harmful content, response validation
    agent_service.py      # AgenticRAGService: LangChain/LangGraph agent with tools

  tools/
    legal_search_tool.py      # Tool wrapping hybrid RAG search
    statute_lookup_tool.py    # Tool for statute lookup
    document_analyzer_tool.py # Tool for analyzing uploaded documents

  documents/
    models.py      # SQLAlchemy models for documents and chunks
    schemas.py     # Pydantic schemas for document API
    service.py     # DocumentService: upload, chunk, and search private corpus
    storage.py     # File system storage integration
    parsers.py     # Helpers to parse different formats

  core/
    config.py      # Pydantic Settings (env vars for API, DB, embeddings, hybrid config)
    database.py    # DB engine/session, helper dependencies
    health_checker.py
    logging.py
    metrics.py
    middleware.py  # Request/response logging + error tracking middleware
    errors.py      # Custom exception types

  auth/
    models.py      # User, OAuthAccount, RefreshToken, UserRole (RBAC)
    schemas.py     # Pydantic auth schemas
    jwt.py         # JWT creation/verification
    oauth.py       # OAuth provider logic
    dependencies.py# FastAPI dependencies (get_current_user, require_admin, etc.)
    service.py     # AuthService: registration, login, tokens, password change

  models/
    schemas.py     # Core API models (ChatRequest, ChatResponse, Source, HealthResponse, etc.)

frontend/
  app.py               # Streamlit chat UI and dashboard
  streamlit/app.py     # Alternative/structured Streamlit frontend
  auth_ui.py           # Auth UI helpers
  components/          # Common UI components

ingestion/
  loaders/
    document_loaders.py  # Loads PDFs, text, json, CUAD data, etc. into DocumentChunk objects
  chunkers/
    document_chunker.py  # Chunking and section-based splitting for legal docs

retrieval/
  embeddings/
    openai_embedding_generator.py  # Calls OpenAI embeddings API with backoff and batching
  bm25_retriever.py           # BM25 lexical search
  tfidf_retriever.py          # TF-IDF-only retriever (fallback)
  semantic_retriever.py       # Semantic retriever using local embeddings (not main path)
  openai_semantic_retriever.py# Semantic retriever using OpenAI + FAISS
  hybrid_retriever.py         # AdvancedHybridRetriever (BM25 + semantic + fusion + optional rerank)
  rerankers/
    cross_encoder_reranker.py # Cross-encoder reranker
  metadata_filter.py          # Metadata-based filters
  explainability.py           # ExplainabilityAnalyzer for retrieval results
  red_team_tester.py          # Red team/safety testing helper

scripts/
  ingest_data.py              # Build/update FAISS index and chunk_metadata
  ingest_data_fixed.py        # Alternative/updated ingestion
  test_*.py                   # Various manual/diagnostic test scripts
  deploy.sh                   # Cloud Run deployment script
  run_all_tests.py, run_tests_with_server.py, smoke_test_e2e.py, etc.

tests/
  unit/, integration/, e2e/   # Automated tests (chat, search, auth, monitoring, etc.)

Dockerfile                    # Production container definition
requirements.txt              # Pinned dependencies
```

---

## 6. Retrieval Pipeline (RAG Workflow)

The RAG workflow for the main `/api/v1/chat` endpoint (implemented in `app/api/routes/chat.py`) is:

1. **Authentication & context**
   - JWT or OAuth-based user context.
   - Role-based mode (public vs solicitor/admin) influences system prompt.

2. **Query guardrails**
   - `GuardrailsService.validate_query()`:
     - Domain gating: is the query about legal topics?
     - Harmful content patterns (self-harm, terrorism, hate, etc.).
     - Minimum legal relevance.
   - If invalid, returns a safe refusal with explanation and no retrieval/LLM call.

3. **Private corpus search (optional)**
   - `DocumentService.search_user_documents()` retrieves matches from the user’s own uploaded documents (if enabled), based on BM25/TF-IDF over private chunks.

4. **Public corpus retrieval**
   - `RAGService.search()` orchestrates:
     - Loading chunk metadata and FAISS index.
     - Hybrid search via `AdvancedHybridRetriever` if FAISS + embeddings are available:
       - BM25 retriever over chunk text.
       - Semantic retriever (OpenAI + FAISS) or fallback to TF-IDF-only.
       - Fusion via RRF or weighted strategy.
       - Optional cross-encoder reranking.
     - Fallback path when FAISS or embeddings are missing: TF-IDF-only retrieval.

5. **Public + private combination**
   - When private results are available, `RAGService._combine_results()` merges public and private results with RRF and truncates to top-k.

6. **Similarity-based gate**
   - Compute average similarity of final retrieved chunks.
   - If similarity is below a configured threshold, the system returns a safe message instead of calling the LLM (reduces hallucinations on weak grounding).

7. **Context assembly**
   - Each retrieved chunk is formatted as:
     - `[n] Title/Act Name - Section\nchunk text`
   - Chunks are concatenated to form the context; structured `Source` objects are also created for response metadata.

8. **LLM call & citation enforcement**
   - `LLMService.generate_legal_answer()`:
     - Builds a **system prompt** with:
       - Role (legal assistant for UK law).
       - Strong disclaimer (not legal advice).
       - Strict citation rules (every sentence must end with `[n]`, only use provided sources).
       - Anti-hallucination instructions (no external knowledge, no invented law).
     - Builds a **user prompt** containing:
       - SOURCES (numbered [1], [2], …).
       - The user’s question.
       - Additional formatting/citation constraints.
     - Calls `client.chat.completions.create()` (OpenAI Chat API).
     - Validates citations:
       - Every citation number must be in `[1..num_sources]`.
       - Every sentence must end with a citation token.
     - If invalid, runs up to two **repair calls** with a specialized “rewrite to comply with citations” prompt.
     - If still invalid, returns a refusal rather than an uncited answer.

9. **Response guardrails**
   - `GuardrailsService.apply_all_rules()` validates the generated answer:
     - Presence and sufficiency of citations.
     - Grounding in terms of number and quality of retrieved chunks.
     - Answer length and minimum quality checks.
   - If guardrails fail, returns a safe and explicit refusal.

10. **Response**
    - Response includes:
      - Answer text.
      - List of `Source` objects (chunk ID, title, snippet, similarity, metadata).
      - Safety report.
      - Latency and retrieval scores.

The same RAG primitives are reused in the **agentic** path (`/api/v1/agentic-chat`) through tools wrapped by LangChain.

---

## 7. Security and Guardrails

### Authentication & Authorization

- **JWT auth**:
  - Access and refresh tokens via `app/auth/jwt.py` and `AuthService` in `app/auth/service.py`.
  - Password hashing with `passlib`.
- **OAuth**:
  - Provider integration (Google, GitHub, Microsoft) in `app/auth/oauth.py` and `app/api/routes/auth.py`.
  - OAuth login endpoints exposed under `/api/v1/auth/oauth/...`.
- **Role-based access (RBAC)**:
  - Roles in `UserRole` enum (`app/auth/models.py`).
  - Dependencies like `require_admin`, `require_solicitor_or_admin`, and `get_current_active_user` in `app/auth/dependencies.py`.

### Guardrails & Safety

- **Query-level guardrails** (`GuardrailsService.validate_query()`):
  - Domain gating: ensure the question is legal in nature.
  - Harmful content detection via regex patterns.
  - Low legal relevance detection.

- **Response-level guardrails** (`GuardrailsService.validate_response()` / `apply_all_rules()`):
  - Citation presence and structure.
  - Grounding in terms of number and quality of retrieved chunks.
  - Answer length and minimum quality checks.

- **Red-team testing**:
  - `retrieval/red_team_tester.py` plus scripts in `scripts/test_red_team.py`.
  - Adversarial prompts for prompt injection, off-topic, harmful, and PII-related queries.

> Presidio-based PII detection and anonymization libraries are present as dependencies, but there is no direct runtime integration yet; they are available for future hardening.

---

## 8. Evaluation & Monitoring

### Evaluation

- **Unit, integration, and E2E tests** under `tests/` and `scripts/test_*.py`:
  - Chat and retrieval endpoints.
  - Auth and route protection.
  - Document upload and retrieval.
  - Monitoring and health endpoints.
- **Red-team tests** (`retrieval/red_team_tester.py`):
  - Validate guardrail coverage for dangerous or off-domain prompts.
- **RAGAS / TruLens**:
  - Included in `requirements.txt` for offline experiments and RAG evaluation, but not wired into production code yet.

### Monitoring & Observability

- **Request/response logging**:
  - `RequestResponseLoggingMiddleware` in `app/core/middleware.py` logs request metadata, payload previews, response status, and latency.
  - `ErrorTrackingMiddleware` logs unhandled exceptions with context.
- **Metrics**:
  - `app/core/metrics.py` implements a `metrics_collector` for API latency, status codes, and tool usage; Prometheus client library is used.
- **Health checks**:
  - Implemented in `app/core/health_checker.py` with:
    - Database connectivity (Postgres).
    - Optional Redis/Qdrant/OpenAI checks.
  - Exposed via routes in `app/api/routes/health.py` (e.g., `/api/v1/health`, `/api/v1/health/live`, `/api/v1/health/ready`).

---

## 9. API Endpoints (High-Level)

The FastAPI app is defined in `app/api/main.py`, which includes and prefixes routers under `/api/v1`.

Key endpoint groups:

- **Chat & RAG**
  - `POST /api/v1/chat` — main chat endpoint with RAG and guardrails.
  - `POST /api/v1/agentic-chat` — agentic RAG path using LangChain tools.
  - `POST /api/v1/search` — hybrid search API returning raw retrieval results.

- **Documents**
  - `POST /api/v1/documents` — upload/document creation.
  - `GET /api/v1/documents` — list documents.
  - `GET /api/v1/documents/{id}` — get document details.
  - (Additional endpoints for updating/deleting documents exist in `app/api/routes/documents.py`.)

- **Authentication**
  - `POST /api/v1/auth/register` — user registration.
  - `POST /api/v1/auth/login` — login with email/password.
  - `POST /api/v1/auth/refresh` — refresh access token.
  - `POST /api/v1/auth/logout`, `/api/v1/auth/logout-all` — logout from sessions.
  - `GET /api/v1/auth/me`, `PUT /api/v1/auth/me` — profile management.
  - `POST /api/v1/auth/change-password` — password change.
  - `GET /api/v1/auth/oauth/{provider}/authorize` — OAuth redirect URL.

- **Health & Metrics**
  - `GET /api/v1/health` — simple health.
  - `GET /api/v1/health/live` — liveness.
  - `GET /api/v1/health/ready` — readiness.
  - `GET /api/v1/metrics` — metrics endpoint (for Prometheus scraping).

- **Debug & Monitoring**
  - `GET /api/v1/debug/...` — diagnostic endpoints (see `app/api/routes/debug.py`).

For exact request/response models, see `app/models/schemas.py` and individual route modules.

---

## 10. Running the Project Locally

### Prerequisites

- Python 3.11
- PostgreSQL (if you want full auth and document features; DEMO mode can run without DB)
- (Optional) Redis, Qdrant for full health check coverage

### Setup

```bash
git clone <this-repo-url>
cd Legal-Chatbot

python -m venv .venv
source .venv/bin/activate  # or .venv\\Scripts\\activate on Windows

pip install --upgrade pip
pip install -r requirements.txt
```

Set environment variables (at minimum):

```bash
export OPENAI_API_KEY="your-openai-key"
# For full DB-backed auth and documents:
export DATABASE_URL="postgresql://user:password@localhost:5432/legal_chatbot"
export JWT_SECRET_KEY="your-jwt-secret"
export SECRET_KEY="your-secret-key"
```

If you want a **pure demo** mode without DB, ensure `DEMO_MODE=true` in environment and configure other settings in `.env` or env vars as needed (see `app/core/config.py`).

### Initialize Database & Index (optional but recommended)

```bash
# Run migrations
python -m alembic upgrade head

# Build or update FAISS index if you have source documents locally
python scripts/ingest_data.py
```

### Run Backend

```bash
uvicorn app.api.main:app --host 0.0.0.0 --port 8000 --reload
```

API docs are available at: `http://localhost:8000/docs`

### Run Frontend

```bash
streamlit run frontend/app.py --server.port 8501
```

The Streamlit UI expects `BACKEND_URL` to point at the FastAPI base URL (e.g., `http://localhost:8000`); if not set, it defaults to localhost.

---

## 11. Docker Deployment

The `Dockerfile` defines a production-oriented container:

```dockerfile
FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /code

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONPATH=/code

EXPOSE 8080

CMD ["python", "-m", "uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8080"]
```

### Build & Run

```bash
docker build -t legal-chatbot-api .

docker run -p 8080:8080 \
  -e OPENAI_API_KEY="your-openai-key" \
  -e DATABASE_URL="postgresql://user:password@host:5432/legal_chatbot" \
  -e JWT_SECRET_KEY="your-jwt-secret" \
  -e SECRET_KEY="your-secret-key" \
  legal-chatbot-api
```

Backend will be available at: `http://localhost:8080`.

### Cloud Run Deployment

The script `scripts/deploy.sh` automates build and deploy via:

- Google Cloud Build (builds Docker image and pushes to Artifact Registry).
- Google Cloud Run (deploys as a managed service).

You can customize project, region, and service name via environment variables (`GCP_PROJECT_ID`, `GCP_REGION`, `CLOUD_RUN_SERVICE`, etc.).

---

## 12. Future Improvements

Based on the current implementation, natural next steps include:

- **Deeper evaluation integration**:
  - Wire RAGAS and TruLens into evaluation pipelines and/or online monitoring.
  - Add labeled sets for retrieval metrics (recall@k, MRR).
- **Multi-tenant support**:
  - Tenant-aware document and user isolation (schema or row-level).
- **More advanced PII handling**:
  - Integrate Presidio analyzers/anonymizers in preprocessing and logging.
- **Scalability**:
  - Move vector search to a distributed store (e.g. Qdrant) for very large corpora.
  - Introduce caching and query/result deduplication.
- **Richer frontend**:
  - More advanced document management and analytics dashboards.

---

## 13. Demo

With the backend and Streamlit app running:

- Open `http://localhost:8501` for the UI.
- Ask a legal question about UK law (e.g., employment rights, sale of goods).
- Inspect:
  - Retrieved sources and their IDs.
  - Citation tokens `[1]`, `[2]` at the end of each sentence in the answer.
  - Safety messages when guardrails reject off-domain or harmful questions.

This end-to-end flow exercises the ingestion, retrieval, LLM, guardrails, and monitoring components implemented in this repository.

