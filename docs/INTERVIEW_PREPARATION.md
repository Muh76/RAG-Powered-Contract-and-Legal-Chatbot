# Legal Chatbot — Interview Preparation Material

Structured material for ML Engineer / Data Scientist interviews. **All content is grounded in this repository**; nothing is invented.

---

# STEP 1 — PROJECT SUMMARY

## 1. Project name
**Legal Chatbot — Production RAG for UK Law**

## 2. Problem it solves
- Legal information is hard to access; generic LLMs hallucinate and cannot be trusted.
- Users need **cited, verifiable** answers grounded in real sources (UK law, contracts).

## 3. Who the users are
- **General public** (public mode): clear language, disclaimers.
- **Solicitors / admins** (solicitor mode): precise legal terminology, same citation rules.
- **Law firms**: explainability, safety testing, audit trails.

## 4. Why it matters
- Reliability and trust: every answer is tied to retrieved chunks and citation tokens `[1]`, `[2]`.
- Reduces hallucination via RAG + strict citation enforcement and retry.
- Suitable for professional and public use with role-based access and guardrails.

## 5. Key innovation
- **Citation enforcement**: every sentence must end with `[n]` from sources; retry loop repairs non-compliant answers; validation rejects before return.
- **Hybrid RAG**: BM25 + semantic (OpenAI embeddings + FAISS) with RRF fusion and optional cross-encoder reranking.
- **Agentic mode**: LangChain/LangGraph agents with tools (legal search, statute lookup, document analyzer) for multi-step queries.
- **Enterprise auth**: JWT + OAuth2 (Google, GitHub, Microsoft), RBAC (Public, Solicitor, Admin), private document upload and retrieval.

---

## Three explanations

### 1️⃣ Elevator pitch (30 seconds)
“I built a production legal assistant for UK law that uses RAG so every answer is grounded in retrieved sources and cited with [1], [2]. It uses hybrid retrieval—BM25 plus semantic search with FAISS—and enforces citations in the LLM with a retry loop. There’s an agentic mode with LangChain for multi-step questions, plus JWT and OAuth, role-based access, and guardrails for domain and safety. We have 131k+ chunks from contracts and UK legislation, and 108+ E2E tests including red-team safety tests.”

### 2️⃣ Non-technical (HR / product)
“The product is a legal Q&A assistant focused on UK law. Instead of the model inventing answers, it only uses text retrieved from a curated legal corpus. Every claim is tied to a source and shown as [1], [2] so users can verify. We block off-topic or harmful queries and support different user roles—public vs solicitors—and optional login with Google or GitHub. We can also run more complex, multi-step questions through an agent that can search and look up statutes in sequence.”

### 3️⃣ Technical (ML engineers)
“RAG pipeline: user query → guardrails (domain/harm) → hybrid retrieval (BM25 + OpenAI embeddings + FAISS, RRF fusion, optional cross-encoder rerank) → context assembly from top-k chunks → prompt with strict citation rules (solicitor/public mode) → OpenAI Chat Completions. Citation validation runs on the raw answer; if sentences lack `[n]` we do up to two repair calls with a rewrite prompt; if still invalid we return a refusal. Private corpus is supported via document upload and merged with public results. Agentic path uses LangChain/LangGraph with tools that call the same RAG and statute lookup. Backend is FastAPI; Postgres + pgvector for auth and documents; Streamlit frontend; deployment target is Docker/Cloud Run.”

---

# STEP 2 — SYSTEM ARCHITECTURE

## High-level flow

```
User Query
    ↓
API Endpoint (FastAPI) — app/api/routes/chat.py
    ↓
Guardrails (query validation) — app/services/guardrails_service.py
    ↓
Query embedding (if hybrid) — retrieval/embeddings/openai_embedding_generator.py
    ↓
Vector + lexical search — app/services/rag_service.py → retrieval/hybrid_retriever.py, bm25_retriever.py, semantic/openai
    ↓
Top-K retrieval (RRF fusion, optional rerank) — retrieval/hybrid_retriever.py, rerankers/cross_encoder_reranker.py
    ↓
Context assembly — app/api/routes/chat.py + app/services/llm_service.py (format [1] Title\nchunk...)
    ↓
Prompt construction — app/services/llm_service.py (system + user prompt, mode=solicitor|public)
    ↓
LLM generation — app/services/llm_service.py (OpenAI chat.completions.create)
    ↓
Citation validation + retry — app/services/llm_service.py (_validate_citations, _every_sentence_ends_with_citation, up to 2 repair calls)
    ↓
Guardrails (response) — app/services/guardrails_service.py (citations, grounding, quality)
    ↓
Response returned — ChatResponse with answer, sources, safety, metrics
```

## Data ingestion
- **Script**: `scripts/ingest_data.py`
- **Loaders**: `ingestion/loaders/document_loaders.py` (PDF, text, JSON, CUAD parquet, UK legislation).
- **Chunking**: `ingestion/chunkers/document_chunker.py` — `ChunkingConfig` (e.g. 1000 chars, 200 overlap, sentence-boundary aware), `DocumentChunker.chunk_document`, `chunk_by_sections`.
- **Embeddings**: OpenAI via `retrieval/embeddings/openai_embedding_generator.py` (text-embedding-3-large, 3072d).
- **Index**: FAISS index + chunk metadata saved to `data/faiss_index.bin`, `data/chunk_metadata.pkl` (and combined pkl in `data/indices/`).

## Document preprocessing
- Loaders produce `DocumentChunk` (chunk_id, text, metadata: title, source, jurisdiction, document_type, section, etc.).
- Chunker splits by size/overlap and optionally by legal section patterns.

## Chunking strategy
- Target chunk size 1000 chars, overlap 200, min 100, max 1500.
- Sentence-boundary aware (break near `. `, `! `, `? `).
- Alternative: chunk by sections (e.g. “Section \d+”).

## Embedding generation
- **Primary**: OpenAI `text-embedding-3-large` (3072d) in `retrieval/embeddings/openai_embedding_generator.py`.
- Config: `app/core/config.py` (OPENAI_EMBEDDING_MODEL, EMBEDDING_DIMENSION=3072).
- Fallback when OpenAI unavailable: TF-IDF only (no vector search).

## Vector database usage
- **FAISS** (in-memory): index in `data/faiss_index.bin`, metadata in `data/chunk_metadata.pkl`.
- Loaded in `app/services/rag_service.py` (`_load_vector_store`).
- **pgvector** and **Qdrant** appear in config/deps but primary retrieval path is FAISS + BM25 in-process.

## Retrieval strategy
- **Hybrid**: BM25 (rank_bm25) + semantic (OpenAI embeddings + FAISS).
- **Fusion**: RRF (default) or weighted; params in config (HYBRID_SEARCH_*).
- **Optional**: cross-encoder rerank (`retrieval/rerankers/cross_encoder_reranker.py`, e.g. ms-marco-MiniLM).
- **Metadata filtering**: `retrieval/metadata_filter.py` (pre/post filter).
- **Explainability**: `retrieval/explainability.py` (ExplainabilityAnalyzer for highlighting, explanations).

## RAG pipeline
- **Orchestration**: `app/services/rag_service.py` — `search()` delegates to hybrid or TF-IDF-only.
- **Hybrid**: `retrieval/hybrid_retriever.py` (AdvancedHybridRetriever) — BM25 + semantic retrievers, fusion, optional rerank.
- **BM25**: `retrieval/bm25_retriever.py`.
- **Semantic**: OpenAI + FAISS (e.g. `retrieval/openai_semantic_retriever.py` / semantic path in RAG service).

## Prompt construction
- **File**: `app/services/llm_service.py` — `generate_legal_answer()`.
- **System prompt**: solicitor vs public mode; strict citation rules, anti-hallucination, “answer only from sources”, disclaimer.
- **User prompt**: “SOURCES (numbered [1], [2], …)” + context (formatted chunks) + “QUESTION: {query}” + format and citation rules.

## LLM inference
- **File**: `app/services/llm_service.py` — `OpenAI(api_key).chat.completions.create(model=OPENAI_MODEL, messages=..., max_tokens=2000, temperature=0.1)`.
- Model: `OPENAI_MODEL` (e.g. gpt-4-turbo-preview) from `app/core/config.py`.

## API / backend layer
- **Entry**: `app/api/main.py` (FastAPI app, lifespan, CORS, middleware).
- **Routes**: `app/api/routes/` — health, auth, chat, documents, search, agentic_chat, metrics, debug.
- **Auth**: JWT + OAuth2 in `app/auth/` (dependencies, service, jwt, oauth); RBAC in dependencies.

## Frontend
- **Streamlit**: `frontend/streamlit/app.py` and `frontend/app.py` — chat UI, backend URL via `BACKEND_URL` (default localhost:8000).
- **Auth UI**: `frontend/auth_ui.py`; protected route component in `frontend/components/`.

## Deployment
- **Docker**: `Dockerfile` — Python 3.11, WORKDIR /code, PYTHONPATH=/code, uvicorn `app.api.main:app` on 8080.
- **Cloud Run**: `scripts/deploy.sh` — gcloud build submit, push to Artifact Registry, deploy to Cloud Run (DEMO_MODE, env vars).

---

# STEP 3 — TECHNOLOGY STACK

| Category | Technologies | Why |
|----------|--------------|-----|
| **Programming** | Python 3.11 | Runtime for backend, ingestion, retrieval, agents. |
| **Frameworks** | FastAPI, Pydantic | Async API, validation, OpenAPI docs. |
| **ML/LLM** | OpenAI (GPT-4, text-embedding-3-large), LangChain, LangGraph | Chat and embeddings; agent orchestration and tools. |
| **Vector / retrieval** | FAISS (cpu), rank_bm25 | In-memory vector search; lexical search; no server for FAISS. |
| **Backend** | PostgreSQL, SQLAlchemy, Alembic, pgvector | Auth, user data, migrations; vector option. |
| **Deployment** | Docker, Google Cloud Run, gcloud | Containerized app; serverless deploy. |
| **MLOps / eval** | ragas, trulens-eval (in requirements) | RAG evaluation and monitoring (deps present; red-team and tests are in repo). |
| **Monitoring** | Loguru, Prometheus client, OpenTelemetry, health_checker | Logging, metrics, health checks. |
| **Security** | python-jose, passlib, Presidio | JWT/OAuth, password hashing, PII detection/redaction. |
| **Frontend** | Streamlit | Quick chat UI and auth flow. |

---

# STEP 4 — DESIGN DECISIONS

- **Embedding model**: text-embedding-3-large (3072d) — strong quality, matches single embedding dimension across ingestion and retrieval; avoids local PyTorch in production path (OpenAI API).
- **Chunking**: 1000 chars, 200 overlap, sentence-boundary aware — balances context length and granularity; overlap reduces boundary cuts.
- **Hybrid retrieval**: BM25 + semantic — covers exact/lexical and semantic match; RRF fusion keeps both contributions.
- **Citation enforcement**: Prompt rules + post-hoc validation + up to 2 repair calls — reduces uncited or wrong-number citations; refusal if still invalid.
- **Guardrails**: Query (domain, harmful patterns) and response (citations, grounding, length) — limits off-topic and unsafe use; enforces “legal only” and citation policy.
- **Agentic mode**: LangChain/LangGraph with tools (legal search, statute lookup, document analyzer) — handles multi-step questions without a single monolithic prompt.
- **Auth**: JWT + OAuth2 and RBAC — supports both programmatic and “login with Google/GitHub” for different user types.
- **Reliability**: Similarity threshold rejection (e.g. avg &lt; 0.4) before LLM call; retry loop for citations; guardrails after generation.
- **Scalability**: Stateless API; FAISS in-process (single replica); Cloud Run scales instances; DB and optional Redis for sessions/cache.

---

# STEP 5 — INTERVIEW STORY

1. **Problem**: Legal information is hard to access; generic LLMs hallucinate; users need cited, verifiable answers for UK law.
2. **Approach**: RAG over a curated corpus (131k+ chunks from CUAD and UK legislation) with strict citation rules and guardrails.
3. **Architecture**: FastAPI backend; hybrid retrieval (BM25 + OpenAI + FAISS) with RRF and optional rerank; context injected into prompts; OpenAI chat completion; citation validation and retry; guardrails on query and response; optional agentic path with LangChain tools.
4. **Challenges**: Citation compliance (solved with validation + repair loop); retrieval quality (hybrid + rerank + similarity threshold); safety and domain (guardrails + red-team tests); production stability (OpenAI embeddings to avoid PyTorch segfaults in async context).
5. **Solutions**: Deterministic citation validation and rewrite prompts; BM25 + FAISS + RRF and optional cross-encoder; GuardrailsService (domain, harmful, grounding); RedTeamTester and safety test cases; ThreadPoolExecutor for blocking retrieval/LLM; DEMO_MODE and lazy init for Cloud Run.
6. **Results**: 108+ E2E tests; red-team coverage; health and metrics endpoints; JWT/OAuth and RBAC; deployable to Cloud Run with DEMO_MODE.

---

# STEP 6 — TECHNICAL DEEP DIVE QUESTIONS (25)

1. **Why RAG instead of fine-tuning?**  
   We need answers grounded in specific, updatable sources (UK law, contracts) and citable. RAG keeps the model’s knowledge separate from the corpus so we can change documents and re-index without retraining. Fine-tuning would bake in facts and make citations harder to enforce.

2. **How does retrieval work?**  
   Hybrid: BM25 over chunk text for lexical match, OpenAI embeddings + FAISS for semantic similarity. We run both, merge with RRF (or weighted), optionally rerank with a cross-encoder, then take top-k. Private user documents can be merged with the same RRF.

3. **How do you reduce hallucinations?**  
   (1) Answer only from retrieved chunks in the prompt. (2) Strict citation rules and “do not use training data” in the prompt. (3) Post-validate every sentence ends with [n] and numbers match sources; retry with a repair prompt up to twice; else refuse. (4) Reject very low similarity retrieval before calling the LLM.

4. **How do you evaluate?**  
   E2E tests (108+), integration tests for services, red-team tests (RedTeamTester, safety_test_cases.json) for guardrails. Ragas and Trulens are in requirements for RAG/LLM evaluation; main automated quality signal in code is tests and red-team results.

5. **How would you scale?**  
   API is stateless; scale Cloud Run replicas. FAISS is in-process so each replica has its own index; for very large corpora we’d consider a separate vector service (e.g. Qdrant) or sharded FAISS. DB and optional Redis for auth/sessions.

6. **Why this embedding model?**  
   text-embedding-3-large gives 3072d and good quality; one dimension across ingestion and search; we use OpenAI API so no GPU or PyTorch in the serving path.

7. **Why FAISS and not only a vector DB?**  
   FAISS is fast in-process and avoids network latency; the corpus fits in memory. pgvector/Qdrant are in stack for flexibility; primary path is FAISS for low latency and simplicity.

8. **What is RRF?**  
   Reciprocal Rank Fusion: merge two ranked lists (BM25 and semantic) by summing 1/(k+rank) and re-ranking; k from config (e.g. 60). No need to normalize scores across BM25 and cosine.

9. **How does the citation retry loop work?**  
   After the first LLM reply we validate citations (regex and sentence-end check). If invalid, we send a second call with a “rewrite so every sentence ends with [n]” prompt, again validate; one more attempt if needed. After that we return a refusal message.

10. **What are guardrails?**  
    GuardrailsService: query validation (legal vs non-legal keywords, harmful patterns, minimum legal relevance) and response validation (has_citations, enough chunks, answer length). Failures return structured reasons and block or regenerate.

11. **How does agentic mode work?**  
    AgenticRAGService uses LangChain/LangGraph (create_react_agent or AgentExecutor) with ChatOpenAI and tools: LegalSearchTool (hybrid RAG), StatuteLookupTool, DocumentAnalyzerTool. The agent chooses tools and iterates; results are aggregated into a final answer with safety and metrics.

12. **How is private corpus merged with public?**  
    DocumentService stores user uploads (metadata + chunks); search can pre-fetch private results and pass them into RAGService.search(); _combine_results does RRF between public and private lists and returns top-k.

13. **Why two response modes (solicitor vs public)?**  
    Solicitor mode uses more precise legal language; public mode explains in clearer terms. Both share the same citation and anti-hallucination rules; the system prompt wording differs.

14. **What is the chunk format for the LLM?**  
    “[1] Act/source\nchunk_text”, “[2] …”, etc., joined by “\n\n”. Numbers in the answer must refer to these indices.

15. **How is retrieval run without blocking the event loop?**  
    Blocking retrieval (and LLM) run in a ThreadPoolExecutor (e.g. loop.run_in_executor) so the async FastAPI loop stays responsive and we avoid PyTorch/FAISS issues in the main thread.

16. **What if FAISS is missing?**  
    RAG service falls back to TF-IDF-only search over chunk_metadata (BM25 path still works). If both FAISS and embeddings are missing, TF-IDF-only; config can require OPENAI_API_KEY when FAISS exists.

17. **How is health checked?**  
    HealthChecker checks Postgres (SELECT 1), Redis, Qdrant, OpenAI (optional). Cached 30s. DEMO_MODE skips DB check. Routes: /api/v1/health, /api/v1/health/detailed, live, ready.

18. **What metrics are exposed?**  
    Prometheus-style metrics (e.g. from app/core/metrics.py): response times, error rates, request counts, tool usage, system (CPU/memory/disk). Endpoint and middleware feed into this.

19. **How is auth enforced on routes?**  
    get_current_active_user (and role-based helpers like require_admin) depend on JWT in Authorization header; in DEMO_MODE a mock user can be returned. OAuth callback issues tokens; refresh tokens supported.

20. **Why Presidio?**  
    PII detection and anonymization for privacy and compliance (e.g. in logs or stored content).

21. **What is in the system prompt?**  
    Role (legal assistant UK), disclaimer (not legal advice), strict citation rules (every sentence [n], only from sources), anti-hallucination (only use provided sources), output format, and refusal conditions.

22. **How is reranking applied?**  
    CrossEncoderReranker (e.g. ms-marco-MiniLM) takes top candidates from hybrid search, scores query–document pairs, re-sorts and takes top-k. Optional via ENABLE_RERANKING and config.

23. **What is explainability?**  
    ExplainabilityAnalyzer in retrieval/explainability.py: explains why a chunk was retrieved (e.g. matched terms, confidence), can add highlighted text and retrieval path info for UI or debugging.

24. **How is deployment done?**  
    Dockerfile builds the app; scripts/deploy.sh runs gcloud builds submit and gcloud run deploy (Artifact Registry, Cloud Run). DEMO_MODE and env vars set so the service can start without DB if needed.

25. **How do you handle rate limits (OpenAI)?**  
    OpenAIEmbeddingGenerator uses retries with backoff and batch_size/requests_per_minute limits; LLM calls have timeout and single-request usage.

---

# STEP 7 — SYSTEM DESIGN QUESTIONS (10)

1. **Scale to millions of users?**  
   Stateless API + horizontal scaling (Cloud Run). Per-request auth (JWT). Consider CDN for static frontend, rate limiting, and DB connection pooling. For very high QPS, consider caching frequent query–response pairs or embedding cache.

2. **Improve retrieval quality?**  
   Larger top-k before rerank; tune RRF k and weights; add more metadata filters; try other embedders or hybrid with a second vector index; A/B test chunk sizes; collect feedback and tune thresholds.

3. **Monitor hallucinations?**  
   Log prompts and responses; sample and compare to sources (citation coverage, n-gram overlap); use NLI or QA model to score faithfulness; alert on low scores or missing citations; optional human review queue.

4. **Add user feedback loops?**  
   Store thumbs up/down or ratings with (query, response_id, user_id); use for reranking or retrieval tuning; periodic fine-tune of a reranker or train a reward model for citation quality.

5. **Multi-tenant?**  
   Tenant_id on users and documents; filter all retrievals and DB queries by tenant; separate or tagged indices per tenant if needed; auth and RBAC per tenant.

6. **Lower latency?**  
   Embedding cache for repeated queries; smaller top-k or faster reranker; model choice (smaller/faster LLM); async and connection pooling; regional deployment.

7. **Handle very long documents?**  
   Hierarchical chunking or summaries; map chunk to doc/section in metadata; retrieve at chunk level, optionally expand to section for context; limit total context length in the prompt.

8. **Improve citation accuracy?**  
   Stricter prompt (examples of good/bad); more repair attempts or a dedicated citation-repair step; NER to align entities to sources; human-in-the-loop on disputed answers.

9. **Disaster recovery?**  
   DB backups and point-in-time recovery; FAISS and metadata in object storage (e.g. GCS) and loaded at startup; multi-region or failover for API; documented restore and rollback.

10. **Cost control?**  
    Cache frequent queries; rate limits and quotas per user/tenant; smaller or cheaper models where acceptable; batch embedding for ingestion; monitor token and API usage per customer.

---

# STEP 8 — CODE WALKTHROUGH (KEY FILES)

| File | Purpose | Key functions/classes |
|------|---------|----------------------|
| `app/api/main.py` | FastAPI app, lifespan, CORS, routes | lifespan (startup/shutdown), app, root/health |
| `app/api/routes/chat.py` | Chat endpoint | chat(), get_rag_service(), get_guardrails_service(), retrieval + LLM + guardrails flow |
| `app/services/rag_service.py` | RAG orchestration | RAGService, _initialize(), _load_vector_store(), search(), _hybrid_search(), _semantic_search() |
| `app/services/llm_service.py` | LLM and citations | LLMService, generate_legal_answer(), _validate_citations(), _every_sentence_ends_with_citation() |
| `app/services/guardrails_service.py` | Safety and domain | GuardrailsService, validate_query(), validate_response(), apply_all_rules() |
| `app/services/agent_service.py` | Agentic RAG | AgenticRAGService, _initialize_tools(), _initialize_agent(), run_agent() |
| `retrieval/hybrid_retriever.py` | Hybrid search | AdvancedHybridRetriever, retrieve(), RRF/weighted fusion |
| `retrieval/bm25_retriever.py` | BM25 search | BM25Retriever |
| `retrieval/embeddings/openai_embedding_generator.py` | Embeddings | OpenAIEmbeddingGenerator, generate_embedding(), batch |
| `ingestion/chunkers/document_chunker.py` | Chunking | DocumentChunker, ChunkingConfig, chunk_document(), chunk_by_sections() |
| `ingestion/loaders/document_loaders.py` | Load documents | DocumentLoaderFactory, PDFLoader, TextLoader, JSONLoader, DocumentChunk |
| `scripts/ingest_data.py` | Build index | load_existing_index(), save_index_and_metadata(), chunk and embed pipeline |
| `app/core/config.py` | Configuration | Settings (pydantic-settings), embedding/retrieval/auth/DB env |
| `app/core/health_checker.py` | Health | HealthChecker, check_database(), check_redis(), etc. |
| `app/auth/dependencies.py` | Auth | get_current_active_user(), require_admin(), DEMO_MODE mock user |
| `app/models/schemas.py` | API models | ChatRequest, ChatResponse, Source, SafetyReport, HealthResponse, etc. |
| `Dockerfile` | Container | Python 3.11, WORKDIR /code, PYTHONPATH, uvicorn 8080 |

---

# STEP 9 — RESUME TALK TRACK (90 seconds)

“I built a production RAG-based legal assistant for UK law that serves both the public and legal professionals. The main challenge was keeping answers trustworthy and citable: we use hybrid retrieval—BM25 and semantic search with FAISS and OpenAI embeddings—and inject only retrieved chunks into the prompt. We enforce citations by validating that every sentence ends with a source number and running a retry loop with a repair prompt if not; if it still fails we refuse to answer. Guardrails restrict queries to the legal domain and block harmful content, and we run red-team tests to validate safety. There’s an agentic mode using LangChain with tools for legal search and statute lookup so we can handle multi-step questions. The backend is FastAPI with JWT and OAuth, role-based access, and optional private document upload that gets merged into retrieval. We have over 131k chunks from contracts and UK legislation, and 108+ E2E tests including performance and safety. The system is containerized and set up to deploy to Google Cloud Run, with health checks and metrics for observability.”

---

# STEP 10 — FOLLOW-UP QUESTIONS

- **Why OpenAI and not open-source LLMs?**  
  We needed strong instruction-following for citation rules and reliability; OpenAI gave the best trade-off for this project. Open-source could be swapped behind the same interface for cost or privacy.

- **How do you tune top-k?**  
  Config (TOP_K_RETRIEVAL, hybrid top-k). We retrieve more before rerank/filter then take final top-k; can A/B test or use validation set.

- **How do you add a new document source?**  
  Add or extend a loader in ingestion/loaders, run ingest script to chunk and embed, then rebuild or incrementally update FAISS and metadata.

- **What if the user asks in another language?**  
  Current design is UK English. Could add translation before/after or multilingual embeddings and a language detector.

- **How do you version the corpus?**  
  Re-run ingestion to produce new FAISS/metadata; could version artifacts (e.g. by date) and load a specific index per env or tenant.

- **Why Streamlit and not React?**  
  Fast iteration and good fit for internal/demo; React would be for a more polished or multi-page product UI.

- **How do you test the LLM output?**  
  E2E tests with mocked or fixed retrieval; red-team tests for guardrails; manual review; optional Ragas/Trulens on samples.

- **What’s next?**  
  Multi-tenant, better eval (e.g. faithfulness metrics), multilingual support, and production hardening (rate limits, caching, monitoring).

---

*End of interview preparation material. Use this with the actual codebase and adjust numbers (e.g. chunk counts, test counts) if the repo changes.*
