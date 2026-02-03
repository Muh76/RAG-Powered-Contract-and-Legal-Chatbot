# Legal Chatbot — Production RAG for UK Law

A **production-ready** legal assistant that answers UK legal questions using Retrieval-Augmented Generation (RAG). Every answer is grounded in retrieved sources and cited with `[1]`, `[2]` tokens. Built with FastAPI, hybrid retrieval (BM25 + FAISS), LangChain agents, and enterprise auth. **131,253+ document chunks** from CUAD contracts and UK legislation. Phases 1–5.4 complete.

---

## Why This Project Matters

**Problem:** Legal information is hard to access, and generic LLMs hallucinate. Users need answers they can trust and verify.

**Solution:** RAG over a curated legal corpus. Hybrid retrieval finds relevant chunks; the LLM answers only from those chunks. Citations are enforced at generation time and validated before response. Agentic mode handles multi-step queries (e.g. “Compare X with Y”).

**Impact:** Reliable, cited answers with audit trails. Suitable for law firms (explainability, safety testing) and the public (clear disclaimers, highlighted sources).

---

## Key Differentiators

- **Citation enforcement:** Every sentence must end with `[n]` from sources. Retry loop repairs non-compliant answers; deterministic validation rejects before return.
- **Hybrid RAG:** BM25 + semantic (FAISS) with RRF fusion and optional cross-encoder reranking.
- **Agentic AI:** LangChain agents with autonomous tool selection and multi-step reasoning.
- **Enterprise auth:** JWT + OAuth2 (Google, GitHub, Microsoft) with RBAC (Public, Solicitor, Admin).
- **Private corpus:** User-specific document upload (PDF/DOCX/TXT) and combined public+private retrieval.
- **108+ E2E tests:** Integration, regression, performance benchmarks, red-team safety tests.
- **Production monitoring:** Structured logging, health checks, metrics, dependency monitoring.

---

## System Overview

```
┌─────────────┐     ┌──────────────┐     ┌─────────────────┐
│  Streamlit  │────▶│   FastAPI    │────▶│  RAG Pipeline   │
│  Frontend  │     │  (Auth/RBAC) │     │ BM25 + FAISS    │
└─────────────┘     └──────────────┘     └────────┬────────┘
       │                     │                     │
       │                     │                     ▼
       │                     │            ┌─────────────────┐
       │                     │            │  LLM + Guardrails│
       │                     │            │  Citation Check  │
       │                     │            └─────────────────┘
       │                     │
       ▼                     ▼
┌─────────────┐     ┌──────────────┐
│  PostgreSQL │     │  Document    │
│  + pgvector │     │  Storage     │
└─────────────┘     └──────────────┘
```

**Core flow:** Query → retrieval (hybrid) → LLM generation → citation validation (retry if needed) → response.

---

## Completed vs Roadmap

| Status | Area |
|--------|------|
| ✅ | Phase 1: MVP RAG, guardrails, dual-mode (Solicitor/Public) |
| ✅ | Phase 2: Hybrid retrieval, reranking, explainability |
| ✅ | Phase 3: Agentic RAG, LangChain tools, multi-step reasoning |
| ✅ | Phase 4.1: 108+ E2E tests, integration, regression, performance |
| ✅ | Phase 4.2: Structured logging, health checks, metrics |
| ✅ | Phase 5.1–5.4: PostgreSQL, JWT/OAuth2, RBAC, document upload, frontend auth |
| 🔜 | Multi-tenant, enhanced GDPR, multilingual, production deployment (optional) |

---

## Quick Start

```bash
git clone https://github.com/Muh76/RAG-Powered-Contract-and-Legal-Chatbot.git
cd RAG-Powered-Contract-and-Legal-Chatbot

pip install -r requirements.txt
export OPENAI_API_KEY="your-api-key"
export DATABASE_URL="postgresql://user:password@localhost:5432/legal_chatbot"
export JWT_SECRET_KEY="your-jwt-secret"
export SECRET_KEY="your-secret-key"

python -m alembic upgrade head
python scripts/ingest_data.py   # if FAISS index not present

uvicorn app.api.main:app --reload --port 8000 &
streamlit run frontend/app.py --server.port 8501
```

**Access:** UI `http://localhost:8501` (auth required) · API docs `http://localhost:8000/docs` · Health `http://localhost:8000/api/v1/health`

---

## Testing, Monitoring & Safety

- **108+ E2E tests** across chat, search, agentic, auth, and document upload.
- **Red-team tests:** 50 adversarial cases for guardrails (prompt injection, domain gating, harmful content).
- **Health endpoints:** `/api/v1/health`, `/api/v1/health/detailed`, `/api/v1/health/live`, `/api/v1/health/ready`.
- **Metrics:** Response times, error rates, request volumes, tool usage, system (CPU/memory/disk).
- **Safety:** PII redaction, domain gating, harmful content detection, citation enforcement, audit logging.

---

## Tech Stack

| Layer | Technologies |
|-------|--------------|
| Backend | FastAPI, PostgreSQL, pgvector |
| Frontend | Streamlit |
| ML/AI | OpenAI GPT-4, SentenceTransformers, PyTorch |
| Retrieval | FAISS, BM25, cross-encoder reranking |
| Agentic | LangChain, LangGraph |
| Infra | Docker, Alembic |
| Observability | Structured JSON logging, health checks, metrics |

---

## Disclaimer & Contact

**This chatbot provides educational information only and does not constitute legal advice.** Always consult qualified legal professionals for specific matters.

- **Email:** mj.babaie@gmail.com  
- **LinkedIn:** [Mohammad Babaie](https://www.linkedin.com/in/mohammadbabaie/)  
- **GitHub:** [Muh76](https://github.com/Muh76)

---

## Deep Dive (Optional)

<details>
<summary><b>Architecture & Project Layout</b></summary>

```
/app           # FastAPI services, routes, LLM, RAG
/frontend      # Streamlit UI
/ingestion     # Document loaders and parsers
/retrieval     # Embeddings, FAISS, hybrid retriever
/guardrails    # Safety policies and validators
/eval          # RAG evaluation framework
/infra         # Docker and deployment configs
/tests         # E2E, integration, unit tests
/docs          # Architecture and security docs
```

</details>

<details>
<summary><b>Knowledge Base & Data</b></summary>

- **131,253+ chunks** ingested and indexed
- **CUAD:** 131,000+ contract chunks (Contract Understanding Atticus Dataset)
- **UK Legislation:** 460+ chunks (Employment Rights Act, Equality Act, Sale of Goods Act)
- **FAISS** vector index + **BM25** keyword index
- **Gold evaluation set:** 150 Q&A pairs
- **Red-team test set:** 50 adversarial cases

</details>

<details>
<summary><b>API Endpoints</b></summary>

| Endpoint | Auth | Description |
|----------|------|-------------|
| `POST /api/v1/chat` | Required | RAG chat with citations |
| `POST /api/v1/agentic-chat` | Required | Agentic chat (tools, multi-step) |
| `POST /api/v1/search/hybrid` | Required | Hybrid search with explainability |
| `POST /api/v1/documents/upload` | Solicitor/Admin | Upload PDF/DOCX/TXT |
| `GET /api/v1/documents` | Solicitor/Admin | List documents |
| `GET /api/v1/health` | Public | Health check |
| `GET /api/v1/metrics/*` | Admin | Metrics and system stats |

</details>

<details>
<summary><b>Citation Enforcement Pipeline</b></summary>

1. **Prompt rules:** System and user prompts require every sentence to end with `[n]`.
2. **Retry loop:** If `_validate_citations()` or `_every_sentence_ends_with_citation()` fails, up to 2 repair calls with a stricter “rewrite with citations” prompt.
3. **Deterministic validation:** Regex checks each sentence ends with `(?:\[\d+\])+[.!?]*`.
4. **Fallback:** If all retries fail, return: *"I cannot answer this question because I cannot provide properly cited legal sources."*

</details>

<details>
<summary><b>Agentic RAG — Tools & Flow</b></summary>

**Tools:** `search_legal_documents`, `get_specific_statute`, `analyze_document`

**Flow:** User query → Agent selects tools → Tool execution → Observation → Reasoning (repeat if needed) → Final answer with citations.

**Example:** "Compare Sale of Goods Act 1979 with Consumer Rights Act 2015" → Agent calls `get_specific_statute` for each → Compares and cites.

</details>

<details>
<summary><b>Document Upload & Private Corpus</b></summary>

- **Upload:** PDF, DOCX, TXT via `POST /api/v1/documents/upload` (Solicitor/Admin)
- **Processing:** Chunking, embedding, indexing into user-scoped corpus
- **Retrieval:** Combined public + private search using RRF fusion
- **Chat:** Private corpus included automatically for authenticated users

</details>

<details>
<summary><b>Testing Commands</b></summary>

```bash
pytest tests/e2e/ -v
pytest tests/integration/ -v
python scripts/test_route_protection.py
python scripts/test_document_upload.py
python scripts/test_agentic_chat.py
python scripts/verify_monitoring.py --url http://localhost:8000
```

</details>

<details>
<summary><b>Documentation Links</b></summary>

- [Architecture Overview](docs/architecture/README.md)
- [Security Guidelines](docs/security/README.md)
- [Evaluation Metrics](docs/eval/README.md)
- [Phase 2 Hybrid Retrieval](docs/phase2_hybrid_retrieval.md)
- [Phase 3 Agentic RAG](docs/phase3_agentic_rag_summary.md)
- [Phase 5 Document Upload](docs/phase5_3_document_upload_complete.md)

</details>

---

*MIT License — see [LICENSE](LICENSE) for details.*
