# Refactor Verification: RAG/Guardrails Startup-Only Init

## 1. RAGService initialization occurs only at app startup

**Evidence:**

- **`app/api/main.py` (lines 19–27):** Lifespan calls `chat_routes.init_chat_services()` once at startup. No other code path calls `init_chat_services()`.
- **`app/api/routes/chat.py`:**
  - **`init_chat_services()` (lines 45–74):** Only place that assigns `rag_service = RAGService()` or `rag_service = _create_degraded_rag_service()`. Runs only when invoked from lifespan.
  - **`get_rag_service()` (lines 77–89):** Pure getter. Returns `rag_service` or raises `HTTPException(503)` if `rag_service is None`. Does **not** call `RAGService()` or any constructor.
  - **Chat handler (line 274):** Calls `rag = get_rag_service()` only. No `RAGService()` call in the handler or its dependencies.

**Conclusion:** RAGService is constructed only inside `init_chat_services()`, which runs once during application startup. The `/api/v1/chat` path only calls `get_rag_service()`, which returns the existing singleton.

---

## 2. No heavy loading happens during /api/v1/chat (for RAG/Guardrails)

**Evidence:**

- **Guardrails:** `get_guardrails_service()` returns the module-level `guardrails_service` set at startup. `guardrails.validate_query(request.query)` is in-memory (keyword/regex checks). No FAISS, no disk, no network.
- **RAG:** `get_rag_service()` returns the module-level `rag_service` set at startup. No FAISS load, no embedding init, no `RAGService()` call on the request path.
- **Heavy work in chat handler:** The only blocking work in the handler is (1) retrieval via `run_in_executor(search_func)` (after `retrieval_start`) and (2) LLM via `run_in_executor(llm.generate_legal_answer)` (after retrieval). Both are offloaded to a thread pool; no RAG/Guardrails initialization on the request path.

**Note:** `DocumentService()` is still instantiated per request when `include_private_corpus` is true and can perform work (e.g. embedding init). That is outside the RAG/Guardrails refactor.

**Conclusion:** For RAG and Guardrails, no heavy loading (FAISS, embeddings, or service construction) occurs during `/api/v1/chat`. Getters are O(1); guardrails validation is lightweight.

---

## 3. The request lifecycle cannot block before retrieval_start (for RAG/Guardrails)

**Request path before `retrieval_start` (chat handler):**

| Step | Code | Blocking? |
|------|------|-----------|
| Auth | `get_current_active_user` (dependency) | Sync DB + JWT (brief). |
| Entry | `request_id`, `start_time`, log | No. |
| Guardrails | `get_guardrails_service()` | No – returns singleton. |
| | `guardrails.validate_query(request.query)` | Yes – sync, in-memory only (fast). |
| Early return? | If `not query_validation["valid"]` | Returns immediately; no retrieval. |
| RAG | `get_rag_service()` | No – returns singleton. |
| Optional | `DocumentService()`, `search_user_documents(...)` | Yes – DB (and possibly DocumentService init). |
| | `retrieval_start = time.time()`, log `retrieval_start` | No. |
| Retrieval | `run_in_executor(search_func)` | Offloaded to thread pool. |

**Conclusion:** Before `retrieval_start`, the handler does **not** run RAG or Guardrails initialization. It only calls pure getters and lightweight guardrails validation. So the request lifecycle does **not** block on RAG/Guardrails heavy loading before `retrieval_start`. Remaining sync work (auth, optional DocumentService/DB) is unchanged and not part of this refactor.

---

## What changed and why it fixes RemoteDisconnected

**Before refactor:**

- RAG and Guardrails were **lazy-initialized** on first request. The first (or first after failure) call to `get_rag_service()` or `get_guardrails_service()` inside `/api/v1/chat` could run:
  - **RAG:** `RAGService()` → FAISS load from disk, optional embedding init (OpenAI or PyTorch), hybrid retriever setup. This could take many seconds and block the event loop (sync init in async handler).
  - **Guardrails:** `GuardrailsService()` (lightweight, but still on request path).
- If that initialization was slow or the client had a timeout (e.g. 120s), the client could close the connection before the server sent a response → **RemoteDisconnected**.

**After refactor:**

- RAG and Guardrails are initialized **once at startup** in `init_chat_services()` (called from lifespan). By the time any `/api/v1/chat` request runs, the singletons already exist.
- `get_rag_service()` and `get_guardrails_service()` are **pure getters**: they only return the existing instance or raise 503/500. No construction, no FAISS load, no embedding init on the request path.
- So the request path no longer pays for RAG/Guardrails init. The only long-running work is retrieval and LLM, which are offloaded to a thread pool and wrapped with timeouts (30s / 60s), so the server either responds or returns 504 before the client times out.

**Summary:** Moving RAG and Guardrails init to startup removes multi-second (or longer) blocking from the first request and prevents the scenario where the client disconnects while the server is still initializing. Timeouts on retrieval and LLM then ensure the server responds or fails with 504 instead of hanging until the client sees RemoteDisconnected.
