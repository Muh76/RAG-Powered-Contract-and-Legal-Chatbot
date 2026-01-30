# Chat endpoint – minimal instrumentation for debugging

Temporary logging was added to `/api/v1/chat` so each request can be traced by a unique `request_id`. No logic was changed.

---

## 1. Exact logging lines added

All lines use the prefix `CHAT_STAGE` and include `request_id=%s` (and optional `stage=...`, `reason=...`, etc.).

| # | Log message | When it runs |
|---|-------------|--------------|
| 1 | `CHAT_STAGE request_id=%s stage=auth_resolved` | Right after auth (handler entered, `current_user` resolved). |
| 2 | `CHAT_STAGE request_id=%s stage=guardrails_query_done valid=%s` | After `validate_query()`; second arg is `query_validation["valid"]`. |
| 3 | `CHAT_STAGE request_id=%s stage=response_return reason=guardrails_reject` | Before returning due to invalid query. |
| 4 | `CHAT_STAGE request_id=%s stage=retrieval_start` | Just before `run_in_executor(search_func)`. |
| 5 | `CHAT_STAGE request_id=%s stage=retrieval_done count=%s` | Right after retrieval; second arg is `len(retrieval_result)`. |
| 6 | `CHAT_STAGE request_id=%s stage=response_return reason=no_retrieval_result` | Before returning when no chunks found. |
| 7 | `CHAT_STAGE request_id=%s stage=post_filter_done sources=%s` | After post-retrieval filtering and building `sources`. |
| 8 | `CHAT_STAGE request_id=%s stage=response_return reason=weak_similarity` | Before returning due to weak similarity. |
| 9 | `CHAT_STAGE request_id=%s stage=llm_call_start` | Just before `run_in_executor(llm.generate_legal_answer)`. |
| 10 | `CHAT_STAGE request_id=%s stage=llm_call_done` | Right after LLM returns. |
| 11 | `CHAT_STAGE request_id=%s stage=response_return reason=citation_failure` | Before returning due to citation guardrail. |
| 12 | `CHAT_STAGE request_id=%s stage=response_return reason=success` | Before final `return response_obj`. |
| 13 | `CHAT_STAGE request_id=%s stage=response_return reason=exception error=%s` | In the outer `except Exception` before raising 500. |

---

## 2. Exact file and locations

**File:** `app/api/routes/chat.py`

| Line (approx) | Location | Inserted line |
|---------------|----------|---------------|
| 21 | Imports | `import uuid` |
| 205–206 | After `start_time = time.time()` | `request_id = str(uuid.uuid4())` and `logger.info("CHAT_STAGE request_id=%s stage=auth_resolved", request_id)` |
| 217 | After `query_validation = guardrails.validate_query(...)` | `logger.info("CHAT_STAGE request_id=%s stage=guardrails_query_done valid=%s", request_id, query_validation.get("valid"))` |
| 222 | Before `return ChatResponse` (guardrails reject) | `logger.info("CHAT_STAGE request_id=%s stage=response_return reason=guardrails_reject", request_id)` |
| 287 | Before retrieval `try` (before `loop = asyncio.get_event_loop()`) | `logger.info("CHAT_STAGE request_id=%s stage=retrieval_start", request_id)` |
| 314–315 | After `await loop.run_in_executor(..., search_func)` | `logger.info("CHAT_STAGE request_id=%s stage=retrieval_done count=%s", request_id, len(retrieval_result) if retrieval_result else 0)` |
| 381 | Before `return ChatResponse` (no retrieval result) | `logger.info("CHAT_STAGE request_id=%s stage=response_return reason=no_retrieval_result", request_id)` |
| 445 | After building `sources` (before retrieval `except`) | `logger.info("CHAT_STAGE request_id=%s stage=post_filter_done sources=%s", request_id, len(sources))` |
| 504 | Before `return ChatResponse` (weak similarity) | `logger.info("CHAT_STAGE request_id=%s stage=response_return reason=weak_similarity", request_id)` |
| 528 | Before `await loop.run_in_executor(..., llm.generate_legal_answer)` | `logger.info("CHAT_STAGE request_id=%s stage=llm_call_start", request_id)` |
| 537 | After `await loop.run_in_executor(..., llm.generate_legal_answer)` | `logger.info("CHAT_STAGE request_id=%s stage=llm_call_done", request_id)` |
| 595 | Before `return ChatResponse` (citation failure) | `logger.info("CHAT_STAGE request_id=%s stage=response_return reason=citation_failure", request_id)` |
| 722 | Before `return response_obj` (success) | `logger.info("CHAT_STAGE request_id=%s stage=response_return reason=success", request_id)` |
| 727 | In `except Exception` before `raise HTTPException(500)` | `logger.error("CHAT_STAGE request_id=%s stage=response_return reason=exception error=%s", request_id, str(e), exc_info=True)` |

---

## 3. What log output shows where the request “dies”

Use a single `request_id` and follow the sequence of `CHAT_STAGE` logs. The **last stage** you see is where the request stopped (or the next stage is where it hung).

| Last stage you see | Interpretation |
|--------------------|----------------|
| `auth_resolved` only | Failure between auth and guardrails (e.g. `get_guardrails_service()` or `validate_query()`). |
| `guardrails_query_done` only | Unlikely; next step is either early return (guardrails_reject) or retrieval_start. If you never see guardrails_reject or retrieval_start, something odd before retrieval. |
| `retrieval_start` but not `retrieval_done` | Request is stuck or crashed **during retrieval** (executor/FAISS/search). Typical for “connection closed without response” if the client times out while waiting here. |
| `retrieval_done` but not `post_filter_done` | Failure during post-retrieval filtering or building `sources` (e.g. exception in that block). |
| `post_filter_done` but not `llm_call_start` | Failure between post-filter and LLM (e.g. weak-similarity return, or exception in that path). |
| `llm_call_start` but not `llm_call_done` | Request is stuck or crashed **in the LLM call** (OpenAI call in executor). Very common for “connection closed without response” if the client times out while waiting for OpenAI. |
| `llm_call_done` but not `response_return reason=...` | Failure after LLM (guardrails, building response, or exception before any return). |
| `response_return reason=success` | Request completed; FastAPI should send the response. If the client still sees “connection closed without response”, the problem is after the handler (e.g. middleware, serialization, or client). |
| `response_return reason=exception` | Handler raised; 500 should be sent. If the client does not see a 500, the failure is after the handler or in the client. |

**Most likely for “Remote end closed connection without response”:**

- Last log = `retrieval_start` → **retrieval** is the failure point (stuck or very slow).
- Last log = `llm_call_start` → **LLM** is the failure point (stuck or very slow; no timeout on OpenAI).

**How to use:** Reproduce the failure, then grep backend logs for `CHAT_STAGE` and the same `request_id` (or the most recent request_id if you don’t have it from the client). The last `CHAT_STAGE` line for that `request_id` is the stage where the request died or after which it hung.
