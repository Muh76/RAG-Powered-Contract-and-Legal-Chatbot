# Full Mock Technical Interview — Legal Chatbot (ML Engineer)

**Format:** One complete interview script. Read it as a full run-through or practice answering each question out loud, then compare with the sample answers.

**Focus areas:** RAG architecture, retrieval design, embeddings, hallucination mitigation, evaluation, scaling, tradeoffs.

---

## Interviewer: "Thanks for joining. I’d like to go deep on your Legal Chatbot project. Walk me through the path a user query takes from the moment it hits your API until the user gets a final answer. What are the main stages, and where does retrieval happen relative to the LLM call?"

**Sample answer:**

The request hits the FastAPI chat endpoint in `app/api/routes/chat.py`. First we run **query guardrails**—domain check (legal vs non-legal keywords), harmful-content patterns, and minimum legal relevance. If that fails we return a short refusal and don’t call retrieval or the LLM.

If the query passes, we do **retrieval**: we call the RAG service’s `search()` method. That runs hybrid search—BM25 plus semantic over FAISS with OpenAI embeddings—then RRF fusion and optionally cross-encoder reranking. We can also pull in the user’s private uploaded documents and merge them with the public corpus via RRF. So retrieval happens **before** any LLM call; the LLM never sees the raw query without context.

Next we **assemble context**: the top-k chunks are formatted as "[1] Title\nchunk text", "[2] ...", and so on. We also run a **similarity check**—if average similarity is below a threshold (e.g. 0.4) we refuse to answer and don’t call the LLM, to avoid low-quality grounding.

Then we **call the LLM**: we build a system prompt (solicitor or public mode) with strict citation and anti-hallucination rules, and a user prompt that contains the numbered sources and the question. We call OpenAI `chat.completions.create`. When we get the answer we **validate citations**—every sentence must end with [n] and the numbers must match the sources. If not, we do up to two **repair** calls with a rewrite prompt; if it still fails we return a refusal message. Finally we run **response guardrails** (citation presence, grounding, answer quality) and return the ChatResponse with answer, sources, safety, and metrics.

So the order is: **guardrails (query) → retrieval → context assembly → optional similarity gate → LLM → citation validation and retry → guardrails (response) → return.**

---

## Interviewer: "Why did you choose RAG instead of fine-tuning a model for legal Q&A?"

**Sample answer:**

We needed answers that are **grounded in specific, updatable sources**—UK legislation and contract clauses—and that users can **verify** with citations. RAG keeps the knowledge in the corpus: we can add or change documents and re-run ingestion without retraining. The model’s job is to summarize and cite, not to memorize law.

Fine-tuning would bake facts into the weights, make it harder to enforce “only cite these chunks,” and require retraining whenever the corpus changes. For a legal use case, auditability and the ability to point to exact sources matter more than squeezing a bit more fluency from a fine-tuned model. So we chose RAG and doubled down on retrieval quality and citation enforcement instead.

---

## Interviewer: "Describe your retrieval design. Why hybrid (BM25 + semantic) instead of only semantic or only keyword?"

**Sample answer:**

We use **hybrid retrieval**: BM25 for lexical match and semantic search with OpenAI embeddings over a FAISS index. We run both, get two ranked lists, and fuse them with **Reciprocal Rank Fusion (RRF)**—so we don’t have to normalize scores across BM25 and cosine similarity. Optionally we rerank the fused list with a cross-encoder (e.g. ms-marco-MiniLM) and then take top-k.

We went hybrid because **legal queries mix exact and semantic need**: e.g. “Section 1 Employment Rights Act” benefits from BM25, while “when can an employee request flexible working?” benefits from semantic similarity. Pure keyword misses paraphrases; pure semantic can miss precise section names. Hybrid gives us both. We also support **metadata filters** (e.g. jurisdiction, document type) and can **merge the public index with a user’s private document chunks** using RRF so one ranked list goes to the LLM.

---

## Interviewer: "What embedding model do you use and why? What would you change if you had to support 10x more documents?"

**Sample answer:**

We use **OpenAI text-embedding-3-large** with 3072 dimensions. We chose it for quality and consistency: one model for both ingestion and query, so no train–serve skew. We use the API so we don’t run PyTorch or GPU in production, which simplified deployment and avoided segfault issues we had earlier with local embedders. The dimension is fixed so the FAISS index and all chunks use the same 3072d.

If we scaled to 10x more documents, we’d hit **memory and latency** with a single in-process FAISS index. Options: (1) **Shard FAISS**—e.g. by jurisdiction or document type—and query relevant shards. (2) Move to a **managed vector DB** (e.g. Qdrant, which we already have in the stack) so we get distributed search and don’t load the full index in every API replica. (3) Keep the same embedder for compatibility but **re-ingest** into the new store and possibly **quantize** or use IVF in FAISS for faster approximate search. We’d also re-check batch size and rate limits on the OpenAI embedding API for bulk ingestion.

---

## Interviewer: "How do you mitigate hallucinations in this system?"

**Sample answer:**

We do it in several layers.

**1. Retrieval-only context.** The LLM prompt contains only the retrieved chunks, numbered [1], [2], … We explicitly instruct it to answer only from these sources and not to use training data. So the model has no way to “invent” a source that isn’t in the prompt.

**2. Strict citation rules in the prompt.** We require every factual sentence to end with [n] and say we’ll refuse if it can’t be supported. We give examples of good and bad format. That pushes the model toward citing and away from unsupported claims.

**3. Post-generation validation and retry.** We run two checks on the raw answer: (a) regex and logic to ensure every sentence ends with a citation and that citation numbers exist in the source list, and (b) that we have at least a minimum number of chunks. If validation fails we send a **repair** prompt: “Rewrite this answer so every sentence ends with [n] from the sources; do not add new claims.” We do up to two repair attempts. If it still fails we **refuse** and return a standard message instead of showing an uncited answer.

**4. Similarity gate before the LLM.** If the average similarity of retrieved chunks is below a threshold (e.g. 0.4), we don’t call the LLM at all; we return a message saying we don’t have enough relevant sources. That avoids generating when grounding is weak.

**5. Response guardrails.** We check that the response has citations, has enough retrieval grounding, and meets basic quality (e.g. length). If not we can reject or flag.

So mitigation is: **limit context to retrieval → prompt rules → citation validation and retry → similarity gate → response guardrails.**

---

## Interviewer: "How do you evaluate this RAG system—offline and in production?"

**Sample answer:**

**Offline / pre-release:** We have **108+ E2E tests** that hit the API (chat, search, agentic, auth, document upload). We also have **red-team tests**: a `RedTeamTester` that loads adversarial cases (prompt injection, off-topic, harmful content) and checks that guardrails block or respond correctly. So we evaluate correctness of behavior and safety in an automated way. We have **ragas** and **trulens-eval** in the stack for RAG-specific metrics (e.g. faithfulness, answer relevance); the main automated signal we use today is the E2E and red-team suite. For retrieval we could add a golden set of query–relevant-doc pairs and measure recall@k or MRR; that’s a natural next step.

**In production:** We expose **health endpoints** (e.g. DB, Redis, optional OpenAI) and **metrics** (latency, error rate, request volume, tool usage). We don’t yet have a continuous hallucination monitor; that would be something like sampling requests, storing prompt + response + sources, and running a faithfulness scorer (e.g. NLI or a small model) on a sample, then alerting if scores drop. So evaluation today is strong on behavior and safety tests, with room to add more retrieval and faithfulness metrics and production monitoring.

---

## Interviewer: "How would you scale this to many more users and much more data?"

**Sample answer:**

**More users:** The API is stateless, so we **scale horizontally**—e.g. more Cloud Run replicas. Auth is JWT/OAuth per request, so no sticky sessions. We’d add **rate limiting** and **quotas** per user or tenant, and possibly **caching** for identical or near-identical queries (e.g. cache embedding + top-k result or full response) to reduce load and cost.

**More data:** Right now FAISS and BM25 are **in-process** per replica. If the corpus grew a lot, a single FAISS index wouldn’t fit in memory or would be slow. We’d (1) move to a **distributed vector store** (e.g. Qdrant, which we already depend on) so we don’t duplicate the full index on every pod, or (2) **shard** by metadata (e.g. jurisdiction, doc type) and query only relevant shards. For BM25 we’d need a distributed or external index (e.g. Elasticsearch, which we have in the stack) instead of in-memory rank_bm25. **Embedding** for ingestion would stay batch OpenAI with rate limiting; we might precompute and store embeddings in the vector DB rather than re-embed on every ingest.

**Cost and latency:** We’d watch **token and API usage** per customer, use **smaller or cheaper models** where acceptable, and consider **async or batch** for non-real-time paths. So: horizontal scaling, caching, moving retrieval to a shared store, and per-tenant limits and monitoring.

---

## Interviewer: "What tradeoffs did you make in this project, and what would you do differently next time?"

**Sample answer:**

**Tradeoffs we made:**

- **OpenAI for embeddings and LLM vs open-source.** We traded cost and vendor dependency for reliability and simpler ops (no GPU, no PyTorch in the hot path). Open-source would save cost and help with data residency; we’d need to invest in hosting and evaluation.

- **FAISS in-process vs a vector DB.** We traded operational simplicity and latency for scalability. FAISS is fast and easy to ship with the app, but every replica holds the full index. For a single-region, moderate-size corpus that’s fine; for 10x data or many regions we’d move to a vector DB.

- **Citation retry loop (up to 2 repairs) vs single shot.** We traded latency and token cost for higher citation compliance. Some requests take 2–3 LLM calls; we accepted that so we rarely return uncited answers.

- **Fixed chunk size (e.g. 1000 chars) vs semantic or section-based chunking.** We traded simplicity and uniform context length for optimal boundaries. Legal sections might be cleaner; we’d A/B test if we had more time.

- **Guardrails as rule-based (keywords, regex) vs ML classifiers.** We traded some flexibility for interpretability and no extra training. For harder cases (e.g. subtle prompt injection) we’d consider a small classifier or an LLM-based guardrail.

**What I’d do differently:** (1) Add **retrieval evaluation** earlier—e.g. recall@k on a curated set—so we can tune top-k and fusion. (2) **Version the corpus and index** (e.g. by date or config) so we can roll back or A/B test. (3) **Production monitoring for faithfulness**—sample prompts and responses and score against sources. (4) Consider **hybrid chunking** (e.g. by section when available, fallback to fixed size) from the start.

---

## Interviewer: "How does your citation validation work technically? How do you decide that an answer is compliant?"

**Sample answer:**

We have two checks, both in `app/services/llm_service.py`.

**1. Citation validity.** We extract all `[n]`-style references from the answer (e.g. with regex) and check that every number `n` is between 1 and the number of sources we provided (e.g. 1 to 10 if we had 10 chunks). We also check that the answer has at least one citation. So we get a boolean: valid_citations and has_citations.

**2. Every sentence ends with a citation.** We split the answer into sentences (e.g. by period, newline) and check that each sentence ends with a citation token—i.e. the last non-whitespace characters are something like `[1]` or `[2][3]`. We don’t allow citations only in the middle of a sentence.

**Compliance** = (1) and (2) both true. If not, we run the repair loop: we send the original answer plus the same sources and a system prompt that says “rewrite so every sentence ends with [n]; do not add new claims.” We re-validate the new answer. We do at most two repair attempts; if still non-compliant we replace the answer with a fixed refusal message so we never return an uncited answer to the user.

---

## Interviewer: "Why RRF for fusion instead of weighted combination of BM25 and semantic scores?"

**Sample answer:**

RRF—Reciprocal Rank Fusion—combines **ranks** instead of raw scores. For each document we sum 1/(k + rank) across the two lists (e.g. k=60). That gives a single score we use to re-rank.

We use RRF because **BM25 and cosine similarity are on different scales**. BM25 scores can be large integers; cosine is in [-1, 1]. Normalizing them or choosing weights is brittle and corpus-dependent. RRF is **scale-invariant**: we only care about order in each list. It’s also robust—a document that ranks well in one list and poorly in the other still gets a reasonable fused score. We have a **weighted** fusion option in config (linear combination of normalized scores) for when we want to favor one signal (e.g. more semantic), but the default is RRF for simplicity and stability.

---

## Interviewer: "If retrieval returns irrelevant chunks for a query, what happens? How do you limit the damage?"

**Sample answer:**

Several things limit the damage:

**1. Similarity threshold before the LLM.** We compute the average similarity of the retrieved chunks. If it’s below a threshold (e.g. 0.4), we **don’t call the LLM**. We return a message like “the retrieved sources don’t have enough relevance to answer confidently” and suggest rephrasing. So irrelevant retrieval often leads to no generation at all.

**2. Citation rules.** The prompt says “answer only from the provided sources” and “if you can’t support it, refuse.” So if the chunks are irrelevant, the model is encouraged to say it can’t answer rather than to fabricate.

**3. Minimum chunk count.** We require at least a minimum number of chunks (e.g. 2) for grounding; otherwise guardrails can flag insufficient grounding.

**4. Post-retrieval filtering (in some paths).** For certain query types we filter results by Act or topic (e.g. employment vs sale of goods) so we don’t pass clearly off-topic chunks to the LLM.

We could go further: **retrieval confidence** (e.g. max or mean score) and a stricter gate, or **two-stage retrieval** (cheap retriever first, then reranker) to drop bad candidates before the LLM. The main defense today is the similarity gate and the “refuse if not in sources” instruction.

---

## Interviewer: "One last question: how does the agentic mode differ from the single-shot chat path, and when would you use each?"

**Sample answer:**

**Single-shot chat:** One query → one retrieval call → one context assembly → one (or a few, for citation retry) LLM call → one answer. Good for single questions that fit in one retrieval + one response.

**Agentic mode:** We use LangChain/LangGraph with an agent that has **tools**: legal search (same hybrid RAG), statute lookup, document analyzer. The agent can **plan**, call one or more tools, look at results, and call again. So for “Compare Section 1 of the Employment Rights Act with the Equality Act on notice periods” it might do two searches or lookups and then synthesize. The agent loop runs for multiple steps (e.g. max iterations 5) until it has enough to answer or hits a limit.

We use **single-shot** for straightforward Q&A and when we want predictable latency and cost. We use **agentic** when the user question clearly needs multiple steps—comparisons, multi-document reasoning, or follow-up tool use—and we’re okay with higher latency and token cost. Both paths use the same retrieval and citation philosophy; the agentic path just orchestrates multiple tool calls before the final answer.

---

**End of mock interview.** Use this script to practice answering out loud or to review the full flow before a real interview.
