# Legal Chatbot - Phase 1 Development Notebook

## Phase 1: Foundations (MVP Chatbot)

### Goals
- ✅ Safe, domain-specific, end-to-end chatbot
- ✅ Simple web UI (Streamlit) for Q&A
- ✅ Backend with FastAPI serving the chatbot
- ✅ Vector DB storing legal texts (CUAD dataset + UK legislation)
- ✅ Retrieval-Augmented Generation (RAG) with citations
- ✅ Guardrails v1: domain gating and safety filters

### Implementation Steps

#### 1. Data Ingestion & Indexing
- [ ] Download CUAD dataset
- [ ] Download UK legislation from Legislation.gov.uk
- [ ] Build document loaders (PDF/Docx → text)
- [ ] Implement chunking strategy (800-1200 tokens with overlap)
- [ ] Create embeddings using sentence-transformers
- [ ] Store in vector database (Qdrant)

#### 2. Retrieval Pipeline (RAG v1)
- [ ] Implement dense vector retrieval
- [ ] Add top-k retrieval (8-12 documents)
- [ ] Create prompt template with citation enforcement
- [ ] Integrate with OpenAI API for generation
- [ ] Implement response validation

#### 3. Guardrails v1
- [ ] Domain gating (refuse non-legal questions)
- [ ] Safety filters (harmful content detection)
- [ ] Grounding check (insufficient context detection)
- [ ] Citation requirement validation
- [ ] Compliance banner implementation

#### 4. API & UI Integration
- [ ] Connect FastAPI with actual RAG pipeline
- [ ] Update Streamlit UI with real responses
- [ ] Add source highlighting in UI
- [ ] Implement feedback collection
- [ ] Add telemetry and monitoring

### Success Criteria
- [ ] Can answer basic UK legal questions with citations
- [ ] Refuses non-legal queries appropriately
- [ ] Response time < 3 seconds
- [ ] All responses include source citations
- [ ] Safety filters block harmful content

### Next Phase Preview
Phase 2 will add:
- Hybrid retrieval (dense + BM25)
- Cross-encoder reranking
- Explainability features
- Advanced guardrails
- Red-team testing
