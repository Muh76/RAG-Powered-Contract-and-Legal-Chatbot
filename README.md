# Legal Chatbot - AI-Powered Legal Assistant

A production-ready legal chatbot built with RAG (Retrieval-Augmented Generation) for UK legal system with future localization for Iranian market.

## 🎯 Project Overview

This project demonstrates end-to-end AI system development with:
- **Phase 1**: MVP chatbot with UK legal corpus (CUAD + Legislation.gov.uk)
- **Phase 2**: Advanced RAG with hybrid retrieval and explainability
- **Phase 3**: Multilingual support (English + Farsi) and role-based responses
- **Phase 4**: Enterprise features with authentication and monetization

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
   - FastAPI docs: http://localhost:8000/docs
   - API Health: http://localhost:8000/api/v1/health

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

### Phase 2 (Advanced RAG)
- 🔄 Hybrid retrieval (dense + BM25)
- 🔄 Cross-encoder reranking
- 🔄 Explainability and source highlighting
- 🔄 Red-team testing and prompt injection detection

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
- **ML/AI**: OpenAI API, SentenceTransformers, RAGAS
- **Vector DB**: Qdrant/pgvector
- **Infrastructure**: Docker, Docker Compose
- **Monitoring**: Prometheus, Grafana, OpenTelemetry

## 📈 Business Value

- **For Law Firms**: Reliable, cited legal answers with audit trails
- **For Public**: Accessible legal information with clear disclaimers
- **For Developers**: Production-ready RAG system with enterprise features

## 🔒 Security & Compliance

- Privacy-first design with PII redaction
- Audit logging for compliance
- Multi-tenant data isolation
- Security scanning and dependency checks

## 📝 Documentation

- [Architecture Overview](docs/architecture/README.md)
- [Security Guidelines](docs/security/README.md)
- [Privacy Policy](docs/privacy/README.md)
- [Evaluation Metrics](docs/eval/README.md)

## 📋 Project Status

- **Phase 1**: ✅ **COMPLETED** - MVP with RAG pipeline, guardrails, and web interface
- **Phase 2**: 🔄 **IN PROGRESS** - Advanced RAG with hybrid retrieval and explainability  
- **Phase 3**: 📋 **PLANNED** - Multilingual support and role-based responses
- **Phase 4**: 📋 **PLANNED** - Enterprise features and monetization

## 🤝 Contributing

This is a portfolio project demonstrating production-ready AI system development with a focus on legal domain applications.

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 📧 Contact

- **Email**: mj.babaie@gmail.com
- **LinkedIn**: https://www.linkedin.com/in/mohammadbabaie/
- **GitHub**: https://github.com/Muh76

---

*This chatbot provides educational information only and does not constitute legal advice. Always consult with qualified legal professionals for specific legal matters.*
