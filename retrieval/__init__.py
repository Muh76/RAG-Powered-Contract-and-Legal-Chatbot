# Legal Chatbot - Retrieval Package
# Phase 2: Advanced RAG Retrieval Components

from .semantic_retriever import SemanticRetriever
from .bm25_retriever import BM25Retriever
from .metadata_filter import MetadataFilter, FilterOperator
from .hybrid_retriever import AdvancedHybridRetriever, FusionStrategy
from .embeddings.embedding_generator import EmbeddingGenerator, EmbeddingConfig
from .rerankers.cross_encoder_reranker import CrossEncoderReranker
from .explainability import ExplainabilityAnalyzer, RetrievalExplanation
from .red_team_tester import RedTeamTester, RedTeamTestResult

__all__ = [
    "SemanticRetriever",
    "BM25Retriever",
    "MetadataFilter",
    "FilterOperator",
    "AdvancedHybridRetriever",
    "FusionStrategy",
    "EmbeddingGenerator",
    "EmbeddingConfig",
    "CrossEncoderReranker",
    "ExplainabilityAnalyzer",
    "RetrievalExplanation",
    "RedTeamTester",
    "RedTeamTestResult"
]
