# Legal Chatbot - Pydantic Models

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum


class ChatMode(str, Enum):
    """Chat response modes"""
    SOLICITOR = "solicitor"
    PUBLIC = "public"


class SafetyFlag(str, Enum):
    """Safety flag types"""
    HARMFUL = "harmful"
    NON_LEGAL = "non_legal"
    PROMPT_INJECTION = "prompt_injection"
    PII_DETECTED = "pii_detected"


class Source(BaseModel):
    """Source document information"""
    chunk_id: str
    title: str
    url: Optional[str] = None
    text_snippet: str
    similarity_score: float
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SafetyReport(BaseModel):
    """Safety analysis report"""
    is_safe: bool
    flags: List[SafetyFlag] = Field(default_factory=list)
    confidence: float
    reasoning: str


class LatencyAndScores(BaseModel):
    """Performance metrics"""
    retrieval_time_ms: float
    generation_time_ms: float
    total_time_ms: float
    retrieval_score: float
    answer_relevance_score: float


class ChatRequest(BaseModel):
    """Chat request model"""
    query: str = Field(..., min_length=1, max_length=1000)
    mode: ChatMode = ChatMode.PUBLIC
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    top_k: int = Field(default=10, ge=1, le=20)
    include_sources: bool = True


class ChatResponse(BaseModel):
    """Chat response model"""
    answer: str
    sources: List[Source] = Field(default_factory=list)
    safety: SafetyReport
    metrics: LatencyAndScores
    confidence_score: float = Field(ge=0.0, le=1.0)
    legal_jurisdiction: str = "UK"
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class DocumentUploadRequest(BaseModel):
    """Document upload request"""
    file_name: str
    file_type: str
    content: bytes
    metadata: Dict[str, Any] = Field(default_factory=dict)


class DocumentUploadResponse(BaseModel):
    """Document upload response"""
    document_id: str
    status: str
    message: str
    chunks_created: int


class FeedbackRequest(BaseModel):
    """User feedback request"""
    session_id: str
    response_id: str
    rating: int = Field(ge=1, le=5)
    comment: Optional[str] = None


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    timestamp: datetime
    version: str
    services: Dict[str, str]


class ErrorResponse(BaseModel):
    """Error response model"""
    error: str
    message: str
    details: Optional[Dict[str, Any]] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class FusionStrategy(str, Enum):
    """Fusion strategy types"""
    RRF = "rrf"
    WEIGHTED = "weighted"


class MetadataFilterRequest(BaseModel):
    """Metadata filter request model"""
    field: str
    value: Any
    operator: str = "eq"  # eq, in, not_in, contains, etc.


class HybridSearchRequest(BaseModel):
    """Hybrid search request model"""
    query: str = Field(..., min_length=1, max_length=1000)
    top_k: int = Field(default=10, ge=1, le=50)
    fusion_strategy: FusionStrategy = FusionStrategy.RRF
    bm25_weight: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    semantic_weight: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    similarity_threshold: float = Field(default=0.0, ge=0.0, le=1.0)
    metadata_filters: List[MetadataFilterRequest] = Field(default_factory=list)
    pre_filter: bool = True
    include_explanation: bool = False  # Add explainability information
    highlight_sources: bool = False  # Highlight matched terms in sources


class HybridSearchResult(BaseModel):
    """Hybrid search result model"""
    chunk_id: str
    text: str
    similarity_score: float  # Fused score
    bm25_score: Optional[float] = None
    semantic_score: Optional[float] = None
    rerank_score: Optional[float] = None  # Cross-encoder reranking score
    bm25_rank: Optional[int] = None
    semantic_rank: Optional[int] = None
    rerank_rank: Optional[int] = None  # Reranked rank
    rank: int
    section: str
    title: Optional[str] = None
    source: Optional[str] = None
    jurisdiction: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    # Explainability fields (if include_explanation=True)
    explanation: Optional[str] = None  # Why this document was retrieved
    confidence: Optional[float] = None  # Confidence score (0-1)
    matched_terms: Optional[List[str]] = None  # Matched query terms
    highlighted_text: Optional[str] = None  # Text with highlighted matches (if highlight_sources=True)
    matched_spans: Optional[List[Dict[str, int]]] = None  # [(start, end), ...] for matches


class HybridSearchResponse(BaseModel):
    """Hybrid search response model"""
    query: str
    results: List[HybridSearchResult] = Field(default_factory=list)
    total_results: int
    fusion_strategy: str
    search_time_ms: float
    bm25_results_count: Optional[int] = None
    semantic_results_count: Optional[int] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
