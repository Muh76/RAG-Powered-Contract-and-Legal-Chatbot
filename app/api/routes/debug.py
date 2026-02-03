"""
Legal Chatbot - Debug API Endpoints
RAG pipeline inspection (no LLM calls).
"""

from fastapi import APIRouter, Depends
from typing import Dict, Any, Optional

from app.auth.dependencies import get_current_active_user
from app.auth.models import User
from app.core.config import settings
from app.api.routes.chat import get_rag_service

router = APIRouter(prefix="/debug", tags=["debug"])


@router.get("/rag")
async def get_rag_debug(
    current_user: User = Depends(get_current_active_user),
) -> Dict[str, Any]:
    """
    Return RAG pipeline debug info. Authenticated. No LLM calls.
    """
    rag = get_rag_service()

    embedding_model = settings.EMBEDDING_MODEL
    embedding_dimension = settings.EMBEDDING_DIMENSION

    faiss_index_dim: Optional[int] = None
    faiss_vector_count: Optional[int] = None
    if rag.faiss_index is not None:
        faiss_index_dim = getattr(rag.faiss_index, "d", None)
        faiss_vector_count = getattr(rag.faiss_index, "ntotal", None)

    metadata_chunk_count = len(rag.chunk_metadata) if rag.chunk_metadata else 0
    hybrid_enabled = rag.hybrid_retriever is not None

    return {
        "embedding_model": embedding_model,
        "embedding_dimension": embedding_dimension,
        "faiss_index_dimension": faiss_index_dim,
        "faiss_vector_count": faiss_vector_count,
        "metadata_chunk_count": metadata_chunk_count,
        "hybrid_enabled": hybrid_enabled,
    }
