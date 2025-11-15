# Legal Chatbot - Search Route
# Phase 2: Hybrid Search API Endpoint

import os
import sys
from pathlib import Path
from fastapi import APIRouter, HTTPException, Query
from typing import Optional, List
import time
import logging
from concurrent.futures import ThreadPoolExecutor
import asyncio

from app.models.schemas import (
    HybridSearchRequest,
    HybridSearchResponse,
    HybridSearchResult,
    MetadataFilterRequest,
    FusionStrategy
)
from app.services.rag_service import RAGService
from retrieval.metadata_filter import MetadataFilter, FilterOperator

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

logger = logging.getLogger(__name__)

router = APIRouter()

# Shared thread pool executor for blocking operations
_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="search_worker")

# Initialize RAG service (singleton pattern)
rag_service_hybrid = None


def get_rag_service_hybrid():
    """Get or create RAG service with hybrid search enabled"""
    global rag_service_hybrid
    if rag_service_hybrid is None:
        logger.info("🔄 Initializing RAGService with hybrid search...")
        try:
            rag_service_hybrid = RAGService(use_hybrid=True)
            logger.info("✅ RAGService with hybrid search initialized")
        except Exception as e:
            logger.error(f"❌ CRITICAL: RAGService initialization failed: {e}")
            logger.error("This may be due to PyTorch segfault. The service will return empty results.")
            # Create a minimal RAGService that will return empty results
            from app.services.rag_service import RAGService as RAGServiceClass
            rag_service_hybrid = RAGServiceClass.__new__(RAGServiceClass)
            rag_service_hybrid.use_hybrid = False
            rag_service_hybrid.hybrid_retriever = None
            rag_service_hybrid.chunk_metadata = []
            logger.warning("⚠️ RAGService created in degraded mode (no hybrid search)")
    return rag_service_hybrid


def create_metadata_filter(metadata_filters: List[MetadataFilterRequest]) -> Optional[MetadataFilter]:
    """Create metadata filter from request filters"""
    if not metadata_filters:
        return None
    
    metadata_filter = MetadataFilter()
    
    for filter_req in metadata_filters:
        operator_str = filter_req.operator.lower()
        
        try:
            # Map operator string to FilterOperator enum
            if operator_str == "eq" or operator_str == "equals":
                metadata_filter.add_equals_filter(filter_req.field, str(filter_req.value))
            elif operator_str == "in":
                # Ensure value is a list
                if isinstance(filter_req.value, list):
                    metadata_filter.add_in_filter(filter_req.field, [str(v) for v in filter_req.value])
                else:
                    metadata_filter.add_in_filter(filter_req.field, [str(filter_req.value)])
            elif operator_str == "not_in" or operator_str == "nin":
                if isinstance(filter_req.value, list):
                    metadata_filter.add_not_in_filter(filter_req.field, [str(v) for v in filter_req.value])
                else:
                    metadata_filter.add_not_in_filter(filter_req.field, [str(filter_req.value)])
            elif operator_str == "contains":
                metadata_filter.add_contains_filter(filter_req.field, str(filter_req.value))
            else:
                # Generic filter using FilterOperator enum
                try:
                    operator = FilterOperator(operator_str)
                    metadata_filter.add_filter(filter_req.field, filter_req.value, operator)
                except ValueError:
                    logger.warning(f"Unknown filter operator: {operator_str}, using equals")
                    metadata_filter.add_equals_filter(filter_req.field, str(filter_req.value))
        except Exception as e:
            logger.warning(f"Error adding filter {filter_req.field}={filter_req.value}: {e}")
            continue
    
    return metadata_filter if not metadata_filter.is_empty() else None


@router.post("/search/hybrid", response_model=HybridSearchResponse)
async def hybrid_search(request: HybridSearchRequest):
    """
    Hybrid search endpoint combining BM25 + Semantic search with metadata filtering.
    
    Args:
        request: Hybrid search request with query, filters, and configuration
        
    Returns:
        Hybrid search response with results including BM25, semantic, and fused scores
    """
    start_time = time.time()
    
    try:
        # Get RAG service with hybrid search
        rag_service = get_rag_service_hybrid()
        
        if not rag_service.hybrid_retriever:
            raise HTTPException(
                status_code=503,
                detail="Hybrid retriever not available. Ensure data ingestion is complete and FAISS index exists."
            )
        
        # Create metadata filter if filters are provided
        metadata_filter = None
        if request.metadata_filters:
            metadata_filter = create_metadata_filter(request.metadata_filters)
            logger.debug(f"Created metadata filter with {len(request.metadata_filters)} conditions")
        
        # Run search in thread pool to avoid blocking event loop
        def perform_search():
            return rag_service.search(
                query=request.query,
                top_k=request.top_k,
                use_hybrid=True,
                fusion_strategy=request.fusion_strategy.value,
                metadata_filter=metadata_filter,
                include_explanation=request.include_explanation,
                highlight_sources=request.highlight_sources
            )
        
        # Execute search asynchronously
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(_executor, perform_search)
        
        search_time_ms = (time.time() - start_time) * 1000
        
        # Count results with BM25 and semantic scores
        bm25_results_count = sum(1 for r in results if r.get("bm25_score") is not None)
        semantic_results_count = sum(1 for r in results if r.get("semantic_score") is not None)
        
        # Format results
        formatted_results = []
        for i, result in enumerate(results, 1):
            # Convert matched_spans to list of dicts if present
            matched_spans = result.get("matched_spans")
            if matched_spans and isinstance(matched_spans, list) and matched_spans:
                if isinstance(matched_spans[0], tuple):
                    matched_spans = [{"start": span[0], "end": span[1]} for span in matched_spans]
            
            formatted_results.append(HybridSearchResult(
                chunk_id=result.get("chunk_id", ""),
                text=result.get("text", ""),
                similarity_score=result.get("similarity_score", 0.0),
                bm25_score=result.get("bm25_score"),
                semantic_score=result.get("semantic_score"),
                rerank_score=result.get("rerank_score"),
                bm25_rank=result.get("bm25_rank"),
                semantic_rank=result.get("semantic_rank"),
                rerank_rank=result.get("rerank_rank"),
                rank=result.get("rank", i),
                section=result.get("section", "Unknown"),
                title=result.get("title"),
                source=result.get("source"),
                jurisdiction=result.get("jurisdiction"),
                metadata=result.get("metadata", {}),
                # Explainability fields
                explanation=result.get("explanation"),
                confidence=result.get("confidence"),
                matched_terms=result.get("matched_terms"),
                highlighted_text=result.get("highlighted_text"),
                matched_spans=matched_spans
            ))
        
        logger.info(
            f"Hybrid search completed: query='{request.query[:50]}...', "
            f"results={len(results)}, time={search_time_ms:.1f}ms"
        )
        
        return HybridSearchResponse(
            query=request.query,
            results=formatted_results,
            total_results=len(results),
            fusion_strategy=request.fusion_strategy.value,
            search_time_ms=search_time_ms,
            bm25_results_count=bm25_results_count,
            semantic_results_count=semantic_results_count
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in hybrid search: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@router.get("/search/hybrid", response_model=HybridSearchResponse)
async def hybrid_search_get(
    query: str = Query(..., min_length=1, max_length=1000),
    top_k: int = Query(default=10, ge=1, le=50),
    fusion_strategy: str = Query(default="rrf", regex="^(rrf|weighted)$"),
    include_explanation: bool = Query(default=False, description="Include explainability information"),
    highlight_sources: bool = Query(default=False, description="Highlight matched terms in source text"),
    similarity_threshold: float = Query(default=0.0, ge=0.0, le=1.0),
    jurisdiction: Optional[str] = None,
    document_type: Optional[str] = None,
    source: Optional[str] = None
):
    """
    GET endpoint for hybrid search with query parameters.
    
    Args:
        query: Search query text
        top_k: Number of top results to return
        fusion_strategy: Fusion strategy ("rrf" or "weighted")
        similarity_threshold: Minimum similarity score
        jurisdiction: Filter by jurisdiction (optional)
        document_type: Filter by document type (optional)
        source: Filter by source (optional)
        
    Returns:
        Hybrid search response
    """
    # Build metadata filters from query parameters
    metadata_filters = []
    
    if jurisdiction:
        metadata_filters.append(MetadataFilterRequest(
            field="jurisdiction",
            value=jurisdiction,
            operator="eq"
        ))
    
    if document_type:
        metadata_filters.append(MetadataFilterRequest(
            field="document_type",
            value=document_type,
            operator="eq"
        ))
    
    if source:
        metadata_filters.append(MetadataFilterRequest(
            field="source",
            value=source,
            operator="eq"
        ))
    
    # Create request from query parameters
    request = HybridSearchRequest(
        query=query,
        top_k=top_k,
        fusion_strategy=FusionStrategy(fusion_strategy),
        similarity_threshold=similarity_threshold,
        metadata_filters=metadata_filters,
        include_explanation=include_explanation,
        highlight_sources=highlight_sources
    )
    
    # Call POST endpoint logic
    return await hybrid_search(request)

