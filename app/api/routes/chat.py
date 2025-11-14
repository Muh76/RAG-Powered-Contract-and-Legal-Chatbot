# Legal Chatbot - Chat Route
# Updated to use real RAG services

import os
import sys
from pathlib import Path
from fastapi import APIRouter, HTTPException
from app.models.schemas import ChatRequest, ChatResponse, Source, SafetyReport, SafetyFlag, LatencyAndScores
from app.services.rag_service import RAGService
from app.services.guardrails_service import GuardrailsService
from app.services.llm_service import LLMService
from datetime import datetime
import time
import logging
from concurrent.futures import ThreadPoolExecutor
import asyncio

# Add project root to Python path (for testing/debugging)
project_root = Path(__file__).parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

logger = logging.getLogger(__name__)

router = APIRouter()

# Shared thread pool executor for blocking operations (prevents segfaults from PyTorch in async context)
# Use a single worker to avoid potential deadlocks with PyTorch/FAISS
_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="rag_worker")

# Initialize services (singleton pattern)
rag_service = None
guardrails_service = None
llm_service = None

def get_rag_service():
    global rag_service
    if rag_service is None:
        logger.info("🔄 Initializing RAGService...")
        try:
            rag_service = RAGService()
            logger.info("✅ RAGService initialized")
        except Exception as e:
            logger.error(f"❌ CRITICAL: RAGService initialization failed: {e}")
            logger.error("This may be due to PyTorch segfault. The service will return empty results.")
            # Create a minimal RAGService that will return empty results
            # Import here to avoid circular import
            from app.services.rag_service import RAGService as RAGServiceClass
            rag_service = RAGServiceClass.__new__(RAGServiceClass)  # Create without calling __init__
            rag_service.embedding_gen = None
            rag_service.faiss_index = None
            rag_service.chunk_metadata = []
            logger.warning("⚠️ RAGService created in degraded mode (no embeddings)")
    return rag_service

def get_guardrails_service():
    global guardrails_service
    if guardrails_service is None:
        guardrails_service = GuardrailsService()  # Let exception propagate
        logger.info("✅ GuardrailsService initialized")
    return guardrails_service

def get_llm_service():
    global llm_service
    if llm_service is None:
        logger.info("🔄 Initializing LLMService...")
        llm_service = LLMService()  # Let exception propagate
        logger.info("✅ LLMService initialized")
    return llm_service

def map_reason_to_safety_flag(reason: str) -> SafetyFlag:
    """Map guardrails reason to SafetyFlag enum"""
    reason_lower = reason.lower()
    if "harmful" in reason_lower or "injection" in reason_lower:
        return SafetyFlag.HARMFUL
    elif "domain" in reason_lower or "non_legal" in reason_lower or "insufficient_legal" in reason_lower:
        return SafetyFlag.NON_LEGAL
    elif "pii" in reason_lower or "personal" in reason_lower:
        return SafetyFlag.PII_DETECTED
    elif "injection" in reason_lower or "prompt" in reason_lower:
        return SafetyFlag.PROMPT_INJECTION
    else:
        # Default to NON_LEGAL for unknown reasons
        return SafetyFlag.NON_LEGAL

@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Chat endpoint with RAG pipeline"""
    start_time = time.time()
    
    try:
        # 1. Validate query with guardrails
        try:
            guardrails = get_guardrails_service()
        except Exception as e:
            logger.error(f"Guardrails service error: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Guardrails service error: {str(e)}")
        
        query_validation = guardrails.validate_query(request.query)
        
        if not query_validation["valid"]:
            # Map reason to SafetyFlag enum
            safety_flag = map_reason_to_safety_flag(query_validation["reason"])
            
            return ChatResponse(
                answer=query_validation["message"],
                sources=[],
                safety=SafetyReport(
                    is_safe=False,
                    flags=[safety_flag],
                    confidence=0.9,
                    reasoning=query_validation.get("suggestion", "")
                ),
                metrics=LatencyAndScores(
                    retrieval_time_ms=0.0,
                    generation_time_ms=0.0,
                    total_time_ms=(time.time() - start_time) * 1000,
                    retrieval_score=0.0,
                    answer_relevance_score=0.0
                ),
                confidence_score=0.0,
                legal_jurisdiction="UK"
            )
        
        # 2. Retrieve relevant chunks
        try:
            rag = get_rag_service()
        except Exception as e:
            logger.error(f"RAG service error: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"RAG service error: {str(e)}. Run data ingestion first.")
        
        # CRITICAL FIX: Run blocking PyTorch/FAISS operations in thread pool to prevent segfault
        retrieval_start = time.time()
        try:
            # Use default executor instead of shared one to avoid potential deadlocks
            retrieved_chunks = await asyncio.to_thread(rag.search, request.query, request.top_k)
        except Exception as search_error:
            logger.error(f"CRITICAL: Search failed in thread pool: {search_error}", exc_info=True)
            retrieved_chunks = []
        retrieval_time = (time.time() - retrieval_start) * 1000
        
        if not retrieved_chunks:
            return ChatResponse(
                answer="I couldn't find relevant information in my knowledge base. Please try rephrasing your question.",
                sources=[],
                safety=SafetyReport(
                    is_safe=True,
                    flags=[],
                    confidence=1.0,
                    reasoning="No results found"
                ),
                metrics=LatencyAndScores(
                    retrieval_time_ms=retrieval_time,
                    generation_time_ms=0.0,
                    total_time_ms=(time.time() - start_time) * 1000,
                    retrieval_score=0.0,
                    answer_relevance_score=0.0
                ),
                confidence_score=0.0,
                legal_jurisdiction="UK"
            )
        
        # 3. Generate answer with LLM
        try:
            llm = get_llm_service()
        except ValueError as e:
            logger.error(f"LLM service error: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"LLM service error: {str(e)}. Check OPENAI_API_KEY in .env file.")
        except Exception as e:
            logger.error(f"LLM service error: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"LLM service error: {str(e)}")
        
        # CRITICAL FIX: Run blocking OpenAI API call in thread pool
        generation_start = time.time()
        try:
            # Use default executor instead of shared one to avoid potential deadlocks
            def generate_answer():
                return llm.generate_legal_answer(
                    query=request.query,
                    retrieved_chunks=retrieved_chunks[:3],  # Use top 3 for context
                    mode=request.mode
                )
            result = await asyncio.to_thread(generate_answer)
        except Exception as llm_error:
            logger.error(f"CRITICAL: LLM generation failed in thread pool: {llm_error}", exc_info=True)
            result = {
                "answer": "I encountered an error generating a response. Please try again.",
                "citations": [],
                "citation_validation": {"valid_citations": False}
            }
        generation_time = (time.time() - generation_start) * 1000
        
        # 4. Format sources with error handling
        sources = []
        try:
            for chunk in retrieved_chunks[:3]:
                try:
                    # Safely access chunk data with defaults
                    chunk_id = chunk.get("chunk_id", "unknown")
                    metadata = chunk.get("metadata", {})
                    text = chunk.get("text", "")
                    similarity_score = chunk.get("similarity_score", 0.0)
                    
                    sources.append(
                        Source(
                            chunk_id=chunk_id,
                            title=metadata.get("title", "Unknown"),
                            url=metadata.get("url", ""),
                            text_snippet=text[:200] if text else "",
                            similarity_score=float(similarity_score),
                            metadata=metadata
                        )
                    )
                except Exception as e:
                    logger.error(f"Error formatting chunk: {e}", exc_info=True)
                    # Skip this chunk and continue
                    continue
        except Exception as e:
            logger.error(f"Error formatting sources: {e}", exc_info=True)
            sources = []
        
        # 5. Validate response - FIXED: Use .get() for safe access
        response_validation = guardrails.validate_response({
            "answer": result.get("answer", ""),
            "citations": result.get("citations", []),
            "citation_validation": result.get("citation_validation", {}),
            "retrieval_info": {"num_chunks_retrieved": len(retrieved_chunks)}
        })
        
        end_time = time.time()
        total_time = (end_time - start_time) * 1000
        
        # Calculate confidence score - FIXED: Use .get() for safe access
        confidence_score = 0.8
        if result.get("citation_validation", {}).get("valid_citations"):
            confidence_score = 0.9
        if len(retrieved_chunks) >= 3:
            confidence_score = min(confidence_score + 0.05, 0.95)
        
        # Map validation reason to SafetyFlag if needed
        safety_flags = []
        if not response_validation["valid"]:
            safety_flag = map_reason_to_safety_flag(response_validation["reason"])
            safety_flags = [safety_flag]
        
        # FIXED: Use .get() for safe access and max() with default
        return ChatResponse(
            answer=result.get("answer", "Error generating response"),
            sources=sources,
            safety=SafetyReport(
                is_safe=response_validation["valid"],
                flags=safety_flags,
                confidence=0.95 if response_validation["valid"] else 0.7,
                reasoning=response_validation["message"]
            ),
            metrics=LatencyAndScores(
                retrieval_time_ms=retrieval_time,
                generation_time_ms=generation_time,
                total_time_ms=total_time,
                retrieval_score=max((chunk.get("similarity_score", 0.0) for chunk in retrieved_chunks), default=0.0) if retrieved_chunks else 0.0,
                answer_relevance_score=confidence_score
            ),
            confidence_score=confidence_score,
            legal_jurisdiction="UK"
        )
        
    except HTTPException:
        # Re-raise HTTP exceptions (they're already properly formatted)
        raise
    except Exception as e:
        logger.error(f"Chat service error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Chat service error: {str(e)}")


# CRITICAL FIX: Pre-initialization removed to prevent PyTorch segfaults during module import
# Services will be initialized lazily on first request via get_*_service() functions
# This prevents segfaults when PyTorch is broken
logger.info("🔄 Services will be initialized on first request (lazy loading to prevent segfaults)")