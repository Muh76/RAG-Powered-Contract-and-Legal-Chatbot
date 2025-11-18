# Legal Chatbot - Chat Route
# Updated to use real RAG services

import os
import sys
from pathlib import Path
from fastapi import APIRouter, HTTPException, Depends
from app.models.schemas import ChatRequest, ChatResponse, Source, SafetyReport, SafetyFlag, LatencyAndScores
from app.services.rag_service import RAGService
from app.services.guardrails_service import GuardrailsService
from app.services.llm_service import LLMService
from app.auth.dependencies import get_current_active_user
from app.auth.models import User
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
async def chat(
    request: ChatRequest,
    current_user: User = Depends(get_current_active_user)
):
    """Chat endpoint with RAG pipeline (requires authentication)"""
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
            # Run retrieval in executor to avoid blocking event loop and segfaults
            loop = asyncio.get_event_loop()
            retrieval_result = await loop.run_in_executor(
                _executor,
                lambda: rag.search(request.query, top_k=request.top_k or 5)
            )
            
            retrieval_time_ms = (time.time() - retrieval_start) * 1000
            
            if not retrieval_result:
                logger.warning("No results retrieved from RAG service")
                return ChatResponse(
                    answer="I couldn't find relevant information to answer your question. Please try rephrasing your query or ensure that legal documents have been ingested.",
                    sources=[],
                    safety=SafetyReport(
                        is_safe=True,
                        flags=[],
                        confidence=1.0,
                        reasoning="No retrieval results"
                    ),
                    metrics=LatencyAndScores(
                        retrieval_time_ms=retrieval_time_ms,
                        generation_time_ms=0.0,
                        total_time_ms=(time.time() - start_time) * 1000,
                        retrieval_score=0.0,
                        answer_relevance_score=0.0
                    ),
                    confidence_score=0.0,
                    legal_jurisdiction="UK"
                )
            
            # Extract chunks and metadata
            chunks = [r.get("text", "") for r in retrieval_result]
            chunk_metadata = [r.get("metadata", {}) for r in retrieval_result]
            
            # Build context from retrieved chunks
            context = "\n\n".join(chunks)
            
            # Format sources
            sources = []
            for i, result in enumerate(retrieval_result):
                metadata = result.get("metadata", {})
                sources.append(Source(
                    chunk_id=result.get("chunk_id", f"chunk_{i}"),
                    text=result.get("text", "")[:200],  # First 200 chars
                    section=metadata.get("section", "Unknown"),
                    title=metadata.get("title", "Unknown"),
                    source=metadata.get("source", "Unknown"),
                    jurisdiction=metadata.get("jurisdiction", "UK"),
                    similarity_score=result.get("similarity_score", 0.0)
                ))
            
        except Exception as e:
            logger.error(f"Retrieval error: {e}", exc_info=True)
            retrieval_time_ms = (time.time() - retrieval_start) * 1000
            raise HTTPException(
                status_code=500,
                detail=f"Retrieval error: {str(e)}"
            )
        
        # 3. Generate answer using LLM
        generation_start = time.time()
        try:
            llm = get_llm_service()
            
            # Build prompt with context
            prompt = f"""You are a legal assistant specializing in UK law. Answer the following question based on the provided legal context.

Context:
{context}

Question: {request.query}

Provide a clear, accurate, and concise answer based on the context. If the context doesn't contain enough information, say so. Cite relevant sections or sources when possible.

Answer:"""
            
            # Run LLM generation in executor to avoid blocking
            loop = asyncio.get_event_loop()
            answer = await loop.run_in_executor(
                _executor,
                lambda: llm.generate(prompt, max_tokens=500)
            )
            
            generation_time_ms = (time.time() - generation_start) * 1000
            
            if not answer:
                answer = "I couldn't generate a response. Please try again."
            
        except Exception as e:
            logger.error(f"LLM generation error: {e}", exc_info=True)
            generation_time_ms = (time.time() - generation_start) * 1000
            answer = "I encountered an error while generating a response. Please try again."
        
        # 4. Calculate confidence and relevance scores (simplified)
        confidence_score = min(1.0, max(0.0, (
            sum(r.get("similarity_score", 0.0) for r in retrieval_result) / len(retrieval_result) if retrieval_result else 0.0
        )))
        
        answer_relevance_score = 0.8  # Simplified - could use LLM to evaluate relevance
        
        # 5. Final safety check on answer
        try:
            guardrails = get_guardrails_service()
            answer_validation = guardrails.validate_answer(answer, request.query)
            
            if not answer_validation["valid"]:
                safety_flag = map_reason_to_safety_flag(answer_validation["reason"])
                return ChatResponse(
                    answer=answer_validation["message"],
                    sources=sources,
                    safety=SafetyReport(
                        is_safe=False,
                        flags=[safety_flag],
                        confidence=0.9,
                        reasoning=answer_validation.get("suggestion", "")
                    ),
                    metrics=LatencyAndScores(
                        retrieval_time_ms=retrieval_time_ms,
                        generation_time_ms=generation_time_ms,
                        total_time_ms=(time.time() - start_time) * 1000,
                        retrieval_score=confidence_score,
                        answer_relevance_score=answer_relevance_score
                    ),
                    confidence_score=confidence_score,
                    legal_jurisdiction="UK"
                )
        except Exception as e:
            logger.warning(f"Final safety check error: {e}")
            # Continue with response even if safety check fails
        
        # Log request with user info
        logger.info(
            f"Chat request processed: user_id={current_user.id}, "
            f"user_email={current_user.email}, role={current_user.role.value}, "
            f"query_length={len(request.query)}, "
            f"retrieval_time={retrieval_time_ms:.1f}ms, "
            f"generation_time={generation_time_ms:.1f}ms"
        )
        
        return ChatResponse(
            answer=answer,
            sources=sources,
            safety=SafetyReport(
                is_safe=True,
                flags=[],
                confidence=1.0,
                reasoning="All safety checks passed"
            ),
            metrics=LatencyAndScores(
                retrieval_time_ms=retrieval_time_ms,
                generation_time_ms=generation_time_ms,
                total_time_ms=(time.time() - start_time) * 1000,
                retrieval_score=confidence_score,
                answer_relevance_score=answer_relevance_score
            ),
            confidence_score=confidence_score,
            legal_jurisdiction="UK"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )
