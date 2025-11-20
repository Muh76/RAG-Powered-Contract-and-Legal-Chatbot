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
from app.documents.service import DocumentService
from app.core.database import get_db
from sqlalchemy.orm import Session
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
    if "citation" in reason_lower or "missing_citations" in reason_lower:
        return SafetyFlag.NON_LEGAL  # Missing citations is a form of non-compliance
    elif "harmful" in reason_lower:
        return SafetyFlag.HARMFUL
    elif "injection" in reason_lower or "prompt" in reason_lower:
        return SafetyFlag.PROMPT_INJECTION
    elif "domain" in reason_lower or "non_legal" in reason_lower or "insufficient_legal" in reason_lower:
        return SafetyFlag.NON_LEGAL
    elif "pii" in reason_lower or "personal" in reason_lower:
        return SafetyFlag.PII_DETECTED
    else:
        # Default to NON_LEGAL for unknown reasons
        return SafetyFlag.NON_LEGAL

@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
    include_private_corpus: bool = True
):
    """
    Chat endpoint with RAG pipeline (requires authentication).
    
    Args:
        request: Chat request with query
        current_user: Current authenticated user
        db: Database session
        include_private_corpus: Whether to include user's private documents in search (default: True)
    """
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
        
        # Search user's private documents if enabled
        private_corpus_results = None
        if include_private_corpus:
            try:
                doc_service = DocumentService()
                private_corpus_results = doc_service.search_user_documents(
                    db=db,
                    user_id=current_user.id,
                    query=request.query,
                    top_k=request.top_k or 5,
                    similarity_threshold=0.7
                )
                logger.info(f"Found {len(private_corpus_results)} results from private corpus for user {current_user.id}")
            except Exception as e:
                logger.warning(f"Error searching private corpus: {e}")
                private_corpus_results = []
        
        # CRITICAL FIX: Run blocking PyTorch/FAISS operations in thread pool to prevent segfault
        retrieval_start = time.time()
        try:
            # Run retrieval in executor to avoid blocking event loop and segfaults
            loop = asyncio.get_event_loop()
            
            # Prepare search function with private corpus if available
            def search_func():
                return rag.search(
                    query=request.query,
                    top_k=request.top_k or 5,
                    user_id=current_user.id if include_private_corpus else None,
                    include_private_corpus=include_private_corpus and private_corpus_results is not None,
                    private_corpus_results=private_corpus_results
                )
            
            retrieval_result = await loop.run_in_executor(
                _executor,
                search_func
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
                # Determine if source is from private or public corpus
                corpus = metadata.get("corpus", "public")
                source_name = "Private Document" if corpus == "private" else metadata.get("source", "Public Corpus")
                
                sources.append(Source(
                    chunk_id=result.get("chunk_id", f"chunk_{i}"),
                    title=metadata.get("title", "Unknown") or result.get("title", "Unknown"),
                    url=metadata.get("url"),  # Optional field
                    text_snippet=result.get("text", "")[:200],  # Fixed: was "text", should be "text_snippet"
                    similarity_score=result.get("similarity_score", 0.0),
                    metadata={
                        "corpus": corpus,
                        "section": metadata.get("section", "Unknown"),
                        "source": source_name,
                        "jurisdiction": metadata.get("jurisdiction", "UK"),
                        **metadata  # Include all other metadata
                    }
                ))
            
        except Exception as e:
            logger.error(f"Retrieval error: {e}", exc_info=True)
            retrieval_time_ms = (time.time() - retrieval_start) * 1000
            raise HTTPException(
                status_code=500,
                detail=f"Retrieval error: {str(e)}"
            )
        
        # 3. Generate answer using LLM with role-based mode
        generation_start = time.time()
        try:
            llm = get_llm_service()
            
            # Determine mode based on user role and request mode
            # If user is admin or solicitor, use solicitor mode; otherwise use public mode
            # Request mode can override if explicitly provided
            user_mode = "solicitor" if current_user.role.value in ["admin", "solicitor"] else "public"
            response_mode = request.mode.value if request.mode and request.mode.value in ["solicitor", "public"] else user_mode
            
            logger.info(f"Using response mode: {response_mode} (user role: {current_user.role.value}, request mode: {request.mode.value if request.mode else 'not provided'})")
            
            # Use generate_legal_answer for proper mode-based responses with citations
            loop = asyncio.get_event_loop()
            llm_result = await loop.run_in_executor(
                _executor,
                lambda: llm.generate_legal_answer(
                    query=request.query,
                    retrieved_chunks=retrieval_result,
                    mode=response_mode
                )
            )
            
            generation_time_ms = (time.time() - generation_start) * 1000
            
            # Extract answer from LLM result
            answer = llm_result.get("answer", "")
            citation_validation = llm_result.get("citation_validation", {})
            
            if not answer:
                answer = "I couldn't generate a response. Please try again."
            
        except Exception as e:
            logger.error(f"LLM generation error: {e}", exc_info=True)
            generation_time_ms = (time.time() - generation_start) * 1000
            answer = "I encountered an error while generating a response. Please try again."
            citation_validation = {"has_citations": False, "error": str(e)}
        
        # 4. Apply comprehensive guardrails
        guardrails_applied = False
        guardrails_result = None
        try:
            guardrails = get_guardrails_service()
            guardrails_result = guardrails.apply_all_rules(
                query=request.query,
                answer=answer,
                retrieved_chunks=retrieval_result,
                citation_validation=citation_validation
            )
            guardrails_applied = True
            
            # If guardrails failed, reject the answer
            if not guardrails_result.get("all_passed", False):
                failures = guardrails_result.get("failures", [])
                failure_messages = [f["message"] for f in failures]
                
                # If citations are missing, this is critical - reject the answer
                citation_failure = next((f for f in failures if f["rule"] == "citation_enforcement"), None)
                if citation_failure:
                    logger.warning(f"Answer rejected due to missing citations: {citation_failure['message']}")
                    return ChatResponse(
                        answer=f"I cannot provide this answer because it lacks proper citations to legal sources. {citation_failure['message']}. Please regenerate with citations in [1], [2] format.",
                        sources=sources,
                        safety=SafetyReport(
                            is_safe=False,
                            flags=[map_reason_to_safety_flag("missing_citations")],
                            confidence=0.95,
                            reasoning=f"Guardrails failed: {citation_failure['message']}"
                        ),
                        metrics=LatencyAndScores(
                            retrieval_time_ms=retrieval_time_ms,
                            generation_time_ms=generation_time_ms,
                            total_time_ms=(time.time() - start_time) * 1000,
                            retrieval_score=0.0,
                            answer_relevance_score=0.0
                        ),
                        confidence_score=0.0,
                        legal_jurisdiction="UK"
                    )
                
                # Other failures - log but allow with warning
                logger.warning(f"Guardrails found issues: {failure_messages}")
                # Add warnings to response
                warnings_text = "\n\n⚠️ Warnings: " + "; ".join(failure_messages)
                answer += warnings_text
                
        except Exception as e:
            logger.error(f"Guardrails error: {e}", exc_info=True)
            # Continue but log the error
        
        # 5. Calculate confidence and relevance scores
        confidence_score = min(1.0, max(0.0, (
            sum(r.get("similarity_score", 0.0) for r in retrieval_result) / len(retrieval_result) if retrieval_result else 0.0
        )))
        
        # Citation count for metrics
        citation_count = citation_validation.get("valid_count", citation_validation.get("count", 0)) if citation_validation else 0
        
        answer_relevance_score = 0.8  # Could be improved with LLM evaluation
        
        # Log request with user info
        logger.info(
            f"Chat request processed: user_id={current_user.id}, "
            f"user_email={current_user.email}, role={current_user.role.value}, "
            f"query_length={len(request.query)}, "
            f"retrieval_time={retrieval_time_ms:.1f}ms, "
            f"generation_time={generation_time_ms:.1f}ms"
        )
        
        # Determine safety status
        safety_flags = []
        safety_confidence = 1.0
        safety_reasoning = "All safety checks passed"
        
        if guardrails_result:
            if not guardrails_result.get("all_passed", False):
                safety_confidence = 0.7
                safety_reasoning = "Some guardrail warnings present"
                warnings = guardrails_result.get("warnings", [])
                if warnings:
                    safety_reasoning += ": " + "; ".join([w["message"] for w in warnings])
        
        # Build response with metadata in reasoning field (for now)
        # Include guardrails info in safety reasoning
        if guardrails_applied and guardrails_result:
            if guardrails_result.get("all_passed", False):
                enhanced_reasoning = safety_reasoning + f" | Guardrails: {', '.join(guardrails_result.get('rules_applied', []))}"
            else:
                failures = guardrails_result.get("failures", [])
                failure_msgs = [f["rule"] for f in failures]
                enhanced_reasoning = safety_reasoning + f" | Guardrails failed: {', '.join(failure_msgs)}"
        else:
            enhanced_reasoning = safety_reasoning + " | Guardrails: Not applied"
        
        response_obj = ChatResponse(
            answer=answer,
            sources=sources,
            safety=SafetyReport(
                is_safe=True,
                flags=safety_flags,
                confidence=safety_confidence,
                reasoning=enhanced_reasoning
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
        
        # Store metadata in response object for frontend access
        # We'll add these as response headers or in a custom way
        # For now, include key info in sources or add to response after creation
        # FastAPI will serialize ChatResponse, so we add metadata via response_model_exclude_unset=False
        # Actually, let's just return the response and handle metadata extraction in frontend
        
        return response_obj
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )
