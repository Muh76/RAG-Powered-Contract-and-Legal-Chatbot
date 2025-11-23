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
from app.core.config import settings
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
    """Get RAG service - initialize with proper error handling"""
    global rag_service
    
    if rag_service is None:
        logger.info("🔄 Initializing RAGService...")
        try:
            # CRITICAL: Try to initialize RAG service
            # If it segfaults, we can't catch it in Python, but at least we try
            rag_service = RAGService()
            
            # Verify RAG service is actually usable
            if rag_service.faiss_index is None and rag_service.embedding_gen is None:
                logger.warning("⚠️ RAG service initialized but has no FAISS index or embeddings")
                # This is degraded but not crashed
            else:
                logger.info("✅ RAGService initialized successfully")
                
        except SystemExit:
            # This might catch segfaults in some cases
            logger.error("❌ RAG service initialization caused process exit (possible segfault)")
            logger.warning("⚠️ Creating degraded RAG service")
            rag_service = _create_degraded_rag_service()
        except Exception as e:
            logger.error(f"❌ RAG initialization failed with exception: {e}")
            logger.warning("⚠️ Creating degraded RAG service (will use TF-IDF fallback)")
            rag_service = _create_degraded_rag_service()
        except:
            # Catch anything else (including segfaults that become exceptions)
            logger.error("❌ RAG initialization failed with unknown error")
            logger.warning("⚠️ Creating degraded RAG service")
            rag_service = _create_degraded_rag_service()
    
    return rag_service


def _create_degraded_rag_service():
    """Create a degraded RAG service that won't crash the server"""
    from app.services.rag_service import RAGService as RAGServiceClass
    rag_service = RAGServiceClass.__new__(RAGServiceClass)
    rag_service.embedding_gen = None
    rag_service.faiss_index = None
    rag_service.chunk_metadata = []
    rag_service.use_hybrid = False
    rag_service.bm25_retriever = None
    rag_service.semantic_retriever = None
    rag_service.hybrid_retriever = None
    rag_service.explainability_analyzer = None
    
    # Try to load FAISS index without embeddings (safer)
    try:
        from pathlib import Path
        import pickle
        import faiss
        
        # Try multiple paths
        faiss_paths = [
            Path("data/faiss_index.bin"),
            Path("data/indices/faiss_index.pkl"),
            Path("notebooks/phase1/data/faiss_index.bin")
        ]
        
        for faiss_path in faiss_paths:
            if not faiss_path.exists():
                continue
                
            try:
                if faiss_path.suffix == '.pkl':
                    # Combined file
                    with open(faiss_path, 'rb') as f:
                        data = pickle.load(f)
                        rag_service.faiss_index = data.get('faiss_index')
                        rag_service.chunk_metadata = data.get('chunk_metadata', [])
                else:
                    # Separate files
                    metadata_path = faiss_path.parent / "chunk_metadata.pkl"
                    if metadata_path.exists():
                        rag_service.faiss_index = faiss.read_index(str(faiss_path))
                        with open(metadata_path, 'rb') as f:
                            rag_service.chunk_metadata = pickle.load(f)
                
                if rag_service.faiss_index:
                    logger.info(f"✅ Loaded FAISS index with {len(rag_service.chunk_metadata)} chunks (no embeddings)")
                    break
            except Exception as e:
                logger.warning(f"Could not load FAISS from {faiss_path}: {e}")
                continue
                
    except Exception as load_error:
        logger.warning(f"Could not load FAISS index: {load_error}")
    
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
        # CRITICAL FIX: Check app state first to avoid segfault during request
        from fastapi import Request
        try:
            # Get RAG service from app state (pre-initialized at startup)
            # This prevents segfaults during user requests
            rag = None
            # Try to get from current request's app state
            # For now, use lazy initialization but with better error handling
            try:
                rag = get_rag_service()
                # Check if RAG service is in degraded mode
                if rag is None or (hasattr(rag, 'embedding_gen') and rag.embedding_gen is None and hasattr(rag, 'faiss_index') and rag.faiss_index is None):
                    logger.error("RAG service is unavailable - in degraded mode")
                    raise HTTPException(
                        status_code=503,
                        detail="RAG service is currently unavailable due to initialization errors. The knowledge base may not be loaded properly. Please contact support or try again later."
                    )
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"RAG service error: {e}", exc_info=True)
                # If it's a segfault, we won't get here, but if it's another error, handle it
                raise HTTPException(status_code=503, detail=f"RAG service error: {str(e)}. Please try again later.")
        except HTTPException:
            raise
        
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
            
            # Use hybrid search if available (BM25 + TF-IDF, NO PyTorch!)
            # Hybrid search now works with BM25 + TF-IDF instead of BM25 + Semantic
            use_hybrid_search = rag.hybrid_retriever is not None
            if use_hybrid_search:
                logger.info("Using hybrid search (BM25 + TF-IDF, NO PyTorch!)")
            else:
                logger.info("Using TF-IDF search")
            
            def search_func():
                # Use hybrid search if available (BM25 + TF-IDF, no PyTorch)
                return rag.search(
                    query=request.query,
                    top_k=(request.top_k or 5) * 2,  # Retrieve more initially for filtering
                    use_hybrid=use_hybrid_search,  # Enable hybrid if available
                    user_id=current_user.id if include_private_corpus else None,
                    include_private_corpus=include_private_corpus and private_corpus_results is not None,
                    private_corpus_results=private_corpus_results
                )
            
            retrieval_result = await loop.run_in_executor(
                _executor,
                search_func
            )
            
            retrieval_time_ms = (time.time() - retrieval_start) * 1000
            
            # CRITICAL: Post-retrieval filtering to remove irrelevant Acts
            # Extract key terms from query to filter relevant Acts
            query_lower = request.query.lower()
            act_keywords = {
                "employment": ["Employment Rights Act", "Employment", "employment_rights", "employment_rights_act"],
                "equality": ["Equality Act", "Equality", "equality_act"],
                "sale": ["Sale of Goods Act", "Sale of Goods", "sale_of_goods", "sale_of_goods_act"],
                "data protection": ["Data Protection Act", "Data Protection", "data_protection"],
                "wages": ["Employment Rights Act", "Employment", "employment_rights"],
                "discrimination": ["Equality Act", "Equality", "equality_act"],
                "goods": ["Sale of Goods Act", "Sale of Goods", "sale_of_goods"]
            }
            
            # Find matching Acts from query
            matching_acts = []
            for keyword, acts in act_keywords.items():
                if keyword in query_lower:
                    matching_acts.extend(acts)
                    break  # Only use first match
            
            # Filter results if we have matching Acts and retrieved results
            if matching_acts and retrieval_result:
                filtered_results = []
                for result in retrieval_result:
                    # Check if result matches relevant Act
                    metadata = result.get('metadata', {})
                    title = metadata.get('title', '').lower()
                    source = metadata.get('source', '').lower()
                    chunk_text = result.get('text', '').lower()
                    chunk_id = result.get('chunk_id', '').lower()
                    
                    # Check if result matches any relevant Act keyword
                    is_relevant = any(act.lower() in title or act.lower() in source or act.lower() in chunk_id for act in matching_acts)
                    
                    # Also check if result has high similarity (keep high-similarity results even if Act doesn't match)
                    similarity = result.get('similarity_score', 0.0)
                    
                    # Keep if relevant OR has high similarity (>= 0.6) OR matches query keywords in text
                    query_words = set(query_lower.split())
                    chunk_words = set(chunk_text.split())
                    keyword_overlap = len(query_words.intersection(chunk_words))
                    
                    if is_relevant or similarity >= 0.6 or keyword_overlap >= 3:
                        filtered_results.append(result)
                    else:
                        logger.debug(f"Filtered out irrelevant result: {metadata.get('title', 'Unknown')} (similarity: {similarity:.3f})")
                
                # If filtering removed too many results, keep top-k by similarity regardless of Act
                if len(filtered_results) < request.top_k and len(filtered_results) > 0:
                    retrieval_result = filtered_results[:request.top_k]
                    logger.info(f"Filtered to {len(retrieval_result)} relevant results")
                elif len(filtered_results) >= request.top_k:
                    # Sort filtered results by similarity and take top-k
                    retrieval_result = sorted(filtered_results, key=lambda x: x.get('similarity_score', 0.0), reverse=True)[:request.top_k]
                    logger.info(f"Filtered to {len(retrieval_result)} relevant results from {len(filtered_results)} matches")
                elif not filtered_results and retrieval_result:
                    # All filtered out - keep top-k by similarity as fallback
                    retrieval_result = sorted(retrieval_result, key=lambda x: x.get('similarity_score', 0.0), reverse=True)[:request.top_k]
                    logger.warning(f"All results filtered out, keeping top {request.top_k} by similarity")
            
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
            
            # Extract chunks and metadata with safe None checks
            chunks = [r.get("text", "") for r in retrieval_result if isinstance(r, dict)]
            chunk_metadata = [r.get("metadata", {}) for r in retrieval_result if isinstance(r, dict)]
            
            # Build context from retrieved chunks
            context = "\n\n".join(chunks)
            
            # Format sources with safe None checks
            sources = []
            for i, result in enumerate(retrieval_result):
                if not isinstance(result, dict):
                    continue
                    
                metadata = result.get("metadata", {}) or {}
                if not isinstance(metadata, dict):
                    metadata = {}
                    
                # Determine if source is from private or public corpus
                corpus = metadata.get("corpus", "public") if isinstance(metadata, dict) else "public"
                source_name = "Private Document" if corpus == "private" else (metadata.get("source", "Public Corpus") if isinstance(metadata, dict) else "Public Corpus")
                
                # Safely extract text snippet with None check
                text_content = result.get("text", "")
                if not isinstance(text_content, str):
                    text_content = ""
                text_snippet = text_content[:200] if text_content else ""
                
                sources.append(Source(
                    chunk_id=result.get("chunk_id", f"chunk_{i}"),
                    title=(metadata.get("title") or result.get("title") or "Unknown") if isinstance(metadata, dict) else "Unknown",
                    url=metadata.get("url") if isinstance(metadata, dict) else None,  # Optional field
                    text_snippet=text_snippet,
                    similarity_score=float(result.get("similarity_score", 0.0)),
                    metadata={
                        "corpus": corpus,
                        "section": metadata.get("section", "Unknown") if isinstance(metadata, dict) else "Unknown",
                        "source": source_name,
                        "jurisdiction": metadata.get("jurisdiction", "UK") if isinstance(metadata, dict) else "UK",
                        **(metadata if isinstance(metadata, dict) else {})  # Include all other metadata
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
            try:
                user_role_value = current_user.role.value if hasattr(current_user.role, 'value') else str(current_user.role)
                user_mode = "solicitor" if user_role_value in ["admin", "solicitor"] else "public"
                
                # Safely get request mode value
                request_mode_value = None
                if request.mode:
                    if hasattr(request.mode, 'value'):
                        request_mode_value = request.mode.value
                    else:
                        request_mode_value = str(request.mode)
                
                response_mode = request_mode_value if request_mode_value and request_mode_value in ["solicitor", "public"] else user_mode
                
                logger.info(f"Using response mode: {response_mode} (user role: {user_role_value}, request mode: {request_mode_value or 'not provided'})")
            except Exception as mode_error:
                logger.warning(f"Error determining response mode: {mode_error}, defaulting to public")
                response_mode = "public"
            
            # CRITICAL: Check retrieval quality BEFORE generation - reject weak matches
            avg_similarity = 0.0
            min_similarity = 0.0
            
            if retrieval_result and isinstance(retrieval_result, list) and len(retrieval_result) > 0:
                similarity_scores = [r.get("similarity_score", 0.0) for r in retrieval_result if isinstance(r, dict) and r.get("similarity_score") is not None]
                if similarity_scores and len(similarity_scores) > 0:
                    try:
                        avg_similarity = sum(similarity_scores) / len(similarity_scores)
                        min_similarity = min(similarity_scores)
                    except (ZeroDivisionError, ValueError, TypeError) as e:
                        logger.warning(f"Error calculating similarity scores: {e}")
                        avg_similarity = 0.0
                        min_similarity = 0.0
                
                # Reject if average similarity is too low - prevents hallucination
                # Note: Threshold adjusted for TF-IDF (less accurate than semantic embeddings)
                # TF-IDF typically produces lower similarity scores than dense embeddings
                similarity_threshold = 0.4  # Lower threshold for TF-IDF embeddings
                if avg_similarity > 0 and avg_similarity < similarity_threshold:
                        generation_time_ms = (time.time() - generation_start) * 1000
                        logger.warning(f"Rejecting query due to weak retrieval similarity: {avg_similarity:.3f}")
                        return ChatResponse(
                            answer=f"I cannot provide a confident answer because the retrieved sources have weak relevance (similarity: {avg_similarity:.3f}). The available sources may not contain sufficient information to answer this question accurately. Please try rephrasing your question or be more specific about the legal topic.",
                            sources=sources,
                            safety=SafetyReport(
                                is_safe=True,
                                flags=[],
                                confidence=0.8,
                                reasoning=f"Grounding validation: Weak similarity ({avg_similarity:.3f}) - insufficient sources for confident answer"
                            ),
                            metrics=LatencyAndScores(
                                retrieval_time_ms=retrieval_time_ms,
                                generation_time_ms=generation_time_ms,
                                total_time_ms=(time.time() - start_time) * 1000,
                                retrieval_score=avg_similarity,
                                answer_relevance_score=0.0
                            ),
                            confidence_score=avg_similarity,
                            legal_jurisdiction="UK",
                            model_used=settings.OPENAI_MODEL,
                            response_mode=response_mode
                        )
            
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
            # Safely extract LLM result with None checks
            answer = llm_result.get("answer", "") if llm_result and isinstance(llm_result, dict) else ""
            citation_validation = llm_result.get("citation_validation", {}) if llm_result and isinstance(llm_result, dict) else {}
            
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
            if guardrails_result and not guardrails_result.get("all_passed", False):
                failures = guardrails_result.get("failures", []) or []
                failure_messages = [f.get("message", "") for f in failures if isinstance(f, dict)]
                
                # If citations are missing, this is critical - reject the answer
                citation_failure = next((f for f in failures if isinstance(f, dict) and f.get("rule") == "citation_enforcement"), None)
                if citation_failure and isinstance(citation_failure, dict):
                    citation_msg = citation_failure.get('message', 'Missing citations')
                    logger.warning(f"Answer rejected due to missing citations: {citation_msg}")
                    return ChatResponse(
                        answer=f"I cannot provide this answer because it lacks proper citations to legal sources. {citation_msg}. Please regenerate with citations in [1], [2] format.",
                        sources=sources,
                        safety=SafetyReport(
                            is_safe=False,
                            flags=[map_reason_to_safety_flag("missing_citations")],
                            confidence=0.95,
                            reasoning=f"Guardrails failed: {citation_failure.get('message', 'Missing citations') if isinstance(citation_failure, dict) else 'Missing citations'}"
                        ),
                        metrics=LatencyAndScores(
                            retrieval_time_ms=retrieval_time_ms,
                            generation_time_ms=generation_time_ms,
                            total_time_ms=(time.time() - start_time) * 1000,
                            retrieval_score=0.0,
                            answer_relevance_score=0.0
                        ),
                        confidence_score=0.0,
                        legal_jurisdiction="UK",
                        model_used=settings.OPENAI_MODEL,
                        response_mode=response_mode
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
        confidence_score = 0.0
        if retrieval_result and isinstance(retrieval_result, list) and len(retrieval_result) > 0:
            try:
                scores = [r.get("similarity_score", 0.0) for r in retrieval_result if isinstance(r, dict)]
                if scores:
                    confidence_score = min(1.0, max(0.0, sum(scores) / len(scores)))
            except Exception as e:
                logger.warning(f"Error calculating confidence score: {e}")
                confidence_score = 0.0
        
        # Citation count for metrics
        citation_count = 0
        if citation_validation and isinstance(citation_validation, dict):
            citation_count = citation_validation.get("valid_count", citation_validation.get("count", 0))
        
        answer_relevance_score = 0.8  # Could be improved with LLM evaluation
        
        # Log request with user info
        try:
            user_role_str = current_user.role.value if hasattr(current_user.role, 'value') else str(current_user.role)
            logger.info(
                f"Chat request processed: user_id={current_user.id}, "
                f"user_email={current_user.email}, role={user_role_str}, "
                f"query_length={len(request.query)}, "
                f"retrieval_time={retrieval_time_ms:.1f}ms, "
                f"generation_time={generation_time_ms:.1f}ms"
            )
        except Exception as log_error:
            logger.warning(f"Error logging request: {log_error}")
        
        # Determine safety status
        safety_flags = []
        safety_confidence = 1.0
        safety_reasoning = "All safety checks passed"
        
        if guardrails_result and isinstance(guardrails_result, dict):
            if not guardrails_result.get("all_passed", False):
                safety_confidence = 0.7
                safety_reasoning = "Some guardrail warnings present"
                warnings = guardrails_result.get("warnings", []) or []
                if warnings and isinstance(warnings, list):
                    warning_messages = [w.get("message", "") for w in warnings if isinstance(w, dict)]
                    if warning_messages:
                        safety_reasoning += ": " + "; ".join(warning_messages)
        
        # Build response with metadata in reasoning field (for now)
        # Include guardrails info in safety reasoning
        if guardrails_applied and guardrails_result and isinstance(guardrails_result, dict):
            if guardrails_result.get("all_passed", False):
                rules_applied = guardrails_result.get('rules_applied', []) or []
                rules_str = ', '.join([str(r) for r in rules_applied]) if rules_applied else 'none'
                enhanced_reasoning = safety_reasoning + f" | Guardrails: {rules_str}"
            else:
                failures = guardrails_result.get("failures", []) or []
                failure_msgs = [f.get("rule", "") for f in failures if isinstance(f, dict)]
                failure_str = ', '.join([f for f in failure_msgs if f]) if failure_msgs else 'unknown'
                enhanced_reasoning = safety_reasoning + f" | Guardrails failed: {failure_str}"
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
            legal_jurisdiction="UK",
            model_used=llm_result.get("model_used", settings.OPENAI_MODEL) if llm_result and isinstance(llm_result, dict) else settings.OPENAI_MODEL,
            response_mode=llm_result.get("mode", response_mode) if llm_result and isinstance(llm_result, dict) else response_mode,
            citation_validation=citation_validation,
            guardrails_applied=guardrails_applied,
            guardrails_result=guardrails_result
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
