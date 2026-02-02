# Legal Chatbot - RAG Service
# Phase 2: Updated with Hybrid Search Support

import os
import sys
import faiss
import pickle
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# PERMANENTLY REMOVED: PyTorch-dependent imports to prevent segfaults
# from retrieval.embeddings.embedding_generator import EmbeddingGenerator, EmbeddingConfig
# from retrieval.semantic_retriever import SemanticRetriever
# from retrieval.hybrid_retriever import AdvancedHybridRetriever, FusionStrategy
# from retrieval.rerankers.cross_encoder_reranker import CrossEncoderReranker
# from retrieval.explainability import ExplainabilityAnalyzer

# System now uses TF-IDF only (no PyTorch dependencies)
from retrieval.bm25_retriever import BM25Retriever
from retrieval.metadata_filter import MetadataFilter
from app.core.config import settings

logger = logging.getLogger(__name__)


class RAGService:
    """RAG service for legal chatbot with hybrid search support"""
    
    def __init__(self, use_hybrid: bool = False):
        """
        Initialize RAG service.
        
        Args:
            use_hybrid: If True, initialize hybrid retriever; if False, use basic semantic search (backward compatible)
        """
        self.embedding_gen = None
        self.faiss_index = None
        self.chunk_metadata = []
        
        # Hybrid search components
        self.use_hybrid = use_hybrid
        self.bm25_retriever = None
        self.semantic_retriever = None
        self.hybrid_retriever = None
        
        # Explainability component - Will be initialized lazily when needed
        self.explainability_analyzer = None
        
        self._initialize()
    
    def _initialize(self):
        """Initialize RAG components with robust error handling to prevent segfaults"""
        # CRITICAL FIX: Initialize components separately to isolate failures
        # This prevents one component failure from crashing the entire service
        
        # Step 1: Load FAISS index first (safer, doesn't require PyTorch)
        logger.info("Loading FAISS index and metadata...")
        try:
            self._load_vector_store()
        except RuntimeError:
            raise  # Fail-fast: dimension mismatch, metadata mismatch - do not degrade
        except Exception as e:
            logger.error("FAISS load failed: %s", e)
            raise  # Do not allow degraded state; crash startup on load failure
        if self.faiss_index is not None:
            logger.info("FAISS loaded: %s chunks", len(self.chunk_metadata))
        
        # Step 2: Initialize embedding generator (for semantic search in hybrid mode)
        # When DISABLE_LOCAL_EMBEDDINGS=True and OpenAI API key present: use OpenAI + FAISS for semantic retrieval
        try:
            disable_embeddings = os.getenv("DISABLE_EMBEDDINGS", "").lower() in ("1", "true", "yes")
            use_openai_for_semantic = (
                getattr(settings, "DISABLE_LOCAL_EMBEDDINGS", True) and settings.OPENAI_API_KEY
            )
            if disable_embeddings and not use_openai_for_semantic:
                logger.info("⚠️ Embeddings disabled via DISABLE_EMBEDDINGS environment variable")
                logger.info("✅ System will use TF-IDF keyword search only")
                self.embedding_gen = None
            elif use_openai_for_semantic:
                logger.info("Semantic retriever: OpenAI + FAISS (DISABLE_LOCAL_EMBEDDINGS=True, API key present)")
            if not (disable_embeddings and not use_openai_for_semantic):
                # CRITICAL: Initialize embeddings IMMEDIATELY for TRUE hybrid search (BM25 + Embeddings)
                # Use OpenAI embeddings (NO PyTorch - eliminates segfaults completely!)
                logger.info("🚀 Initializing OpenAI embedding generator for BM25 + Embeddings hybrid search...")
                logger.info("✅ Using OpenAI API - NO PyTorch required (eliminates segfaults!)")

                try:
                    from retrieval.embeddings.openai_embedding_generator import OpenAIEmbeddingGenerator, OpenAIEmbeddingConfig

                    # Check if OpenAI API key is available
                    api_key = settings.OPENAI_API_KEY
                    if not api_key:
                        logger.error("❌ OPENAI_API_KEY not configured")
                        logger.warning("⚠️ Set OPENAI_API_KEY environment variable or in config")
                        logger.warning("⚠️ Falling back to TF-IDF (not true hybrid)")
                        self.embedding_gen = None
                    else:
                        logger.info("✅ OpenAI API key found, initializing OpenAI embeddings...")

                        # Initialize OpenAI embedding generator
                        embedding_config = OpenAIEmbeddingConfig(
                            api_key=api_key,
                            model=settings.OPENAI_EMBEDDING_MODEL,
                            dimension=settings.EMBEDDING_DIMENSION
                        )

                        self.embedding_gen = OpenAIEmbeddingGenerator(embedding_config)

                        # Test that it actually works
                        try:
                            test_emb = self.embedding_gen.generate_embedding("test")
                            if test_emb and len(test_emb) > 0:
                                logger.info(f"✅✅✅ OpenAI embedding generator initialized!")
                                logger.info(f"✅ Embedding generation verified: {len(test_emb)} dimensions")
                                logger.info("✅✅✅ TRUE HYBRID: BM25 + SEMANTIC EMBEDDINGS enabled!")
                                logger.info("✅✅✅ NO PYTORCH - NO SEGFAULTS!")
                            else:
                                raise RuntimeError("Embedding generation returned empty result")
                        except Exception as test_error:
                            logger.error(f"❌ OpenAI embedding generation test failed: {test_error}")
                            logger.warning("⚠️ Check OPENAI_API_KEY and API access")
                            logger.warning("⚠️ Falling back to TF-IDF (not true hybrid)")
                            self.embedding_gen = None
                            
                except ImportError:
                    if use_openai_for_semantic:
                        logger.warning("⚠️ OpenAI embedding generator not available; TF-IDF fallback (no PyTorch when DISABLE_LOCAL_EMBEDDINGS=True)")
                        self.embedding_gen = None
                    else:
                        # Fallback to PyTorch if OpenAI generator not available
                        logger.warning("⚠️ OpenAI embedding generator not available, trying PyTorch...")
                        try:
                            from retrieval.embeddings.embedding_generator import EmbeddingGenerator, EmbeddingConfig
                            from retrieval.embeddings.embedding_generator import _check_pytorch_available
                            
                            if not _check_pytorch_available():
                                logger.error("❌ PyTorch is not available")
                                logger.warning("⚠️ Falling back to TF-IDF (not true hybrid)")
                                self.embedding_gen = None
                            else:
                                logger.info("✅ PyTorch check passed, initializing embeddings...")
                                embedding_config = EmbeddingConfig(
                                    model_name=settings.EMBEDDING_MODEL,
                                    dimension=settings.EMBEDDING_DIMENSION
                                )
                                self.embedding_gen = EmbeddingGenerator(embedding_config)
                                
                                if self.embedding_gen.model is not None:
                                    logger.info("✅✅✅ Embedding generator initialized (PyTorch fallback)")
                                    test_emb = self.embedding_gen.generate_embedding("test")
                                    if test_emb and len(test_emb) > 0:
                                        logger.info(f"✅ Embedding verified: {len(test_emb)} dimensions")
                                else:
                                    logger.error("❌ Embedding model is None")
                                    self.embedding_gen = None
                        except Exception as pytorch_error:
                            logger.error(f"❌ PyTorch initialization failed: {pytorch_error}")
                            logger.warning("⚠️ Falling back to TF-IDF (not true hybrid)")
                            self.embedding_gen = None
                        
                except Exception as embed_error:
                    logger.error(f"❌ Failed to initialize OpenAI embeddings: {embed_error}", exc_info=True)
                    logger.warning("⚠️ Check OPENAI_API_KEY configuration")
                    logger.warning("⚠️ Falling back to TF-IDF (not true hybrid)")
                    self.embedding_gen = None
        except Exception as e:
            logger.warning(f"⚠️ Failed to prepare embedding generator: {e}")
            logger.info("✅ System will use TF-IDF keyword search (stable fallback)")
            self.embedding_gen = None
        
        # Step 3: Initialize hybrid retriever (BM25 + OpenAI + FAISS when FAISS exists)
        if self.use_hybrid and self.chunk_metadata:
            try:
                self._initialize_hybrid_retriever()
            except RuntimeError:
                raise  # Fail-fast: FAISS exists but cannot be used
            except Exception as e:
                logger.warning(f"Failed to initialize hybrid retriever: {e}")
                self.hybrid_retriever = None
        
        # Fail-fast: FAISS exists but embeddings unavailable - no TF-IDF fallback
        if self.faiss_index is not None and self.embedding_gen is None:
            raise RuntimeError(
                "FAISS index exists but embeddings are unavailable (OPENAI_API_KEY missing or invalid). "
                "TF-IDF fallback is disabled when FAISS exists. Set OPENAI_API_KEY or remove data/faiss_index.bin to use TF-IDF-only mode."
            )
        
        # Final status check
        if self.embedding_gen is None and self.faiss_index is None:
            logger.warning("⚠️ RAG service initialized in TF-IDF-only mode (no FAISS, no embeddings)")
        elif self.hybrid_retriever is not None:
            logger.info("Hybrid retrieval active: BM25 + OpenAI + FAISS")
        else:
            logger.info("✅ RAG service fully initialized")
    
    def _initialize_hybrid_retriever(self):
        """Initialize hybrid retriever with BM25 + Semantic embeddings (or TF-IDF fallback)"""
        try:
            if not self.chunk_metadata or len(self.chunk_metadata) == 0:
                logger.warning("Cannot initialize hybrid retriever: no chunk metadata available")
                return
            
            try:
                from retrieval.hybrid_retriever import AdvancedHybridRetriever, FusionStrategy
            except ImportError as e:
                logger.warning(f"Could not import hybrid retriever components: {e}")
                self.hybrid_retriever = None
                return
            
            # Extract documents for BM25
            documents = [chunk.get("text", "") for chunk in self.chunk_metadata]
            
            # Initialize BM25 retriever (always available - no PyTorch)
            self.bm25_retriever = BM25Retriever(documents)
            logger.info("✅ BM25 retriever initialized")
            
            # Use embedding generator if available (should be initialized in _initialize)
            embedding_gen = self.embedding_gen
            
            # Initialize semantic retriever if embeddings are available
            semantic_retriever = None
            # Check if we have OpenAI embeddings (which don't need SemanticRetriever with PyTorch)
            is_openai_embeddings = (
                embedding_gen is not None and 
                hasattr(embedding_gen, 'config') and 
                hasattr(embedding_gen.config, 'api_key') and
                self.faiss_index is not None
            )
            
            if embedding_gen is not None:
                # Check if embedding_gen has a model attribute (PyTorch) or is OpenAI-based
                has_model = hasattr(embedding_gen, 'model') and embedding_gen.model is not None
                
                if is_openai_embeddings:
                    # For OpenAI embeddings, use OpenAI-compatible semantic retriever (NO PYTORCH!)
                    logger.info("Semantic retriever: OpenAI + FAISS")
                    logger.info("✅ Using OpenAI embeddings for TRUE HYBRID SEARCH (BM25 + Embeddings)")
                    try:
                        from retrieval.openai_semantic_retriever import OpenAISemanticRetriever
                        semantic_retriever = OpenAISemanticRetriever(
                            embedding_gen=embedding_gen,
                            faiss_index=self.faiss_index,
                            chunk_metadata=self.chunk_metadata
                        )
                        if semantic_retriever.is_ready():
                            logger.info("✅✅✅ TRUE HYBRID: BM25 + OpenAI Embeddings (NO PYTORCH!)")
                        else:
                            logger.warning("OpenAI semantic retriever not ready")
                            semantic_retriever = None
                    except Exception as e:
                        logger.error(f"Failed to initialize OpenAI semantic retriever: {e}", exc_info=True)
                        semantic_retriever = None
                elif has_model:
                    # PyTorch-based embeddings - can use SemanticRetriever
                    try:
                        from retrieval.semantic_retriever import SemanticRetriever
                        semantic_retriever = SemanticRetriever()
                        if semantic_retriever.is_ready():
                            logger.info("✅ Semantic retriever initialized (hybrid: BM25 + Semantic embeddings)")
                            # Store embedding_gen reference for later use
                            self.embedding_gen = embedding_gen
                        else:
                            logger.warning("Semantic retriever not ready")
                            semantic_retriever = None
                    except Exception as e:
                        logger.warning(f"Failed to initialize semantic retriever: {e}")
                        semantic_retriever = None
                else:
                    logger.warning("Embedding generator available but model is None")
                    semantic_retriever = None
            
            # TF-IDF fallback disabled when FAISS exists - raise instead of degrading
            if semantic_retriever is None:
                if self.faiss_index is not None:
                    raise RuntimeError(
                        "FAISS index exists but semantic retriever (OpenAI+FAISS) could not be initialized. "
                        "TF-IDF fallback is disabled when FAISS exists. Fix OPENAI_API_KEY or embedding config."
                    )
                # No FAISS: allow TF-IDF for hybrid (BM25 + TF-IDF)
                logger.warning("⚠️ No FAISS index; using TF-IDF as semantic component (BM25 + TF-IDF)")
                try:
                    from retrieval.tfidf_retriever import TFIDFRetriever
                    tfidf_retriever = TFIDFRetriever(self.chunk_metadata)
                    if tfidf_retriever.is_ready():
                        semantic_retriever = tfidf_retriever
                        logger.info("Retrievers active: BM25 + TF-IDF (no FAISS)")
                    else:
                        logger.error("❌ TF-IDF retriever not ready, hybrid search disabled")
                        self.hybrid_retriever = None
                        return
                except Exception as e:
                    logger.error(f"❌ Failed to initialize TF-IDF retriever: {e}")
                    self.hybrid_retriever = None
                    return
            
            # Disable reranker (optional - requires PyTorch)
            reranker = None
            
            # Create hybrid retriever with BM25 + Semantic (or TF-IDF fallback)
            self.hybrid_retriever = AdvancedHybridRetriever(
                bm25_retriever=self.bm25_retriever,
                semantic_retriever=semantic_retriever,  # Semantic embeddings OR TF-IDF
                chunk_metadata=self.chunk_metadata,
                fusion_strategy=settings.HYBRID_SEARCH_FUSION_STRATEGY,
                bm25_weight=settings.HYBRID_SEARCH_BM25_WEIGHT,
                semantic_weight=settings.HYBRID_SEARCH_SEMANTIC_WEIGHT,
                rrf_k=settings.HYBRID_SEARCH_RRF_K,
                reranker=reranker,
                enable_reranking=False  # Disable reranking (optional PyTorch feature)
            )
            
            # Log which retrievers are active
            if semantic_retriever and hasattr(semantic_retriever, 'faiss_index') and semantic_retriever.faiss_index is not None:
                logger.info("Hybrid retrieval active: BM25 + OpenAI + FAISS")
            elif semantic_retriever and hasattr(semantic_retriever, 'embedding_gen') and semantic_retriever.embedding_gen:
                logger.info("Hybrid retrieval active: BM25 + OpenAI + FAISS")
            elif semantic_retriever:
                logger.info("Retrievers active: BM25 + TF-IDF (no FAISS)")
            
        except Exception as e:
            logger.error(f"Error initializing hybrid retriever: {e}", exc_info=True)
            self.hybrid_retriever = None
    
    def _load_vector_store(self):
        """Load FAISS index and chunk metadata. Paths: data/faiss_index.bin, data/chunk_metadata.pkl.
        Crashes startup on dimension mismatch or metadata count mismatch. No degraded state."""
        faiss_path = project_root / "data" / "faiss_index.bin"
        metadata_path = project_root / "data" / "chunk_metadata.pkl"

        self.faiss_index = None
        self.chunk_metadata = []

        if faiss_path.exists() and metadata_path.exists():
            self.faiss_index = faiss.read_index(str(faiss_path))
            with open(metadata_path, "rb") as f:
                self.chunk_metadata = pickle.load(f)

            faiss_d = self.faiss_index.d
            ntotal = getattr(self.faiss_index, "ntotal", 0)
            n_chunks = len(self.chunk_metadata) if self.chunk_metadata else 0
            runtime_dim = settings.EMBEDDING_DIMENSION

            logger.info(
                "FAISS load: faiss_dim=%s | runtime_embedding_dim=%s | metadata_chunk_count=%s | faiss_ntotal=%s",
                faiss_d, runtime_dim, n_chunks, ntotal,
            )

            if faiss_d != runtime_dim:
                raise RuntimeError(
                    f"FAISS embedding dimension mismatch: index has d={faiss_d}, "
                    f"settings.EMBEDDING_DIMENSION={runtime_dim}. "
                    "Re-run ingestion with the correct embedding model (e.g. python scripts/ingest_data.py)."
                )
            if ntotal != n_chunks:
                raise RuntimeError(
                    f"FAISS metadata count mismatch: index has ntotal={ntotal}, "
                    f"chunk_metadata has {n_chunks} chunks. "
                    "Re-run ingestion to rebuild index and metadata."
                )
        elif metadata_path.exists():
            self.faiss_index = None
            try:
                with open(metadata_path, "rb") as f:
                    self.chunk_metadata = pickle.load(f)
                n_chunks = len(self.chunk_metadata) if self.chunk_metadata else 0
                logger.info("Metadata-only loaded: FAISS ntotal=N/A | chunks_loaded=%s (TF-IDF only)", n_chunks)
            except Exception as e:
                logger.warning("Could not load data/chunk_metadata.pkl: %s", e)
                self.chunk_metadata = []
        else:
            logger.warning("FAISS index not found. Expected data/faiss_index.bin and data/chunk_metadata.pkl")
    
    def search(
        self,
        query: str,
        top_k: int = 10,
        use_hybrid: Optional[bool] = None,
        fusion_strategy: Optional[str] = None,
        metadata_filter: Optional[MetadataFilter] = None,
        include_explanation: bool = False,
        highlight_sources: bool = False,
        user_id: Optional[int] = None,
        include_private_corpus: bool = False,
        private_corpus_results: Optional[List[Dict[str, Any]]] = None,
        request_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search for relevant chunks.
        
        Args:
            query: Search query text
            top_k: Number of top results to return
            use_hybrid: If True, use hybrid search; if None, use instance default (self.use_hybrid)
            fusion_strategy: Fusion strategy for hybrid search ("rrf" or "weighted"). Only used if use_hybrid=True
            metadata_filter: Optional metadata filter for hybrid search
            include_explanation: Whether to include explanation in results
            highlight_sources: Whether to highlight matched terms
            user_id: Optional user ID for private corpus search
            include_private_corpus: Whether to include user's private documents in search
            private_corpus_results: Pre-fetched private corpus results (optional)
            
        Returns:
            List of result dictionaries
        """
        logger.info(f"🔍 Starting search for query: '{query[:50]}...' (top_k={top_k})")
        
        # Check service state: need at least chunk_metadata for any search
        if len(self.chunk_metadata) == 0:
            logger.error("❌ No chunk metadata available - search cannot proceed")
            logger.info(
                "RAGService.search early return: retriever_type=N/A | faiss_ntotal=N/A | embedding_dim=N/A | chunks_retrieved=0"
            )
            return []
        if self.faiss_index is not None:
            logger.info(f"📊 Service state: FAISS index has {self.faiss_index.ntotal} vectors, {len(self.chunk_metadata)} chunks")
        
        # When FAISS is missing but metadata exists: TF-IDF only (reversible fallback for Sale of Goods etc.)
        use_hybrid_search = use_hybrid if use_hybrid is not None else self.use_hybrid
        if self.faiss_index is None:
            use_hybrid_search = False
            logger.info("📊 No FAISS index; using TF-IDF-only search over %s chunks", len(self.chunk_metadata))
        
        if use_hybrid_search and self.hybrid_retriever:
            semantic_ret = self.hybrid_retriever.semantic_retriever if self.hybrid_retriever else None
            uses_faiss = (
                semantic_ret is not None
                and hasattr(semantic_ret, "faiss_index")
                and semantic_ret.faiss_index is not None
            )
            retriever_type = "hybrid (BM25+OpenAI+FAISS)" if uses_faiss else "hybrid (BM25+TF-IDF)"
            if uses_faiss:
                logger.info("Retrievers active: BM25 + OpenAI + FAISS")
            else:
                logger.info("Retrievers active: BM25 + TF-IDF (no FAISS)")
            public_results = self._hybrid_search(
                query, top_k, fusion_strategy, metadata_filter, request_id=request_id
            )
        else:
            has_emb = (
                self.embedding_gen is not None
                and hasattr(self.embedding_gen, "model")
                and self.embedding_gen.model is not None
                and self.faiss_index is not None
            )
            retriever_type = "FAISS (OpenAI+FAISS)" if has_emb else "TF-IDF"
            if has_emb:
                logger.info("Retrievers active: OpenAI + FAISS")
            else:
                logger.info("Retrievers active: TF-IDF only (no FAISS)")
            public_results = self._semantic_search(
                query, top_k, request_id=request_id, retriever_type=retriever_type
            )
        
        logger.info(f"📋 Public corpus returned {len(public_results)} results")
        
        # Combine with private corpus if requested
        if include_private_corpus and user_id and private_corpus_results:
            logger.info(f"🔗 Combining with {len(private_corpus_results)} private corpus results")
            combined_results = self._combine_results(public_results, private_corpus_results, top_k)
        else:
            combined_results = public_results
        
        # Per-request debug summary (INFO); top 3 chunk details at DEBUG (hybrid/semantic log their own)
        req_prefix = f"request_id={request_id} " if request_id else ""
        logger.info(
            "RAG_DEBUG %sretriever_type=%s | chunks_after_fusion=%s | chunks_final=%s",
            req_prefix, retriever_type, len(public_results), len(combined_results),
        )
        for i, r in enumerate(public_results[:3], 1):
            chunk_id = r.get("chunk_id", "N/A")
            title = (r.get("metadata") or {}).get("title", r.get("title", "N/A"))
            logger.debug("RAG_DEBUG %stop_%s chunk_id=%s title=%s", req_prefix, i, chunk_id, title)
        
        # Add explainability if requested
        if include_explanation or highlight_sources:
            combined_results = self._add_explainability(combined_results, query, include_explanation, highlight_sources)
        
        return combined_results
    
    def _combine_results(
        self,
        public_results: List[Dict[str, Any]],
        private_results: List[Dict[str, Any]],
        top_k: int
    ) -> List[Dict[str, Any]]:
        """
        Combine public and private corpus results using reciprocal rank fusion (RRF).
        
        Args:
            public_results: Results from public corpus
            private_results: Results from private corpus
            top_k: Number of top results to return
            
        Returns:
            Combined and ranked results
        """
        import numpy as np
        
        # Create a dictionary to track combined scores
        combined_scores = {}
        k = 60  # RRF parameter
        
        # Add public results with RRF scores
        for rank, result in enumerate(public_results, start=1):
            chunk_id = result.get("chunk_id", f"public_{rank}")
            if chunk_id not in combined_scores:
                combined_scores[chunk_id] = {
                    "result": result,
                    "score": 0.0,
                    "is_private": False
                }
            combined_scores[chunk_id]["score"] += 1.0 / (k + rank)
            # Mark metadata to indicate public corpus
            if "metadata" not in combined_scores[chunk_id]["result"]:
                combined_scores[chunk_id]["result"]["metadata"] = {}
            combined_scores[chunk_id]["result"]["metadata"]["corpus"] = "public"
        
        # Add private results with RRF scores
        for rank, result in enumerate(private_results, start=1):
            chunk_id = result.get("chunk_id", f"private_{rank}")
            if chunk_id not in combined_scores:
                combined_scores[chunk_id] = {
                    "result": result,
                    "score": 0.0,
                    "is_private": True
                }
            combined_scores[chunk_id]["score"] += 1.0 / (k + rank)
            # Mark metadata to indicate private corpus
            if "metadata" not in combined_scores[chunk_id]["result"]:
                combined_scores[chunk_id]["result"]["metadata"] = {}
            combined_scores[chunk_id]["result"]["metadata"]["corpus"] = "private"
        
        # Sort by combined score
        sorted_results = sorted(
            combined_scores.values(),
            key=lambda x: x["score"],
            reverse=True
        )
        
        # Format results
        formatted_results = []
        for item in sorted_results[:top_k]:
            result = item["result"].copy()
            result["similarity_score"] = item["score"]  # Update with combined score
            formatted_results.append(result)
        
        return formatted_results
    
    def _hybrid_search(
        self,
        query: str,
        top_k: int = 10,
        fusion_strategy: Optional[str] = None,
        metadata_filter: Optional[MetadataFilter] = None,
        request_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Perform hybrid search using BM25 + Semantic fusion"""
        try:
            # Temporarily override fusion strategy if provided
            original_strategy = self.hybrid_retriever.fusion_strategy
            if fusion_strategy:
                self.hybrid_retriever.fusion_strategy = fusion_strategy
            
            # Perform hybrid search
            results = self.hybrid_retriever.search(
                query=query,
                top_k=top_k,
                metadata_filter=metadata_filter,
                pre_filter=True,
                request_id=request_id,
            )
            
            # Restore original strategy
            if fusion_strategy:
                self.hybrid_retriever.fusion_strategy = original_strategy
            
            # Format results to match existing API format
            formatted_results = []
            for result in results:
                formatted_result = {
                    "chunk_id": result.get("chunk_id", ""),
                    "text": result.get("text", ""),
                    "metadata": result.get("metadata", {}),
                    "similarity_score": result.get("similarity_score", 0.0),
                    "section": result.get("section", "Unknown"),
                    # Additional hybrid-specific fields
                    "bm25_score": result.get("bm25_score"),
                    "semantic_score": result.get("semantic_score"),
                    "bm25_rank": result.get("bm25_rank"),
                    "semantic_rank": result.get("semantic_rank"),
                    "fusion_strategy": self.hybrid_retriever.fusion_strategy,
                    # Reranking fields (if available)
                    "rerank_score": result.get("rerank_score"),
                    "rerank_rank": result.get("rerank_rank")
                }
                formatted_results.append(formatted_result)
            
            logger.debug(f"Hybrid search returned {len(formatted_results)} results")
            return formatted_results
            
        except Exception as e:
                logger.error(f"Hybrid search failed: {e}", exc_info=True)
                # Fall back to semantic search
                logger.warning("Falling back to semantic search")
                return self._semantic_search(query, top_k)
    
    def _add_explainability(
        self,
        results: List[Dict[str, Any]],
        query: str,
        include_explanation: bool = True,
        highlight_sources: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Add explainability information and highlighting to results.
        
        Args:
            results: List of retrieval results
            query: Search query
            include_explanation: If True, add explanation fields
            highlight_sources: If True, add highlighted text
            
        Returns:
            List of results with explainability fields added
        """
        # Initialize explainability analyzer lazily to avoid PyTorch imports at startup
        if self.explainability_analyzer is None:
            try:
                # Lazy import - only import when actually needed
                from retrieval.explainability import ExplainabilityAnalyzer
                self.explainability_analyzer = ExplainabilityAnalyzer()
            except ImportError as e:
                logger.warning(f"Could not import explainability analyzer: {e}")
                self.explainability_analyzer = None
            except Exception as e:
                logger.warning(f"Could not initialize explainability analyzer: {e}")
                self.explainability_analyzer = None
        
        if self.explainability_analyzer is None:
            # Can't add explainability, return results as-is
            return results
        
        try:
            if highlight_sources:
                # Add highlighted text
                results = self.explainability_analyzer.highlight_sources(results, query, highlight_tag="**")
            
            if include_explanation:
                # Add explanations
                explanations = self.explainability_analyzer.explain_results(results, query)
                
                for result, explanation in zip(results, explanations):
                    result["explanation"] = explanation.explanation
                    result["confidence"] = explanation.confidence
                    result["matched_terms"] = explanation.matched_terms
                    result["matched_spans"] = explanation.matched_text_spans
        except Exception as e:
            logger.warning(f"Explainability processing failed: {e}")
            # Return results without explainability if processing fails
        
        return results
    
    def _get_or_init_embedding_gen(self):
        """Lazy initialization of embedding generator (prevents segfaults at startup)"""
        if self.embedding_gen is not None:
            return self.embedding_gen
        
        if not hasattr(self, '_embedding_config') or self._embedding_config is None:
            return None
        
        try:
            from retrieval.embeddings.embedding_generator import EmbeddingGenerator, EmbeddingConfig
            
            logger.info("🔄 Initializing embedding generator (lazy load)...")
            embedding_config = EmbeddingConfig(
                model_name=self._embedding_config["model_name"],
                dimension=self._embedding_config["dimension"]
            )
            
            self.embedding_gen = EmbeddingGenerator(embedding_config)
            
            if self.embedding_gen.model is None:
                logger.warning("⚠️ Embedding model not available - using TF-IDF fallback")
                self.embedding_gen = None
                return None
            else:
                logger.info("✅ Embedding generator initialized (semantic search available)")
                return self.embedding_gen
        except Exception as e:
            logger.warning(f"⚠️ Failed to initialize embedding generator: {e}")
            self.embedding_gen = None
            return None
    
    def _search_with_embeddings(self, query: str, top_k: int, embedding_gen) -> List[Dict[str, Any]]:
        """Search using embeddings and FAISS index"""
        try:
            # Check if embedding_gen is available (works for both OpenAI and PyTorch)
            if not self.faiss_index or not embedding_gen:
                return []
            
            # For OpenAI embeddings, model property returns self (truthy)
            # For PyTorch embeddings, model is the actual model object
            has_model = hasattr(embedding_gen, 'model') and embedding_gen.model is not None
            if not has_model:
                return []
            
            # Generate query embedding
            query_embedding_list = embedding_gen.generate_embedding(query)
            if not query_embedding_list:
                return []
            
            query_embedding = np.array(query_embedding_list, dtype=np.float32)
            
            # Normalize embedding
            norm = np.linalg.norm(query_embedding)
            if norm == 0 or np.isnan(norm) or np.isinf(norm):
                return []
            query_normalized = query_embedding / norm
            query_normalized = np.ascontiguousarray(query_normalized, dtype=np.float32)
            
            # Search FAISS index
            k = min(top_k, self.faiss_index.ntotal)
            similarities, indices = self.faiss_index.search(query_normalized.reshape(1, -1), k)
            
            # Format results
            results = []
            for idx, similarity in zip(indices[0], similarities[0]):
                if idx == -1 or idx >= len(self.chunk_metadata):
                    continue
                
                chunk_data = self.chunk_metadata[idx]
                results.append({
                    "chunk_id": chunk_data.get("chunk_id", f"chunk_{idx}"),
                    "text": chunk_data.get("text", ""),
                    "metadata": chunk_data.get("metadata", {}),
                    "similarity_score": float(similarity),
                    "section": chunk_data.get("metadata", {}).get("section", "Unknown")
                })
            
            return results[:top_k]
        except Exception as e:
            logger.error(f"Embedding search failed: {e}", exc_info=True)
            return []
    
    def _semantic_search(
        self,
        query: str,
        top_k: int = 10,
        request_id: Optional[str] = None,
        retriever_type: str = "FAISS",
    ) -> List[Dict[str, Any]]:
        """Semantic search using embeddings. TF-IDF fallback only when FAISS does not exist."""
        embedding_gen = self.embedding_gen

        has_valid_embeddings = (
            embedding_gen is not None
            and hasattr(embedding_gen, "model")
            and embedding_gen.model is not None
            and self.faiss_index is not None
        )

        if has_valid_embeddings:
            logger.info("Semantic retriever: OpenAI + FAISS")
            try:
                results = self._search_with_embeddings(query, top_k, embedding_gen)
            except Exception as e:
                raise RuntimeError(
                    f"FAISS search failed: {e}. TF-IDF fallback is disabled when FAISS exists."
                ) from e
        elif self.faiss_index is not None:
            raise RuntimeError(
                "FAISS exists but embeddings unavailable. TF-IDF fallback disabled."
            )
        else:
            logger.info("Retrievers active: TF-IDF only (no FAISS)")
            results = self._search_with_tfidf(query, top_k)

        # Per-request debug logging (INFO/DEBUG only)
        req_prefix = f"request_id={request_id} " if request_id else ""
        logger.info(
            "RAG_DEBUG %sretriever_type=%s | bm25_chunks=0 | faiss_chunks=%s | chunks_after_fusion=%s",
            req_prefix, retriever_type, len(results), len(results),
        )
        for i, r in enumerate(results[:3], 1):
            chunk_id = r.get("chunk_id", "N/A")
            title = (r.get("metadata") or {}).get("title", r.get("title", "N/A"))
            logger.debug("RAG_DEBUG %stop_%s chunk_id=%s title=%s", req_prefix, i, chunk_id, title)
        return results
        
        # REMOVED: All PyTorch embedding code to prevent segfaults
        # This code path is no longer used - we always use TF-IDF
        # Keeping structure for backward compatibility but it won't execute
        try:
            # Generate query embedding - DISABLED to prevent PyTorch imports
            query_matrix = None
            # CRITICAL: Never use embeddings - always fall back to TF-IDF
            if False and self.embedding_gen and hasattr(self.embedding_gen, 'generate_embedding') and self.embedding_gen.model is not None:
                try:
                    # Sentence transformers - FIXED: Convert list to numpy array
                    query_embedding_list = self.embedding_gen.generate_embedding(query)
                    
                    # Validate embedding was generated
                    if not query_embedding_list or len(query_embedding_list) == 0:
                        logger.error("Empty embedding generated")
                        return []
                    
                    query_embedding = np.array(query_embedding_list, dtype=np.float32)
                    
                    # Validate array is valid
                    if query_embedding.size == 0:
                        logger.error("Invalid embedding array")
                        return []
                    
                    # CRITICAL FIX: Check dimension mismatch
                    expected_dim = self.faiss_index.d
                    actual_dim = len(query_embedding)
                    
                    if actual_dim != expected_dim:
                        logger.error(f"Dimension mismatch! FAISS index expects {expected_dim}D, got {actual_dim}D embedding. Re-run ingestion with matching model.")
                        return []
                    
                    # CRITICAL FIX: Prevent zero division
                    norm = np.linalg.norm(query_embedding)
                    if norm == 0 or np.isnan(norm) or np.isinf(norm):
                        logger.error("Query embedding has zero or invalid norm")
                        return []
                    
                    query_normalized = query_embedding / norm
                    
                    # CRITICAL FIX: Ensure contiguous memory for FAISS (prevents segfault)
                    query_normalized = np.ascontiguousarray(query_normalized, dtype=np.float32)
                    query_matrix = query_normalized.reshape(1, -1)
                    
                except RuntimeError as embed_error:
                    # RuntimeError means model failed to load (PyTorch issue)
                    error_msg = str(embed_error)
                    logger.error(f"CRITICAL: Embedding generation failed (PyTorch issue): {embed_error}", exc_info=True)
                    if "libtorch" in error_msg.lower() or "dlopen" in error_msg.lower():
                        logger.error("PyTorch library is broken. This causes segfaults. Please fix PyTorch installation.")
                    return []  # Return empty instead of crashing
                except Exception as embed_error:
                    logger.error(f"CRITICAL: Embedding generation failed: {embed_error}", exc_info=True)
                    return []  # Return empty instead of crashing
            else:
                # Model not available - PyTorch likely broken
                logger.error("Embedding model not available. PyTorch installation may be broken (missing libtorch_cpu.dylib).")
                logger.error("This prevents search from working. Please fix PyTorch or reinstall sentence-transformers.")
                return []
            
            if query_matrix is None:
                logger.error("Query matrix is None")
                return []
            
            # Validate query_matrix before FAISS search
            if query_matrix.shape[0] != 1 or query_matrix.shape[1] != self.faiss_index.d:
                logger.error(f"Invalid query matrix shape: {query_matrix.shape}, expected (1, {self.faiss_index.d})")
                return []
            
            # Search - FIXED: query_matrix is now contiguous and validated
            try:
                scores, indices = self.faiss_index.search(
                    query_matrix,
                    k=min(top_k, len(self.chunk_metadata))
                )
            except Exception as faiss_error:
                logger.error(f"CRITICAL: FAISS search failed: {faiss_error}", exc_info=True)
                return []  # Return empty instead of crashing
            
            # Format results
            results = []
            try:
                logger.debug(f"FAISS search returned {len(scores[0])} scores, {len(indices[0])} indices")
                valid_results = 0
                for score, idx in zip(scores[0], indices[0]):
                    if idx < len(self.chunk_metadata) and idx >= 0:
                        chunk_data = self.chunk_metadata[idx]
                        # Validate score is not NaN or Inf
                        score_value = float(score)
                        if np.isnan(score_value) or np.isinf(score_value):
                            score_value = 0.0
                        
                        results.append({
                            "chunk_id": chunk_data.get("chunk_id", f"chunk_{idx}"),
                            "text": chunk_data.get("text", ""),
                            "metadata": chunk_data.get("metadata", {}),
                            "similarity_score": score_value,
                            "section": chunk_data.get("metadata", {}).get("section", "Unknown")
                        })
                        valid_results += 1
                    else:
                        logger.warning(f"Invalid index {idx} (metadata length: {len(self.chunk_metadata)})")
                
                logger.info(f"✅ Formatted {valid_results} valid results from FAISS search")
                if valid_results == 0:
                    logger.warning("⚠️ No valid results after formatting - all indices were invalid")
            except Exception as format_error:
                logger.error(f"Error formatting results: {format_error}", exc_info=True)
                return []
            
            return results
            
        except Exception as e:
            logger.error(f"Error during search: {e}", exc_info=True)
            return []
    
    def _search_with_tfidf(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Fallback TF-IDF keyword search when embeddings are unavailable"""
        try:
            # CRITICAL FIX: Use sklearn only - no PyTorch dependencies
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity
            
            # Extract texts from metadata
            if not self.chunk_metadata or len(self.chunk_metadata) == 0:
                logger.warning("No chunk metadata available for TF-IDF search")
                return []
            
            texts = [chunk.get("text", "") for chunk in self.chunk_metadata]
            
            if not texts or len(texts) == 0:
                logger.warning("No texts available for TF-IDF search")
                return []
            
            # Create TF-IDF vectors
            vectorizer = TfidfVectorizer(max_features=1000, stop_words='english', ngram_range=(1, 2))
            doc_vectors = vectorizer.fit_transform(texts)
            query_vector = vectorizer.transform([query])
            
            # Calculate similarities
            similarities = cosine_similarity(query_vector, doc_vectors).flatten()
            
            # Get top-k results
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            results = []
            for idx in top_indices:
                if similarities[idx] > 0:
                    chunk_data = self.chunk_metadata[idx]
                    results.append({
                        "chunk_id": chunk_data.get("chunk_id", f"chunk_{idx}"),
                        "text": chunk_data.get("text", ""),
                        "metadata": chunk_data.get("metadata", {}),
                        "similarity_score": float(similarities[idx]),
                        "section": chunk_data.get("metadata", {}).get("section", "Unknown")
                    })
            
            return results
        except Exception as e:
            logger.error(f"TF-IDF search failed: {e}", exc_info=True)
            return []
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the RAG service"""
        stats = {
            "use_hybrid": self.use_hybrid,
            "has_embedding_gen": self.embedding_gen is not None and self.embedding_gen.model is not None,
            "has_faiss_index": self.faiss_index is not None,
            "num_chunks": len(self.chunk_metadata),
            "has_hybrid_retriever": self.hybrid_retriever is not None
        }
        
        if self.hybrid_retriever:
            stats["hybrid_stats"] = self.hybrid_retriever.get_stats()
        
        return stats