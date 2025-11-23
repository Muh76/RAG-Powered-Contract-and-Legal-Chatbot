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
        try:
            logger.info("Loading FAISS index and metadata...")
            self._load_vector_store()
            logger.info(f"✅ Loaded {len(self.chunk_metadata)} chunks from FAISS index")
        except Exception as e:
            logger.error(f"Failed to load FAISS index: {e}")
            self.faiss_index = None
            self.chunk_metadata = []
            # Continue anyway - can use TF-IDF fallback
        
        # Step 2: PERMANENTLY DISABLE embedding generator to prevent PyTorch segfaults
        # PyTorch/SentenceTransformer has been removed - system now uses TF-IDF only
        # This ensures the system works seamlessly without crashes
        logger.info("⚠️ Embeddings PERMANENTLY DISABLED to prevent PyTorch segfaults")
        logger.info("✅ System will use TF-IDF keyword search only (stable and fast)")
        self.embedding_gen = None
        
        # Step 3: PERMANENTLY DISABLE hybrid retriever to prevent PyTorch imports
        # Hybrid retriever requires embeddings which require PyTorch
        logger.info("⚠️ Hybrid retriever PERMANENTLY DISABLED to prevent PyTorch imports")
        self.hybrid_retriever = None
        self.use_hybrid = False  # Force disable hybrid search
        
        # Final status check
        if self.embedding_gen is None and self.faiss_index is None:
            logger.warning("⚠️ RAG service initialized in degraded mode (no embeddings, no FAISS index)")
        elif self.embedding_gen is None:
            logger.warning("⚠️ RAG service initialized with FAISS but no embeddings (will use TF-IDF fallback)")
        else:
            logger.info("✅ RAG service fully initialized")
    
    def _initialize_hybrid_retriever(self):
        """Initialize hybrid retriever components - with lazy imports to prevent PyTorch segfaults"""
        try:
            if not self.chunk_metadata or len(self.chunk_metadata) == 0:
                logger.warning("Cannot initialize hybrid retriever: no chunk metadata available")
                return
            
            # CRITICAL FIX: Only import PyTorch-dependent modules when actually needed
            # This prevents PyTorch from being imported at module load time
            try:
                from retrieval.semantic_retriever import SemanticRetriever
                from retrieval.hybrid_retriever import AdvancedHybridRetriever, FusionStrategy
            except ImportError as e:
                logger.warning(f"Could not import hybrid retriever components: {e}")
                self.hybrid_retriever = None
                return
            
            # Extract documents for BM25
            documents = [chunk.get("text", "") for chunk in self.chunk_metadata]
            
            # Initialize BM25 retriever
            self.bm25_retriever = BM25Retriever(documents)
            logger.info("✅ BM25 retriever initialized")
            
            # Initialize semantic retriever (only if embeddings available)
            if self.embedding_gen is None or (hasattr(self.embedding_gen, 'model') and self.embedding_gen.model is None):
                logger.warning("Cannot initialize semantic retriever: no embeddings available")
                self.hybrid_retriever = None
                return
            
            try:
                self.semantic_retriever = SemanticRetriever()
                if not self.semantic_retriever.is_ready():
                    logger.warning("Semantic retriever not ready, hybrid search will have limited functionality")
                    self.hybrid_retriever = None
                    return
            except Exception as e:
                logger.warning(f"Failed to initialize semantic retriever: {e}")
                self.hybrid_retriever = None
                return
            
            # Initialize reranker if enabled (lazy import to prevent PyTorch crash)
            reranker = None
            if settings.ENABLE_RERANKING:
                try:
                    # Lazy import - only when reranking is actually enabled
                    from retrieval.rerankers.cross_encoder_reranker import CrossEncoderReranker
                    reranker = CrossEncoderReranker(
                        model_name=settings.RERANKER_MODEL,
                        batch_size=settings.RERANKER_BATCH_SIZE
                    )
                    if reranker.is_ready():
                        logger.info("✅ Cross-encoder reranker initialized")
                    else:
                        logger.warning("⚠️ Cross-encoder reranker not available")
                        reranker = None
                except Exception as e:
                    logger.warning(f"Failed to initialize reranker: {e}")
                    reranker = None
            
            # Create hybrid retriever
            self.hybrid_retriever = AdvancedHybridRetriever(
                bm25_retriever=self.bm25_retriever,
                semantic_retriever=self.semantic_retriever,
                chunk_metadata=self.chunk_metadata,
                fusion_strategy=settings.HYBRID_SEARCH_FUSION_STRATEGY,
                bm25_weight=settings.HYBRID_SEARCH_BM25_WEIGHT,
                semantic_weight=settings.HYBRID_SEARCH_SEMANTIC_WEIGHT,
                rrf_k=settings.HYBRID_SEARCH_RRF_K,
                reranker=reranker,
                enable_reranking=settings.ENABLE_RERANKING and reranker is not None
            )
            logger.info("✅ Hybrid retriever initialized")
            
        except Exception as e:
            logger.error(f"Error initializing hybrid retriever: {e}", exc_info=True)
            self.hybrid_retriever = None
    
    def _load_vector_store(self):
        """Load FAISS index and chunk metadata with robust error handling"""
        # Try multiple possible paths
        possible_paths = [
            project_root / "data" / "indices" / "faiss_index.pkl",
            project_root / "data" / "faiss_index.bin",
            project_root / "data" / "processed" / "faiss_index.bin",
            project_root / "notebooks" / "phase1" / "data" / "faiss_index.bin",
            Path("data/indices/faiss_index.pkl"),
            Path("data/faiss_index.bin"),
            Path("notebooks/phase1/data/faiss_index.bin"),
            Path.cwd() / "data" / "faiss_index.bin"
        ]
        
        faiss_path = None
        metadata_path = None
        
        for path in possible_paths:
            if path.exists():
                faiss_path = path
                # Try both .pkl and separate metadata file
                if path.suffix == '.pkl':
                    metadata_path = path  # Combined file
                else:
                    metadata_path = path.parent / "chunk_metadata.pkl"
                break
        
        # Check for combined .pkl file first
        pkl_path = None
        for path in possible_paths:
            if path.exists() and path.suffix == '.pkl':
                pkl_path = path
                break
        
        if pkl_path:
            # Load from combined .pkl file
            try:
                with open(pkl_path, 'rb') as f:
                    data = pickle.load(f)
                    self.faiss_index = data.get('faiss_index')
                    self.chunk_metadata = data.get('chunk_metadata', [])
                if self.faiss_index:
                    logger.info(f"✅ Loaded FAISS index from {pkl_path}: {self.faiss_index.ntotal} vectors, {len(self.chunk_metadata)} chunks")
                else:
                    logger.warning(f"⚠️  FAISS index is None in {pkl_path}")
                    self.faiss_index = None
                    self.chunk_metadata = []
            except Exception as e:
                logger.error(f"❌ Error loading combined FAISS index from {pkl_path}: {e}")
                self.faiss_index = None
                self.chunk_metadata = []
        elif faiss_path and faiss_path.exists():
            # Load from separate files
            if metadata_path and metadata_path.exists():
                try:
                    self.faiss_index = faiss.read_index(str(faiss_path))
                    with open(metadata_path, "rb") as f:
                        self.chunk_metadata = pickle.load(f)
                    logger.info(f"✅ Loaded FAISS index with {self.faiss_index.ntotal} vectors from {faiss_path}")
                except Exception as e:
                    logger.error(f"❌ Error loading vector store: {e}")
                    self.faiss_index = None
                    self.chunk_metadata = []
            else:
                logger.warning(f"FAISS index found but metadata not found: {metadata_path}")
                self.faiss_index = None
                self.chunk_metadata = []
        else:
            logger.warning("FAISS index not found. Run data ingestion first.")
            logger.warning(f"Looked in: {[str(p) for p in possible_paths]}")
            logger.error("To fix: Run 'python scripts/ingest_data.py' to create the FAISS index")
            self.faiss_index = None
            self.chunk_metadata = []
    
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
        private_corpus_results: Optional[List[Dict[str, Any]]] = None
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
        
        # Check service state
        if self.faiss_index is None:
            logger.error("❌ FAISS index is None - search cannot proceed")
            return []
        if len(self.chunk_metadata) == 0:
            logger.error("❌ No chunk metadata available - search cannot proceed")
            return []
        
        logger.info(f"📊 Service state: FAISS index has {self.faiss_index.ntotal} vectors, {len(self.chunk_metadata)} chunks")
        
        # Determine if we should use hybrid search
        use_hybrid_search = use_hybrid if use_hybrid is not None else self.use_hybrid
        
        # Get public corpus results
        if use_hybrid_search and self.hybrid_retriever:
            logger.info("🔀 Using hybrid search")
            public_results = self._hybrid_search(query, top_k, fusion_strategy, metadata_filter)
        else:
            # Fall back to original semantic search (backward compatible)
            logger.info("🔎 Using semantic search")
            public_results = self._semantic_search(query, top_k)
        
        logger.info(f"📋 Public corpus returned {len(public_results)} results")
        
        # Combine with private corpus if requested
        if include_private_corpus and user_id and private_corpus_results:
            logger.info(f"🔗 Combining with {len(private_corpus_results)} private corpus results")
            combined_results = self._combine_results(public_results, private_corpus_results, top_k)
        else:
            combined_results = public_results
        
        logger.info(f"✅ Final search returned {len(combined_results)} results")
        
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
        metadata_filter: Optional[MetadataFilter] = None
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
                pre_filter=True
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
    
    def _semantic_search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Original semantic search method (backward compatible) - NOW USING TF-IDF ONLY"""
        # PERMANENT FIX: Always use TF-IDF to avoid PyTorch segfaults
        # PyTorch/SentenceTransformer has been removed to prevent crashes
        logger.info("Using TF-IDF keyword search (PyTorch embeddings disabled for stability)")
        try:
            return self._search_with_tfidf(query, top_k)
        except Exception as tfidf_error:
            logger.error(f"TF-IDF search failed: {tfidf_error}", exc_info=True)
            return []
        
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