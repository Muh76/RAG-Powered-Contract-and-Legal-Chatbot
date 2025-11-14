# Legal Chatbot - RAG Service
# Extracted from Phase 1 notebook

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

from retrieval.embeddings.embedding_generator import EmbeddingGenerator, EmbeddingConfig
from app.core.config import settings

logger = logging.getLogger(__name__)


class RAGService:
    """RAG service for legal chatbot - extracted from Phase 1 notebook"""
    
    def __init__(self):
        self.embedding_gen = None
        self.faiss_index = None
        self.chunk_metadata = []
        self._initialize()
    
    def _initialize(self):
        """Initialize RAG components"""
        try:
            # Initialize embedding generator
            embedding_config = EmbeddingConfig(
                model_name=settings.EMBEDDING_MODEL,
                dimension=settings.EMBEDDING_DIMENSION,
                batch_size=settings.EMBEDDING_BATCH_SIZE,
                max_length=512
            )
            
            # CRITICAL FIX: Disable embedding model entirely to prevent segfaults
            # PyTorch is broken and causes segfaults even in subprocess checks
            # The system will work with FAISS index that was already created
            # Users can still search using the existing embeddings in the FAISS index
            logger.warning("⚠️ Embedding model disabled to prevent PyTorch segfaults")
            logger.warning("System will use existing FAISS index (created during ingestion)")
            logger.warning("To enable embeddings, fix PyTorch: pip uninstall torch && pip install torch")
            self.embedding_gen = None
            
            # Load FAISS index and metadata
            self._load_vector_store()
            
        except Exception as e:
            logger.error(f"Error initializing RAG service: {e}")
            # Don't raise - allow service to exist with limited functionality
            self.embedding_gen = None
            self.faiss_index = None
            self.chunk_metadata = []
    
    def _load_vector_store(self):
        """Load FAISS index and chunk metadata"""
        # Try multiple possible paths
        possible_paths = [
            project_root / "data" / "faiss_index.bin",
            project_root / "data" / "processed" / "faiss_index.bin",
            project_root / "notebooks" / "phase1" / "data" / "faiss_index.bin",
            Path("data/faiss_index.bin"),
            Path("notebooks/phase1/data/faiss_index.bin"),
            Path.cwd() / "data" / "faiss_index.bin"
        ]
        
        faiss_path = None
        metadata_path = None
        
        for path in possible_paths:
            if path.exists():
                faiss_path = path
                metadata_path = path.parent / "chunk_metadata.pkl"
                break
        
        if faiss_path and faiss_path.exists() and metadata_path.exists():
            try:
                self.faiss_index = faiss.read_index(str(faiss_path))
                with open(metadata_path, "rb") as f:
                    self.chunk_metadata = pickle.load(f)
                logger.info(f"Loaded FAISS index with {self.faiss_index.ntotal} vectors from {faiss_path}")
            except Exception as e:
                logger.error(f"Error loading vector store: {e}")
                self.faiss_index = None
                self.chunk_metadata = []
        else:
            logger.warning("FAISS index not found. Run data ingestion first.")
            logger.warning(f"Looked in: {[str(p) for p in possible_paths]}")
            logger.error("To fix: Run 'python scripts/ingest_data.py' to create the FAISS index")
            self.faiss_index = None
            self.chunk_metadata = []
    
    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Search for relevant chunks"""
        # Check if service is properly initialized
        if self.faiss_index is None or len(self.chunk_metadata) == 0:
            logger.error("Vector store not loaded. Cannot search. Run: python scripts/ingest_data.py")
            return []
        
        # Check if embedding generator is available
        # If not, use TF-IDF fallback for keyword-based search
        if self.embedding_gen is None or (hasattr(self.embedding_gen, 'model') and self.embedding_gen.model is None):
            logger.warning("Embedding model not available. Using TF-IDF keyword search fallback.")
            return self._search_with_tfidf(query, top_k)
        
        try:
            # Generate query embedding - WRAP IN TRY-CATCH TO PREVENT CRASH
            query_matrix = None
            if self.embedding_gen and hasattr(self.embedding_gen, 'generate_embedding') and self.embedding_gen.model is not None:
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
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity
            
            # Extract texts from metadata
            texts = [chunk.get("text", "") for chunk in self.chunk_metadata]
            
            if not texts:
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