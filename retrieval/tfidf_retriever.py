# Legal Chatbot - TF-IDF Retriever
# Wrapper for TF-IDF search to work with hybrid retriever (no PyTorch)

import numpy as np
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class TFIDFRetriever:
    """
    TF-IDF retriever wrapper that mimics SemanticRetriever interface
    for use in hybrid search (no PyTorch required).
    """
    
    def __init__(self, chunk_metadata: List[Dict[str, Any]]):
        """
        Initialize TF-IDF retriever.
        
        Args:
            chunk_metadata: List of chunk dictionaries with 'text' and 'metadata' keys
        """
        self.chunk_metadata = chunk_metadata
        self.vectorizer = None
        self.doc_vectors = None
        self._initialize()
    
    def _initialize(self):
        """Initialize TF-IDF vectorizer"""
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            
            if not self.chunk_metadata or len(self.chunk_metadata) == 0:
                logger.warning("No chunk metadata available for TF-IDF retriever")
                return
            
            texts = [chunk.get("text", "") for chunk in self.chunk_metadata]
            if not texts or len(texts) == 0:
                logger.warning("No texts available for TF-IDF retriever")
                return
            
            # Create TF-IDF vectors
            self.vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words='english',
                ngram_range=(1, 2)
            )
            self.doc_vectors = self.vectorizer.fit_transform(texts)
            logger.info(f"✅ TF-IDF retriever initialized with {len(texts)} documents")
            
        except Exception as e:
            logger.error(f"Failed to initialize TF-IDF retriever: {e}", exc_info=True)
            self.vectorizer = None
            self.doc_vectors = None
    
    def is_ready(self) -> bool:
        """Check if retriever is ready"""
        return self.vectorizer is not None and self.doc_vectors is not None
    
    def search(
        self,
        query: str,
        top_k: int = 10,
        similarity_threshold: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Search using TF-IDF similarity.
        
        Args:
            query: Search query
            top_k: Number of top results to return
            similarity_threshold: Minimum similarity score threshold
            
        Returns:
            List of result dictionaries with 'chunk_id', 'text', 'metadata', 'similarity_score'
        """
        if not self.is_ready():
            logger.warning("TF-IDF retriever not ready")
            return []
        
        try:
            from sklearn.metrics.pairwise import cosine_similarity
            
            # Transform query to TF-IDF vector
            query_vector = self.vectorizer.transform([query])
            
            # Calculate similarities
            similarities = cosine_similarity(query_vector, self.doc_vectors).flatten()
            
            # Get top-k results
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            results = []
            for idx in top_indices:
                score = float(similarities[idx])
                if score < similarity_threshold:
                    continue
                
                chunk_data = self.chunk_metadata[idx]
                results.append({
                    "chunk_id": chunk_data.get("chunk_id", f"chunk_{idx}"),
                    "text": chunk_data.get("text", ""),
                    "metadata": chunk_data.get("metadata", {}),
                    "similarity_score": score,
                    "section": chunk_data.get("metadata", {}).get("section", "Unknown")
                })
            
            logger.debug(f"TF-IDF search returned {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"TF-IDF search failed: {e}", exc_info=True)
            return []
    
    def get_index_stats(self) -> Dict[str, Any]:
        """Get index statistics (compatible with SemanticRetriever interface)"""
        return {
            "type": "tfidf",
            "ready": self.is_ready(),
            "num_chunks": len(self.chunk_metadata) if self.chunk_metadata else 0,
            "vectorizer_initialized": self.vectorizer is not None,
            "vocabulary_size": len(self.vectorizer.vocabulary_) if self.vectorizer else 0
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get retriever statistics"""
        return self.get_index_stats()

