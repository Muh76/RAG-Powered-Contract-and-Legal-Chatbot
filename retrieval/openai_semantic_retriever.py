# OpenAI Semantic Retriever Wrapper
# Wraps OpenAI embeddings to work with AdvancedHybridRetriever (no PyTorch needed)

import numpy as np
import faiss
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class OpenAISemanticRetriever:
    """
    Semantic retriever using OpenAI embeddings and FAISS.
    Compatible with AdvancedHybridRetriever interface.
    NO PYTORCH REQUIRED - uses OpenAI API directly.
    """
    
    def __init__(
        self,
        embedding_gen,  # OpenAIEmbeddingGenerator instance
        faiss_index,    # FAISS index
        chunk_metadata: List[Dict[str, Any]]
    ):
        """
        Initialize OpenAI semantic retriever.
        
        Args:
            embedding_gen: OpenAIEmbeddingGenerator instance
            faiss_index: FAISS index loaded from disk
            chunk_metadata: List of chunk dictionaries
        """
        self.embedding_gen = embedding_gen
        self.faiss_index = faiss_index
        self.chunk_metadata = chunk_metadata
        self.embedding_dimension = embedding_gen.dimension if hasattr(embedding_gen, 'dimension') else 1536
        
        logger.info(f"✅ OpenAI Semantic Retriever initialized (dimension: {self.embedding_dimension})")
    
    def is_ready(self) -> bool:
        """Check if retriever is ready"""
        return (
            self.embedding_gen is not None and
            self.faiss_index is not None and
            len(self.chunk_metadata) > 0
        )
    
    def search(
        self,
        query: str,
        top_k: int = 10,
        similarity_threshold: float = 0.0,
        metadata_filter: Optional[Any] = None  # MetadataFilter
    ) -> List[Dict[str, Any]]:
        """
        Search using OpenAI embeddings and FAISS.
        
        Args:
            query: Search query
            top_k: Number of results to return
            similarity_threshold: Minimum similarity score
            metadata_filter: Optional metadata filter (applied post-search for now)
            
        Returns:
            List of result dictionaries
        """
        if not self.is_ready():
            logger.warning("OpenAI Semantic Retriever not ready")
            return []
        
        try:
            # Generate query embedding using OpenAI
            query_embedding_list = self.embedding_gen.generate_embedding(query)
            if not query_embedding_list:
                logger.warning("Failed to generate query embedding")
                return []
            
            query_embedding = np.array(query_embedding_list, dtype=np.float32)
            
            # Normalize embedding
            norm = np.linalg.norm(query_embedding)
            if norm == 0 or np.isnan(norm) or np.isinf(norm):
                logger.warning("Query embedding has invalid norm")
                return []
            
            query_normalized = query_embedding / norm
            query_normalized = np.ascontiguousarray(query_normalized, dtype=np.float32)
            
            # Search FAISS index
            k = min(top_k * 2, self.faiss_index.ntotal)  # Get more results for filtering
            similarities, indices = self.faiss_index.search(
                query_normalized.reshape(1, -1),
                k
            )
            
            # Format results
            results = []
            for idx, similarity in zip(indices[0], similarities[0]):
                if idx == -1 or idx >= len(self.chunk_metadata):
                    continue
                
                if similarity < similarity_threshold:
                    continue
                
                chunk_data = self.chunk_metadata[idx]
                
                # Apply metadata filter if provided
                if metadata_filter:
                    metadata = chunk_data.get("metadata", {})
                    if not metadata_filter.matches(metadata):
                        continue
                
                results.append({
                    "chunk_id": chunk_data.get("chunk_id", f"chunk_{idx}"),
                    "text": chunk_data.get("text", ""),
                    "metadata": chunk_data.get("metadata", {}),
                    "similarity_score": float(similarity),
                    "section": chunk_data.get("metadata", {}).get("section", "Unknown")
                })
            
            # Return top_k after filtering
            return results[:top_k]
            
        except Exception as e:
            logger.error(f"OpenAI semantic search failed: {e}", exc_info=True)
            return []

