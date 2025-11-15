# Legal Chatbot - Cross-Encoder Reranker
# Phase 2: Advanced Reranking Implementation

import os
import sys
import logging
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import torch

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from app.core.config import settings

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """
    Cross-encoder reranker using sentence-transformers for improved retrieval accuracy.
    
    Cross-encoders provide better accuracy than bi-encoders by encoding query-document
    pairs together, but are slower. Best used for reranking top-k results from hybrid search.
    """
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        batch_size: int = 32,
        device: Optional[str] = None,
        max_length: int = 512
    ):
        """
        Initialize cross-encoder reranker.
        
        Args:
            model_name: Name of the cross-encoder model (default: ms-marco-MiniLM)
            batch_size: Batch size for reranking (default: 32)
            device: Device to use ('cpu', 'cuda', or None for auto-detection)
            max_length: Maximum sequence length (default: 512)
        """
        self.model = None
        self.model_name = model_name or "cross-encoder/ms-marco-MiniLM-L-6-v2"
        self.batch_size = batch_size
        self.max_length = max_length
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        try:
            self._load_model()
        except Exception as e:
            logger.error(f"Failed to load cross-encoder model: {e}")
            logger.warning("Cross-encoder reranking will be disabled. Falling back to no reranking.")
            self.model = None
    
    def _load_model(self):
        """Load cross-encoder model from sentence-transformers"""
        try:
            from sentence_transformers import CrossEncoder
            
            logger.info(f"🔄 Loading cross-encoder model: {self.model_name}")
            self.model = CrossEncoder(
                self.model_name,
                max_length=self.max_length,
                device=self.device
            )
            logger.info(f"✅ Cross-encoder model loaded successfully on {self.device}")
        except ImportError:
            logger.error("sentence-transformers not installed. Install with: pip install sentence-transformers")
            raise
        except Exception as e:
            logger.error(f"Error loading cross-encoder model: {e}")
            raise
    
    def is_ready(self) -> bool:
        """Check if reranker is ready to use"""
        return self.model is not None
    
    def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: Optional[int] = None,
        score_threshold: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Rerank documents based on query-document similarity.
        
        Args:
            query: Search query
            documents: List of document texts to rerank
            top_k: Number of top results to return (None = return all)
            score_threshold: Minimum score threshold (default: 0.0)
            
        Returns:
            List of reranked documents with scores, sorted by relevance
        """
        if not self.is_ready():
            logger.warning("Cross-encoder not available, returning documents unchanged")
            return [
                {
                    "text": doc,
                    "score": 0.0,
                    "rank": i + 1
                }
                for i, doc in enumerate(documents)
            ]
        
        if not documents:
            return []
        
        try:
            # Create query-document pairs for cross-encoder
            pairs = [[query, doc] for doc in documents]
            
            # Get scores from cross-encoder (in batches for efficiency)
            scores = []
            for i in range(0, len(pairs), self.batch_size):
                batch = pairs[i:i + self.batch_size]
                batch_scores = self.model.predict(
                    batch,
                    show_progress_bar=False,
                    convert_to_numpy=True
                )
                scores.extend(batch_scores)
            
            # Convert to list if needed
            if isinstance(scores, np.ndarray):
                scores = scores.tolist()
            
            # Create results with scores
            results = [
                {
                    "text": doc,
                    "score": float(score),
                    "rank": i + 1
                }
                for i, (doc, score) in enumerate(zip(documents, scores))
                if score >= score_threshold
            ]
            
            # Sort by score (descending)
            results.sort(key=lambda x: x["score"], reverse=True)
            
            # Update ranks after sorting
            for i, result in enumerate(results):
                result["rank"] = i + 1
            
            # Return top_k if specified
            if top_k is not None and top_k > 0:
                results = results[:top_k]
            
            logger.debug(f"Reranked {len(documents)} documents, returned {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Error during reranking: {e}", exc_info=True)
            # Return documents unchanged on error
            return [
                {
                    "text": doc,
                    "score": 0.0,
                    "rank": i + 1
                }
                for i, doc in enumerate(documents)
            ]
    
    def rerank_results(
        self,
        query: str,
        retrieval_results: List[Dict[str, Any]],
        top_k: Optional[int] = None,
        score_threshold: float = 0.0,
        preserve_metadata: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Rerank retrieval results (with metadata).
        
        Args:
            query: Search query
            retrieval_results: List of retrieval result dictionaries with 'text' field
            top_k: Number of top results to return (None = return all)
            score_threshold: Minimum score threshold
            preserve_metadata: If True, preserve all metadata fields (default: True)
            
        Returns:
            List of reranked results with rerank_score added
        """
        if not retrieval_results:
            return []
        
        # Extract document texts
        documents = [result.get("text", "") for result in retrieval_results]
        
        # Rerank documents
        reranked = self.rerank(query, documents, top_k=None, score_threshold=score_threshold)
        
        # Map reranked scores back to original results with metadata
        results = []
        doc_text_to_original = {result.get("text", ""): result for result in retrieval_results}
        
        for rerank_result in reranked:
            doc_text = rerank_result["text"]
            rerank_score = rerank_result["score"]
            rerank_rank = rerank_result["rank"]
            
            # Get original result
            original_result = doc_text_to_original.get(doc_text, {"text": doc_text})
            
            # Create new result preserving metadata
            result = {
                "rerank_score": rerank_score,
                "rerank_rank": rerank_rank,
                "original_rank": original_result.get("rank", rerank_rank),
                "similarity_score": original_result.get("similarity_score", rerank_score),
                **original_result  # Preserve all original fields
            }
            
            # Update final similarity score to rerank score (or combine if preferred)
            if not preserve_metadata or "rerank_score" in original_result:
                # If reranking again, use rerank score directly
                result["similarity_score"] = rerank_score
            else:
                # Combine original and rerank scores (weighted average)
                original_score = original_result.get("similarity_score", 0.0)
                # Normalize scores and combine (60% rerank, 40% original)
                result["similarity_score"] = (0.6 * rerank_score) + (0.4 * original_score)
            
            results.append(result)
        
        # Return top_k if specified
        if top_k is not None and top_k > 0:
            results = results[:top_k]
        
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the reranker"""
        return {
            "model_name": self.model_name,
            "is_ready": self.is_ready(),
            "device": self.device,
            "batch_size": self.batch_size,
            "max_length": self.max_length
        }

