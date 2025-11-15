# Legal Chatbot - Advanced Hybrid Retriever
# Phase 2: Module 2 & 3 - Hybrid Search with Metadata Filtering

import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
from collections import defaultdict
from pathlib import Path
import pickle
import logging

from retrieval.bm25_retriever import BM25Retriever
from retrieval.semantic_retriever import SemanticRetriever
from retrieval.metadata_filter import MetadataFilter, FilterOperator
from app.core.config import settings

logger = logging.getLogger(__name__)


class FusionStrategy:
    """Fusion strategy types"""
    RRF = "rrf"  # Reciprocal Rank Fusion
    WEIGHTED = "weighted"  # Weighted combination
    INTERPOLATION = "interpolation"  # Linear interpolation (alias for weighted)


class AdvancedHybridRetriever:
    """
    Advanced hybrid retriever combining BM25 + Semantic search with metadata filtering.
    
    This class provides:
    1. BM25 keyword-based search
    2. Semantic search using embeddings
    3. Metadata filtering (pre-filter or post-filter)
    4. Fusion strategies (RRF, weighted)
    5. Top-k final results
    """
    
    def __init__(
        self,
        bm25_retriever: BM25Retriever,
        semantic_retriever: SemanticRetriever,
        chunk_metadata: Optional[List[Dict[str, Any]]] = None,
        bm25_weight: float = None,
        semantic_weight: float = None,
        fusion_strategy: str = None,
        rrf_k: int = None
    ):
        """
        Initialize advanced hybrid retriever.
        
        Args:
            bm25_retriever: BM25 retriever instance
            semantic_retriever: Semantic retriever instance
            chunk_metadata: Optional chunk metadata list for matching results (if None, will try to load from semantic retriever)
            bm25_weight: Weight for BM25 scores in weighted fusion (default: from settings)
            semantic_weight: Weight for semantic scores in weighted fusion (default: from settings)
            fusion_strategy: Fusion strategy ("rrf" or "weighted", default: from settings)
            rrf_k: RRF parameter k (default: from settings)
        """
        self.bm25_retriever = bm25_retriever
        self.semantic_retriever = semantic_retriever
        self.chunk_metadata = chunk_metadata
        
        # Try to get chunk_metadata from semantic retriever if not provided
        if self.chunk_metadata is None and hasattr(semantic_retriever, 'chunk_metadata'):
            self.chunk_metadata = semantic_retriever.chunk_metadata
        
        # Fusion configuration
        self.bm25_weight = bm25_weight or settings.HYBRID_SEARCH_BM25_WEIGHT
        self.semantic_weight = semantic_weight or settings.HYBRID_SEARCH_SEMANTIC_WEIGHT
        self.fusion_strategy = fusion_strategy or settings.HYBRID_SEARCH_FUSION_STRATEGY
        self.rrf_k = rrf_k or settings.HYBRID_SEARCH_RRF_K
        
        # Validate weights sum to 1.0 (for weighted fusion)
        if self.fusion_strategy in [FusionStrategy.WEIGHTED, FusionStrategy.INTERPOLATION]:
            total_weight = self.bm25_weight + self.semantic_weight
            if abs(total_weight - 1.0) > 0.01:
                logger.warning(f"Weights don't sum to 1.0 ({total_weight:.2f}), normalizing...")
                self.bm25_weight /= total_weight
                self.semantic_weight /= total_weight
        
        logger.info(
            f"AdvancedHybridRetriever initialized: "
            f"fusion={self.fusion_strategy}, "
            f"bm25_weight={self.bm25_weight:.2f}, "
            f"semantic_weight={self.semantic_weight:.2f}"
        )
    
    @classmethod
    def from_chunk_metadata(
        cls,
        chunk_metadata: List[Dict[str, Any]],
        semantic_retriever: Optional[SemanticRetriever] = None,
        bm25_k1: float = 1.2,
        bm25_b: float = 0.75,
        **kwargs
    ) -> 'AdvancedHybridRetriever':
        """
        Create AdvancedHybridRetriever from chunk metadata.
        
        Args:
            chunk_metadata: List of chunk dictionaries with 'text' and 'metadata' keys
            semantic_retriever: Optional semantic retriever (if None, creates new one)
            bm25_k1: BM25 k1 parameter
            bm25_b: BM25 b parameter
            **kwargs: Additional arguments for AdvancedHybridRetriever initialization
            
        Returns:
            AdvancedHybridRetriever instance
        """
        # Extract documents for BM25
        documents = [chunk.get("text", "") for chunk in chunk_metadata]
        
        # Initialize BM25 retriever
        bm25_retriever = BM25Retriever(documents, k1=bm25_k1, b=bm25_b)
        
        # Initialize semantic retriever if not provided
        if semantic_retriever is None:
            semantic_retriever = SemanticRetriever()
        
        # Create hybrid retriever
        return cls(
            bm25_retriever=bm25_retriever,
            semantic_retriever=semantic_retriever,
            chunk_metadata=chunk_metadata,
            **kwargs
        )
    
    def search(
        self,
        query: str,
        top_k: int = None,
        metadata_filter: Optional[MetadataFilter] = None,
        pre_filter: bool = True,
        similarity_threshold: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Perform hybrid search combining BM25 + Semantic search.
        
        Args:
            query: Search query text
            top_k: Number of top results to return (default: from settings)
            metadata_filter: Optional metadata filter to apply
            pre_filter: If True, filter before search; if False, filter after fusion (default: True)
            similarity_threshold: Minimum similarity score for semantic search (default: 0.0)
            
        Returns:
            List of dictionaries containing:
            - chunk_id: Unique identifier for the chunk
            - text: Chunk text content
            - metadata: Chunk metadata
            - similarity_score: Final fused score
            - bm25_score: BM25 score (if available)
            - semantic_score: Semantic similarity score (if available)
            - rank: Rank of the result (1-indexed)
        """
        top_k = top_k or settings.HYBRID_SEARCH_TOP_K_FINAL
        
        # Get expanded top_k for fusion (retrieve more, then fuse and return top_k)
        top_k_bm25 = settings.HYBRID_SEARCH_TOP_K_BM25
        top_k_semantic = settings.HYBRID_SEARCH_TOP_K_SEMANTIC
        
        # Pre-filter: Apply metadata filter before search
        bm25_indices_to_search = None
        semantic_indices_to_search = None
        
        if pre_filter and metadata_filter and not metadata_filter.is_empty() and self.chunk_metadata:
            # Get matching indices for pre-filtering
            matching_indices = metadata_filter.filter_indices(self.chunk_metadata)
            if matching_indices:
                bm25_indices_to_search = set(matching_indices)
                semantic_indices_to_search = matching_indices
                logger.debug(f"Pre-filtering: {len(matching_indices)}/{len(self.chunk_metadata)} chunks match filters")
            else:
                logger.warning("Pre-filter returned no matching chunks")
                return []
        
        # Step 1: BM25 search
        bm25_results = []
        try:
            bm25_raw = self.bm25_retriever.search(query, top_k=top_k_bm25)
            
            # Convert BM25 results to dict format with metadata
            for rank, (doc_idx, bm25_score) in enumerate(bm25_raw, 1):
                # Apply pre-filter if enabled
                if bm25_indices_to_search is not None and doc_idx not in bm25_indices_to_search:
                    continue
                
                # Get chunk metadata if available
                chunk_data = None
                if self.chunk_metadata and doc_idx < len(self.chunk_metadata):
                    chunk_data = self.chunk_metadata[doc_idx]
                
                doc_text = self.bm25_retriever.get_document(doc_idx)
                if doc_text:
                    chunk_id = chunk_data.get("chunk_id", f"chunk_{doc_idx}") if chunk_data else f"chunk_{doc_idx}"
                    bm25_results.append({
                        "doc_idx": doc_idx,
                        "chunk_id": chunk_id,
                        "text": doc_text,
                        "bm25_score": bm25_score,
                        "bm25_rank": rank,
                        "metadata": chunk_data.get("metadata", {}) if chunk_data else {}
                    })
        except Exception as e:
            logger.error(f"BM25 search failed: {e}")
            bm25_results = []
        
        # Step 2: Semantic search
        semantic_results = []
        try:
            semantic_raw = self.semantic_retriever.search(
                query,
                top_k=top_k_semantic,
                similarity_threshold=similarity_threshold
            )
            
            # Apply pre-filter if enabled
            if semantic_indices_to_search is not None:
                # Filter semantic results by matching indices
                filtered_semantic = []
                for result in semantic_raw:
                    chunk_id = result.get("chunk_id", "")
                    # Extract index from chunk_id (e.g., "chunk_0" -> 0)
                    try:
                        if chunk_id.startswith("chunk_"):
                            idx = int(chunk_id.split("_")[1])
                            if idx in semantic_indices_to_search:
                                filtered_semantic.append(result)
                    except (ValueError, IndexError):
                        pass
                semantic_results = filtered_semantic
            else:
                semantic_results = semantic_raw
                
        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            semantic_results = []
        
        # Step 3: Match BM25 and semantic results by chunk_id
        bm25_by_id = {r["chunk_id"]: r for r in bm25_results if r.get("chunk_id")}
        semantic_by_id = {r["chunk_id"]: r for r in semantic_results if r.get("chunk_id")}
        
        # Also match by doc_idx if chunk_metadata is available
        bm25_by_idx = {r["doc_idx"]: r for r in bm25_results if r.get("doc_idx") is not None}
        semantic_by_idx = {}
        for result in semantic_results:
            chunk_id = result.get("chunk_id", "")
            try:
                if chunk_id.startswith("chunk_"):
                    idx = int(chunk_id.split("_")[1])
                    semantic_by_idx[idx] = result
            except (ValueError, IndexError):
                pass
        
        # Step 4: Get all unique chunks
        all_chunk_ids = set(bm25_by_id.keys()) | set(semantic_by_id.keys())
        all_indices = set(bm25_by_idx.keys()) | set(semantic_by_idx.keys())
        
        # Step 5: Fuse scores
        fused_results = []
        processed_chunks = set()
        
        # Process by chunk_id
        for chunk_id in all_chunk_ids:
            if chunk_id in processed_chunks:
                continue
            
            bm25_result = bm25_by_id.get(chunk_id)
            semantic_result = semantic_by_id.get(chunk_id)
            
            bm25_score = bm25_result.get("bm25_score", 0.0) if bm25_result else 0.0
            bm25_rank = bm25_result.get("bm25_rank") if bm25_result else None
            semantic_score = semantic_result.get("similarity_score", 0.0) if semantic_result else 0.0
            semantic_rank = semantic_result.get("rank") if semantic_result else None
            
            # Fuse scores
            fused_score = self._fuse_scores(
                bm25_score=bm25_score,
                semantic_score=semantic_score,
                bm25_rank=bm25_rank,
                semantic_rank=semantic_rank
            )
            
            # Use semantic result as base (has more metadata) or BM25 result
            base_result = semantic_result if semantic_result else bm25_result
            if not base_result:
                continue
            
            fused_result = {
                "chunk_id": chunk_id,
                "text": base_result.get("text", ""),
                "metadata": base_result.get("metadata", {}),
                "similarity_score": fused_score,
                "bm25_score": bm25_score if bm25_result else None,
                "semantic_score": semantic_score if semantic_result else None,
                "bm25_rank": bm25_rank,
                "semantic_rank": semantic_rank,
                "section": base_result.get("section", "Unknown"),
                "title": base_result.get("title", "Unknown"),
                "source": base_result.get("source", "Unknown"),
                "jurisdiction": base_result.get("jurisdiction", "Unknown")
            }
            
            fused_results.append(fused_result)
            processed_chunks.add(chunk_id)
        
        # Process by index (for chunks without matching chunk_ids)
        for idx in all_indices:
            chunk_id = f"chunk_{idx}"
            if chunk_id in processed_chunks:
                continue
            
            bm25_result = bm25_by_idx.get(idx)
            semantic_result = semantic_by_idx.get(idx)
            
            if not bm25_result and not semantic_result:
                continue
            
            bm25_score = bm25_result.get("bm25_score", 0.0) if bm25_result else 0.0
            bm25_rank = bm25_result.get("bm25_rank") if bm25_result else None
            semantic_score = semantic_result.get("similarity_score", 0.0) if semantic_result else 0.0
            semantic_rank = semantic_result.get("rank") if semantic_result else None
            
            fused_score = self._fuse_scores(
                bm25_score=bm25_score,
                semantic_score=semantic_score,
                bm25_rank=bm25_rank,
                semantic_rank=semantic_rank
            )
            
            base_result = semantic_result if semantic_result else bm25_result
            fused_result = {
                "chunk_id": chunk_id,
                "text": base_result.get("text", ""),
                "metadata": base_result.get("metadata", {}),
                "similarity_score": fused_score,
                "bm25_score": bm25_score if bm25_result else None,
                "semantic_score": semantic_score if semantic_result else None,
                "bm25_rank": bm25_rank,
                "semantic_rank": semantic_rank,
                "section": base_result.get("section", "Unknown"),
                "title": base_result.get("title", "Unknown"),
                "source": base_result.get("source", "Unknown"),
                "jurisdiction": base_result.get("jurisdiction", "Unknown")
            }
            
            fused_results.append(fused_result)
            processed_chunks.add(chunk_id)
        
        # Step 6: Post-filter: Apply metadata filter after fusion
        if not pre_filter and metadata_filter and not metadata_filter.is_empty():
            fused_results = metadata_filter.filter_chunks(fused_results)
        
        # Step 7: Sort by fused score and return top_k
        fused_results.sort(key=lambda x: x["similarity_score"], reverse=True)
        final_results = fused_results[:top_k]
        
        # Add final rank
        for rank, result in enumerate(final_results, 1):
            result["rank"] = rank
        
        logger.debug(
            f"Hybrid search completed: query='{query[:50]}...', "
            f"bm25_results={len(bm25_results)}, "
            f"semantic_results={len(semantic_results)}, "
            f"fused_results={len(fused_results)}, "
            f"final_results={len(final_results)}"
        )
        
        return final_results
    
    def _fuse_scores(
        self,
        bm25_score: float,
        semantic_score: float,
        bm25_rank: Optional[int],
        semantic_rank: Optional[int]
    ) -> float:
        """
        Fuse BM25 and semantic scores using the configured fusion strategy.
        
        Args:
            bm25_score: BM25 score
            semantic_score: Semantic similarity score
            bm25_rank: BM25 rank (1-indexed, None if not in results)
            semantic_rank: Semantic rank (1-indexed, None if not in results)
            
        Returns:
            Fused score
        """
        if self.fusion_strategy == FusionStrategy.RRF:
            return self._reciprocal_rank_fusion(bm25_rank, semantic_rank)
        elif self.fusion_strategy in [FusionStrategy.WEIGHTED, FusionStrategy.INTERPOLATION]:
            return self._weighted_fusion(bm25_score, semantic_score)
        else:
            logger.warning(f"Unknown fusion strategy: {self.fusion_strategy}, using weighted")
            return self._weighted_fusion(bm25_score, semantic_score)
    
    def _reciprocal_rank_fusion(
        self,
        bm25_rank: Optional[int],
        semantic_rank: Optional[int]
    ) -> float:
        """
        Reciprocal Rank Fusion (RRF) formula: score = sum(1 / (k + rank))
        
        Args:
            bm25_rank: BM25 rank (1-indexed, None if not in results)
            semantic_rank: Semantic rank (1-indexed, None if not in results)
            
        Returns:
            RRF score
        """
        rrf_score = 0.0
        
        if bm25_rank is not None:
            rrf_score += 1.0 / (self.rrf_k + bm25_rank)
        
        if semantic_rank is not None:
            rrf_score += 1.0 / (self.rrf_k + semantic_rank)
        
        return rrf_score
    
    def _weighted_fusion(
        self,
        bm25_score: float,
        semantic_score: float
    ) -> float:
        """
        Weighted fusion: combined_score = bm25_weight * normalized_bm25 + semantic_weight * semantic_score
        
        Args:
            bm25_score: BM25 score (may be unnormalized)
            semantic_score: Semantic similarity score (0.0 to 1.0, already normalized)
            
        Returns:
            Weighted fused score
        """
        # Normalize BM25 score to [0, 1] range
        # BM25 scores are typically positive and can be large
        # We'll normalize using a heuristic: divide by max(bm25_score, 10) and clip to [0, 1]
        normalized_bm25 = min(bm25_score / max(bm25_score, 10.0), 1.0) if bm25_score > 0 else 0.0
        
        # Combine normalized scores
        fused_score = (
            self.bm25_weight * normalized_bm25 +
            self.semantic_weight * semantic_score
        )
        
        return fused_score
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the hybrid retriever.
        
        Returns:
            Dictionary with statistics
        """
        return {
            "fusion_strategy": self.fusion_strategy,
            "bm25_weight": self.bm25_weight,
            "semantic_weight": self.semantic_weight,
            "rrf_k": self.rrf_k if self.fusion_strategy == FusionStrategy.RRF else None,
            "bm25_stats": self.bm25_retriever.get_index_stats(),
            "semantic_stats": self.semantic_retriever.get_index_stats(),
            "num_chunks": len(self.chunk_metadata) if self.chunk_metadata else None
        }
