# Legal Chatbot - Explainability and Source Highlighting
# Phase 2: Advanced RAG Explainability

import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class RetrievalExplanation:
    """Explanation for why a document was retrieved"""
    chunk_id: str
    text: str
    similarity_score: float
    bm25_score: Optional[float] = None
    semantic_score: Optional[float] = None
    rerank_score: Optional[float] = None
    bm25_rank: Optional[int] = None
    semantic_rank: Optional[int] = None
    rerank_rank: Optional[int] = None
    final_rank: int = 1
    matched_terms: List[str] = None
    matched_text_spans: List[Tuple[int, int]] = None
    explanation: str = ""
    confidence: float = 0.0


class ExplainabilityAnalyzer:
    """
    Analyzes retrieval results to provide explainability and source highlighting.
    
    Provides:
    1. Matched term highlighting in source documents
    2. Explanation of why documents were retrieved
    3. Confidence score breakdown
    4. Retrieval path visualization
    """
    
    def __init__(self, case_sensitive: bool = False):
        """
        Initialize explainability analyzer.
        
        Args:
            case_sensitive: Whether to perform case-sensitive matching (default: False)
        """
        self.case_sensitive = case_sensitive
    
    def extract_query_terms(self, query: str) -> List[str]:
        """
        Extract meaningful terms from query.
        
        Args:
            query: Search query text
            
        Returns:
            List of extracted terms
        """
        # Simple term extraction (can be enhanced with NLP)
        # Remove common stop words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'as', 'is', 'are', 'was', 'were', 'be',
            'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
            'should', 'could', 'what', 'when', 'where', 'why', 'how', 'this',
            'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they'
        }
        
        # Extract words (alphanumeric sequences)
        words = re.findall(r'\b\w+\b', query.lower() if not self.case_sensitive else query)
        
        # Filter stop words and short words
        terms = [w for w in words if w not in stop_words and len(w) > 2]
        
        return terms
    
    def highlight_matched_terms(
        self,
        text: str,
        query: str,
        highlight_tag: str = "**"
    ) -> Tuple[str, List[Tuple[int, int]]]:
        """
        Highlight matched terms in text.
        
        Args:
            text: Document text
            text: Search query
            highlight_tag: HTML/formatting tag to wrap matches (default: "**" for markdown)
            
        Returns:
            Tuple of (highlighted_text, matched_spans) where spans are (start, end) positions
        """
        query_terms = self.extract_query_terms(query)
        
        if not query_terms:
            return text, []
        
        # Find all matches (case-insensitive)
        matched_spans = []
        text_lower = text.lower() if not self.case_sensitive else text
        
        for term in query_terms:
            pattern = re.escape(term)
            if not self.case_sensitive:
                pattern = f"(?i){pattern}"
            
            for match in re.finditer(pattern, text):
                start, end = match.span()
                matched_spans.append((start, end))
        
        # Sort spans by position
        matched_spans.sort(key=lambda x: x[0])
        
        # Merge overlapping spans
        merged_spans = []
        for start, end in matched_spans:
            if merged_spans and start <= merged_spans[-1][1]:
                # Overlapping or adjacent, merge
                merged_spans[-1] = (merged_spans[-1][0], max(merged_spans[-1][1], end))
            else:
                merged_spans.append((start, end))
        
        # Create highlighted text (working backwards to preserve positions)
        highlighted_text = text
        for start, end in reversed(merged_spans):
            highlighted_text = (
                highlighted_text[:start] +
                f"{highlight_tag}{highlighted_text[start:end]}{highlight_tag}" +
                highlighted_text[end:]
            )
        
        return highlighted_text, merged_spans
    
    def explain_retrieval(
        self,
        result: Dict[str, Any],
        query: str
    ) -> RetrievalExplanation:
        """
        Generate explanation for why a document was retrieved.
        
        Args:
            result: Retrieval result dictionary
            query: Search query
            
        Returns:
            RetrievalExplanation object
        """
        chunk_id = result.get("chunk_id", "")
        text = result.get("text", "")
        similarity_score = result.get("similarity_score", 0.0)
        bm25_score = result.get("bm25_score")
        semantic_score = result.get("semantic_score")
        rerank_score = result.get("rerank_score")
        bm25_rank = result.get("bm25_rank")
        semantic_rank = result.get("semantic_rank")
        rerank_rank = result.get("rerank_rank")
        final_rank = result.get("rank", 1)
        
        # Extract matched terms
        query_terms = self.extract_query_terms(query)
        
        # Find matched terms in text
        text_lower = text.lower() if not self.case_sensitive else text
        matched_terms = [term for term in query_terms if term in text_lower]
        
        # Get matched text spans
        _, matched_spans = self.highlight_matched_terms(text, query)
        
        # Generate explanation
        explanation_parts = []
        
        if bm25_score is not None and bm25_score > 0:
            explanation_parts.append(
                f"Strong keyword match (BM25 score: {bm25_score:.2f})"
            )
            if matched_terms:
                explanation_parts.append(f"Matched terms: {', '.join(matched_terms[:5])}")
        
        if semantic_score is not None and semantic_score > 0.5:
            explanation_parts.append(
                f"High semantic similarity (score: {semantic_score:.3f})"
            )
        
        if rerank_score is not None and rerank_score > 0:
            explanation_parts.append(
                f"Cross-encoder reranking confirmed relevance (score: {rerank_score:.3f})"
            )
        
        if bm25_rank is not None:
            explanation_parts.append(f"BM25 rank: #{bm25_rank}")
        
        if semantic_rank is not None:
            explanation_parts.append(f"Semantic rank: #{semantic_rank}")
        
        if rerank_rank is not None:
            explanation_parts.append(f"After reranking: #{rerank_rank}")
        
        explanation = ". ".join(explanation_parts) if explanation_parts else "Retrieved based on hybrid search"
        
        # Calculate confidence (normalized 0-1)
        confidence = min(1.0, similarity_score)
        if rerank_score is not None:
            # Reranking provides additional confidence signal
            confidence = (confidence + min(1.0, rerank_score)) / 2.0
        
        return RetrievalExplanation(
            chunk_id=chunk_id,
            text=text,
            similarity_score=similarity_score,
            bm25_score=bm25_score,
            semantic_score=semantic_score,
            rerank_score=rerank_score,
            bm25_rank=bm25_rank,
            semantic_rank=semantic_rank,
            rerank_rank=rerank_rank,
            final_rank=final_rank,
            matched_terms=matched_terms,
            matched_text_spans=matched_spans,
            explanation=explanation,
            confidence=confidence
        )
    
    def explain_results(
        self,
        results: List[Dict[str, Any]],
        query: str
    ) -> List[RetrievalExplanation]:
        """
        Generate explanations for all retrieval results.
        
        Args:
            results: List of retrieval result dictionaries
            query: Search query
            
        Returns:
            List of RetrievalExplanation objects
        """
        explanations = []
        for result in results:
            explanation = self.explain_retrieval(result, query)
            explanations.append(explanation)
        
        return explanations
    
    def highlight_sources(
        self,
        results: List[Dict[str, Any]],
        query: str,
        highlight_tag: str = "**"
    ) -> List[Dict[str, Any]]:
        """
        Add highlighted text to retrieval results.
        
        Args:
            results: List of retrieval result dictionaries
            query: Search query
            highlight_tag: HTML/formatting tag for highlighting
            
        Returns:
            List of results with highlighted_text field added
        """
        highlighted_results = []
        
        for result in results:
            text = result.get("text", "")
            highlighted_text, matched_spans = self.highlight_matched_terms(
                text, query, highlight_tag
            )
            
            result_copy = result.copy()
            result_copy["highlighted_text"] = highlighted_text
            result_copy["matched_spans"] = matched_spans
            result_copy["num_matches"] = len(matched_spans)
            
            highlighted_results.append(result_copy)
        
        return highlighted_results

