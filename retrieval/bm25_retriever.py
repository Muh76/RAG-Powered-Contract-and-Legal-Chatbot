# Legal Chatbot - BM25 Retriever
# Phase 2: Module 2 - BM25 Retrieval Implementation

import re
import math
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict, Counter
import logging

logger = logging.getLogger(__name__)


class BM25Retriever:
    """
    BM25 (Best Matching 25) retriever for keyword-based search.
    
    BM25 is a ranking function used to estimate the relevance of documents
    to a given search query. It combines term frequency (TF) with inverse
    document frequency (IDF) to score documents.
    """
    
    def __init__(
        self,
        documents: List[str],
        k1: float = 1.2,
        b: float = 0.75,
        stop_words: Optional[set] = None
    ):
        """
        Initialize BM25 retriever.
        
        Args:
            documents: List of document texts to index
            k1: BM25 parameter controlling term frequency saturation (default: 1.2)
            b: BM25 parameter controlling length normalization (default: 0.75)
            stop_words: Set of stop words to filter. If None, uses default English stop words.
        """
        self.documents = documents
        self.k1 = k1
        self.b = b
        
        # Default stop words
        if stop_words is None:
            self.stop_words = {
                'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
                'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did',
                'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those',
                'it', 'its', 'they', 'them', 'their', 'what', 'which', 'who', 'whom', 'where', 'when', 'why',
                'how', 'all', 'each', 'every', 'both', 'few', 'more', 'most', 'other', 'some', 'such', 'no',
                'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 'just', 'now'
            }
        else:
            self.stop_words = stop_words
        
        # Build index
        self._build_index()
        logger.info(f"BM25 index built with {len(self.documents)} documents, vocabulary size: {len(self.doc_freqs)}")
    
    def _preprocess_text(self, text: str) -> List[str]:
        """
        Preprocess text: tokenize, lowercase, remove stop words.
        
        Args:
            text: Input text
            
        Returns:
            List of preprocessed terms
        """
        # Extract words (alphanumeric sequences)
        words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
        # Filter stop words and short words
        words = [word for word in words if word not in self.stop_words and len(word) > 2]
        return words
    
    def _build_index(self):
        """Build BM25 index with term frequencies and document statistics."""
        self.doc_freqs = defaultdict(int)  # Document frequency (how many docs contain term)
        self.doc_terms = []  # Terms per document
        self.doc_lengths = []  # Length of each document in terms
        
        # Process each document
        for doc in self.documents:
            terms = self._preprocess_text(doc)
            self.doc_terms.append(terms)
            self.doc_lengths.append(len(terms))
            
            # Count document frequency for each unique term
            unique_terms = set(terms)
            for term in unique_terms:
                self.doc_freqs[term] += 1
        
        # Calculate average document length
        self.avg_doc_length = np.mean(self.doc_lengths) if self.doc_lengths else 0
        
        # Calculate IDF for each term
        self.idf = {}
        total_docs = len(self.documents)
        for term, doc_freq in self.doc_freqs.items():
            # BM25 IDF formula
            self.idf[term] = math.log((total_docs - doc_freq + 0.5) / (doc_freq + 0.5))
    
    def search(
        self,
        query: str,
        top_k: int = 10
    ) -> List[Tuple[int, float]]:
        """
        Search using BM25 scoring.
        
        Args:
            query: Search query text
            top_k: Number of top results to return
            
        Returns:
            List of tuples (document_index, bm25_score), sorted by score descending
        """
        query_terms = self._preprocess_text(query)
        
        if not query_terms:
            return []
        
        scores = []
        
        # Score each document
        for doc_idx, doc_terms in enumerate(self.doc_terms):
            score = 0.0
            doc_length = self.doc_lengths[doc_idx]
            
            # Count term frequencies in document
            term_counts = Counter(doc_terms)
            
            # Calculate BM25 score for each query term
            for term in query_terms:
                if term in term_counts:
                    tf = term_counts[term]  # Term frequency in document
                    idf = self.idf.get(term, 0.0)  # Inverse document frequency
                    
                    # BM25 formula
                    numerator = tf * (self.k1 + 1)
                    denominator = tf + self.k1 * (1 - self.b + self.b * (doc_length / self.avg_doc_length))
                    
                    score += idf * (numerator / denominator)
            
            # Only include documents with positive scores
            if score > 0:
                scores.append((doc_idx, score))
        
        # Sort by score (descending) and return top_k
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]
    
    def get_document(self, doc_idx: int) -> Optional[str]:
        """
        Get document text by index.
        
        Args:
            doc_idx: Document index
            
        Returns:
            Document text or None if index is invalid
        """
        if 0 <= doc_idx < len(self.documents):
            return self.documents[doc_idx]
        return None
    
    def get_index_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the BM25 index.
        
        Returns:
            Dictionary with index statistics
        """
        return {
            "num_documents": len(self.documents),
            "vocabulary_size": len(self.doc_freqs),
            "avg_doc_length": self.avg_doc_length,
            "k1": self.k1,
            "b": self.b,
            "stop_words_count": len(self.stop_words)
        }

