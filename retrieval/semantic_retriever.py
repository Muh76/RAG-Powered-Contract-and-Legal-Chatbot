# Legal Chatbot - Semantic Retriever
# Phase 2: Module 1.2 - Semantic Retrieval Implementation

import faiss
import pickle
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import logging

from retrieval.embeddings.embedding_generator import EmbeddingGenerator, EmbeddingConfig
from app.core.config import settings

logger = logging.getLogger(__name__)


class SemanticRetriever:
    """
    Semantic retriever using embeddings and FAISS for vector similarity search.
    
    This class provides semantic search capabilities by:
    1. Generating embeddings for queries using SentenceTransformers
    2. Performing vector similarity search using FAISS
    3. Returning top-k most semantically similar chunks
    """
    
    def __init__(
        self,
        faiss_index_path: Optional[Path] = None,
        metadata_path: Optional[Path] = None,
        embedding_config: Optional[EmbeddingConfig] = None,
        project_root: Optional[Path] = None
    ):
        """
        Initialize semantic retriever.
        
        Args:
            faiss_index_path: Path to FAISS index file. If None, will search common locations.
            metadata_path: Path to chunk metadata pickle file. If None, will search common locations.
            embedding_config: Configuration for embedding generator. If None, uses default from settings.
            project_root: Project root directory. If None, will try to detect.
        """
        self.project_root = project_root or Path(__file__).parent.parent.parent
        self.embedding_gen = None
        self.faiss_index = None
        self.chunk_metadata = []
        self.embedding_dimension = None
        
        # Initialize embedding generator
        self._initialize_embedding_generator(embedding_config)
        
        # Load FAISS index and metadata
        self._load_vector_store(faiss_index_path, metadata_path)
    
    def _initialize_embedding_generator(self, embedding_config: Optional[EmbeddingConfig]):
        """Initialize the embedding generator."""
        try:
            if embedding_config is None:
                embedding_config = EmbeddingConfig(
                    model_name=settings.EMBEDDING_MODEL,
                    dimension=settings.EMBEDDING_DIMENSION,
                    batch_size=settings.EMBEDDING_BATCH_SIZE,
                    max_length=512
                )
            
            self.embedding_gen = EmbeddingGenerator(embedding_config)
            
            if self.embedding_gen.model is None:
                raise RuntimeError(
                    "Embedding model failed to load. "
                    "Ensure PyTorch is properly installed and venv is activated."
                )
            
            self.embedding_dimension = self.embedding_gen.get_embedding_dimension()
            logger.info(f"✅ Semantic retriever initialized with embedding dimension: {self.embedding_dimension}")
            
        except Exception as e:
            logger.error(f"Failed to initialize embedding generator: {e}")
            raise RuntimeError(f"Cannot initialize semantic retriever: {e}")
    
    def _load_vector_store(
        self,
        faiss_index_path: Optional[Path],
        metadata_path: Optional[Path]
    ):
        """Load FAISS index and chunk metadata from disk."""
        # If paths provided, use them directly
        if faiss_index_path and metadata_path:
            if faiss_index_path.exists() and metadata_path.exists():
                self._load_index_and_metadata(faiss_index_path, metadata_path)
                return
            else:
                logger.warning(f"Provided paths do not exist: {faiss_index_path}, {metadata_path}")
        
        # Search common locations
        possible_paths = [
            self.project_root / "data" / "faiss_index.bin",
            self.project_root / "data" / "processed" / "faiss_index.bin",
            self.project_root / "notebooks" / "phase1" / "data" / "faiss_index.bin",
            Path("data/faiss_index.bin"),
            Path("notebooks/phase1/data/faiss_index.bin"),
            Path.cwd() / "data" / "faiss_index.bin"
        ]
        
        for path in possible_paths:
            if path.exists():
                metadata_p = path.parent / "chunk_metadata.pkl"
                if metadata_p.exists():
                    self._load_index_and_metadata(path, metadata_p)
                    return
        
        # No index found
        logger.warning("FAISS index not found. Run data ingestion first.")
        logger.warning(f"Searched in: {[str(p) for p in possible_paths]}")
        logger.error("To fix: Run 'python scripts/ingest_data.py' to create the FAISS index")
        self.faiss_index = None
        self.chunk_metadata = []
    
    def _load_index_and_metadata(self, faiss_path: Path, metadata_path: Path):
        """Load FAISS index and metadata from specified paths."""
        try:
            self.faiss_index = faiss.read_index(str(faiss_path))
            with open(metadata_path, "rb") as f:
                self.chunk_metadata = pickle.load(f)
            
            # Validate dimension match
            if self.faiss_index.d != self.embedding_dimension:
                logger.error(
                    f"Dimension mismatch! FAISS index has {self.faiss_index.d}D, "
                    f"but embeddings are {self.embedding_dimension}D. "
                    f"Re-run ingestion with matching model."
                )
                self.faiss_index = None
                self.chunk_metadata = []
                return
            
            logger.info(
                f"✅ Loaded FAISS index with {self.faiss_index.ntotal} vectors "
                f"from {faiss_path}"
            )
            logger.info(f"   Index dimension: {self.faiss_index.d}")
            logger.info(f"   Chunk metadata: {len(self.chunk_metadata)} chunks")
            
        except Exception as e:
            logger.error(f"Error loading vector store: {e}")
            self.faiss_index = None
            self.chunk_metadata = []
            raise
    
    def _generate_query_embedding(self, query: str) -> np.ndarray:
        """
        Generate normalized embedding for a query.
        
        Args:
            query: Query text
            
        Returns:
            Normalized embedding as numpy array (1D, float32)
        """
        if self.embedding_gen is None or self.embedding_gen.model is None:
            raise RuntimeError("Embedding generator not available")
        
        # Generate embedding
        query_embedding_list = self.embedding_gen.generate_embedding(query)
        
        # Validate
        if not query_embedding_list or len(query_embedding_list) == 0:
            raise ValueError("Empty embedding generated")
        
        query_embedding = np.array(query_embedding_list, dtype=np.float32)
        
        if query_embedding.size == 0:
            raise ValueError("Invalid embedding array")
        
        # Validate dimension
        if len(query_embedding) != self.embedding_dimension:
            raise ValueError(
                f"Dimension mismatch: expected {self.embedding_dimension}D, "
                f"got {len(query_embedding)}D"
            )
        
        # Normalize for cosine similarity
        norm = np.linalg.norm(query_embedding)
        if norm == 0 or np.isnan(norm) or np.isinf(norm):
            raise ValueError("Query embedding has zero or invalid norm")
        
        query_normalized = query_embedding / norm
        
        # Ensure contiguous memory for FAISS
        return np.ascontiguousarray(query_normalized, dtype=np.float32)
    
    def search(
        self,
        query: str,
        top_k: int = 10,
        similarity_threshold: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Perform semantic search for a query.
        
        Args:
            query: Query text to search for
            top_k: Number of top results to return
            similarity_threshold: Minimum similarity score (0.0 to 1.0)
            
        Returns:
            List of dictionaries containing:
            - chunk_id: Unique identifier for the chunk
            - text: Chunk text content
            - metadata: Chunk metadata (title, source, section, etc.)
            - similarity_score: Cosine similarity score (0.0 to 1.0)
            - rank: Rank of the result (1-indexed)
        """
        # Check if index is loaded
        if self.faiss_index is None or len(self.chunk_metadata) == 0:
            logger.error("Vector store not loaded. Cannot search.")
            return []
        
        # Validate top_k
        if top_k <= 0:
            logger.warning(f"Invalid top_k={top_k}, using default top_k=10")
            top_k = 10
        
        top_k = min(top_k, len(self.chunk_metadata))
        
        try:
            # Generate query embedding
            query_embedding = self._generate_query_embedding(query)
            
            # Reshape for FAISS (1, dimension)
            query_matrix = query_embedding.reshape(1, -1)
            
            # Validate query matrix
            if query_matrix.shape[0] != 1 or query_matrix.shape[1] != self.faiss_index.d:
                raise ValueError(
                    f"Invalid query matrix shape: {query_matrix.shape}, "
                    f"expected (1, {self.faiss_index.d})"
                )
            
            # Perform FAISS search
            # FAISS IndexFlatIP uses inner product, which equals cosine similarity for normalized vectors
            scores, indices = self.faiss_index.search(query_matrix, k=top_k)
            
            # Format results
            results = []
            for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), 1):
                # Validate index
                if idx < 0 or idx >= len(self.chunk_metadata):
                    continue
                
                # Apply similarity threshold
                score_value = float(score)
                if score_value < similarity_threshold:
                    continue
                
                # Validate score
                if np.isnan(score_value) or np.isinf(score_value):
                    score_value = 0.0
                
                # Get chunk data
                chunk_data = self.chunk_metadata[idx]
                
                results.append({
                    "chunk_id": chunk_data.get("chunk_id", f"chunk_{idx}"),
                    "text": chunk_data.get("text", ""),
                    "metadata": chunk_data.get("metadata", {}),
                    "similarity_score": score_value,
                    "rank": rank,
                    "section": chunk_data.get("metadata", {}).get("section", "Unknown"),
                    "title": chunk_data.get("metadata", {}).get("title", "Unknown"),
                    "source": chunk_data.get("metadata", {}).get("source", "Unknown"),
                    "jurisdiction": chunk_data.get("metadata", {}).get("jurisdiction", "Unknown")
                })
            
            logger.debug(f"Semantic search returned {len(results)} results for query: {query[:50]}...")
            return results
            
        except Exception as e:
            logger.error(f"Error during semantic search: {e}", exc_info=True)
            return []
    
    def search_batch(
        self,
        queries: List[str],
        top_k: int = 10,
        similarity_threshold: float = 0.0
    ) -> List[List[Dict[str, Any]]]:
        """
        Perform semantic search for multiple queries in batch.
        
        Args:
            queries: List of query texts
            top_k: Number of top results per query
            similarity_threshold: Minimum similarity score
            
        Returns:
            List of result lists (one per query)
        """
        results = []
        for query in queries:
            query_results = self.search(query, top_k=top_k, similarity_threshold=similarity_threshold)
            results.append(query_results)
        return results
    
    def get_index_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the loaded index.
        
        Returns:
            Dictionary with index statistics
        """
        if self.faiss_index is None:
            return {
                "index_loaded": False,
                "num_vectors": 0,
                "dimension": 0,
                "num_chunks": 0
            }
        
        return {
            "index_loaded": True,
            "num_vectors": self.faiss_index.ntotal,
            "dimension": self.faiss_index.d,
            "num_chunks": len(self.chunk_metadata),
            "embedding_dimension": self.embedding_dimension,
            "index_type": type(self.faiss_index).__name__
        }
    
    def is_ready(self) -> bool:
        """Check if retriever is ready to perform searches."""
        return (
            self.faiss_index is not None and
            len(self.chunk_metadata) > 0 and
            self.embedding_gen is not None and
            self.embedding_gen.model is not None
        )

