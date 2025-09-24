# Legal Chatbot - Embeddings and Vector Store

import os
import json
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import logging
from dataclasses import dataclass
import hashlib

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingConfig:
    """Configuration for embeddings"""
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    dimension: int = 384
    batch_size: int = 32
    max_length: int = 512


class EmbeddingGenerator:
    """Generate embeddings for text chunks"""
    
    def __init__(self, config: EmbeddingConfig = None):
        self.config = config or EmbeddingConfig()
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the embedding model"""
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.config.model_name)
            logger.info(f"Loaded embedding model: {self.config.model_name}")
        except ImportError:
            logger.error("sentence-transformers not installed. Install with: pip install sentence-transformers")
            raise
    
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text"""
        try:
            embedding = self.model.encode(text, convert_to_tensor=False)
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            raise
    
    def generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a batch of texts"""
        try:
            embeddings = self.model.encode(
                texts, 
                batch_size=self.config.batch_size,
                convert_to_tensor=False
            )
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"Error generating batch embeddings: {e}")
            raise
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings"""
        return self.config.dimension


class VectorStore:
    """Base class for vector stores"""
    
    def store_embeddings(self, chunks: List[Dict[str, Any]], embeddings: List[List[float]]) -> bool:
        """Store embeddings with metadata"""
        raise NotImplementedError
    
    def search_similar(self, query_embedding: List[float], top_k: int = 10, threshold: float = 0.7) -> List[Dict[str, Any]]:
        """Search for similar embeddings"""
        raise NotImplementedError
    
    def get_embedding(self, chunk_id: str) -> Optional[List[float]]:
        """Get embedding by chunk ID"""
        raise NotImplementedError


class QdrantVectorStore(VectorStore):
    """Qdrant vector store implementation"""
    
    def __init__(self, collection_name: str = "legal_documents", host: str = "localhost", port: int = 6333):
        self.collection_name = collection_name
        self.host = host
        self.port = port
        self.client = None
        self._connect()
    
    def _connect(self):
        """Connect to Qdrant"""
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, VectorParams, PointStruct
            
            self.client = QdrantClient(host=self.host, port=self.port)
            self.PointStruct = PointStruct
            self.Distance = Distance
            self.VectorParams = VectorParams
            
            # Create collection if it doesn't exist
            self._create_collection()
            logger.info(f"Connected to Qdrant at {self.host}:{self.port}")
            
        except ImportError:
            logger.error("qdrant-client not installed. Install with: pip install qdrant-client")
            raise
        except Exception as e:
            logger.error(f"Error connecting to Qdrant: {e}")
            raise
    
    def _create_collection(self):
        """Create collection if it doesn't exist"""
        try:
            collections = self.client.get_collections()
            collection_names = [col.name for col in collections.collections]
            
            if self.collection_name not in collection_names:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=self.VectorParams(
                        size=384,  # Dimension of embeddings
                        distance=self.Distance.COSINE
                    )
                )
                logger.info(f"Created collection: {self.collection_name}")
        except Exception as e:
            logger.error(f"Error creating collection: {e}")
            raise
    
    def store_embeddings(self, chunks: List[Dict[str, Any]], embeddings: List[List[float]]) -> bool:
        """Store embeddings with metadata"""
        try:
            points = []
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                point_id = self._generate_point_id(chunk)
                
                point = self.PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload={
                        "chunk_id": chunk["chunk_id"],
                        "text": chunk["text"],
                        "metadata": chunk["metadata"],
                        "chunk_index": chunk["chunk_index"]
                    }
                )
                points.append(point)
            
            # Batch insert
            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            
            logger.info(f"Stored {len(points)} embeddings in Qdrant")
            return True
            
        except Exception as e:
            logger.error(f"Error storing embeddings: {e}")
            return False
    
    def search_similar(self, query_embedding: List[float], top_k: int = 10, threshold: float = 0.7) -> List[Dict[str, Any]]:
        """Search for similar embeddings"""
        try:
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=top_k,
                score_threshold=threshold
            )
            
            similar_chunks = []
            for result in results:
                similar_chunks.append({
                    "chunk_id": result.payload["chunk_id"],
                    "text": result.payload["text"],
                    "metadata": result.payload["metadata"],
                    "similarity_score": result.score,
                    "chunk_index": result.payload["chunk_index"]
                })
            
            return similar_chunks
            
        except Exception as e:
            logger.error(f"Error searching similar embeddings: {e}")
            return []
    
    def get_embedding(self, chunk_id: str) -> Optional[List[float]]:
        """Get embedding by chunk ID"""
        try:
            # This would require storing embeddings separately or using Qdrant's retrieval
            # For now, return None as we don't have a direct way to get embeddings by ID
            return None
        except Exception as e:
            logger.error(f"Error getting embedding: {e}")
            return None
    
    def _generate_point_id(self, chunk: Dict[str, Any]) -> int:
        """Generate a unique point ID for the chunk"""
        # Use hash of chunk_id to generate consistent ID
        return int(hashlib.md5(chunk["chunk_id"].encode()).hexdigest()[:8], 16)


class SQLiteVectorStore(VectorStore):
    """SQLite-based vector store for development/testing"""
    
    def __init__(self, db_path: str = "data/embeddings.db"):
        self.db_path = db_path
        self.conn = None
        self._connect()
    
    def _connect(self):
        """Connect to SQLite database"""
        try:
            import sqlite3
            import json
            
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
            self.conn = sqlite3.connect(self.db_path)
            self.json = json
            
            # Create table if it doesn't exist
            self._create_table()
            logger.info(f"Connected to SQLite database: {self.db_path}")
            
        except Exception as e:
            logger.error(f"Error connecting to SQLite: {e}")
            raise
    
    def _create_table(self):
        """Create embeddings table"""
        cursor = self.conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS embeddings (
                chunk_id TEXT PRIMARY KEY,
                embedding TEXT,
                text TEXT,
                metadata TEXT,
                chunk_index INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self.conn.commit()
    
    def store_embeddings(self, chunks: List[Dict[str, Any]], embeddings: List[List[float]]) -> bool:
        """Store embeddings with metadata"""
        try:
            cursor = self.conn.cursor()
            
            for chunk, embedding in zip(chunks, embeddings):
                cursor.execute("""
                    INSERT OR REPLACE INTO embeddings 
                    (chunk_id, embedding, text, metadata, chunk_index)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    chunk["chunk_id"],
                    self.json.dumps(embedding),
                    chunk["text"],
                    self.json.dumps(chunk["metadata"]),
                    chunk["chunk_index"]
                ))
            
            self.conn.commit()
            logger.info(f"Stored {len(chunks)} embeddings in SQLite")
            return True
            
        except Exception as e:
            logger.error(f"Error storing embeddings: {e}")
            return False
    
    def search_similar(self, query_embedding: List[float], top_k: int = 10, threshold: float = 0.7) -> List[Dict[str, Any]]:
        """Search for similar embeddings using cosine similarity"""
        try:
            cursor = self.conn.cursor()
            cursor.execute("SELECT chunk_id, embedding, text, metadata, chunk_index FROM embeddings")
            
            results = []
            query_np = np.array(query_embedding)
            
            for row in cursor.fetchall():
                chunk_id, embedding_str, text, metadata_str, chunk_index = row
                embedding = self.json.loads(embedding_str)
                metadata = self.json.loads(metadata_str)
                
                # Calculate cosine similarity
                embedding_np = np.array(embedding)
                similarity = np.dot(query_np, embedding_np) / (np.linalg.norm(query_np) * np.linalg.norm(embedding_np))
                
                if similarity >= threshold:
                    results.append({
                        "chunk_id": chunk_id,
                        "text": text,
                        "metadata": metadata,
                        "similarity_score": float(similarity),
                        "chunk_index": chunk_index
                    })
            
            # Sort by similarity and return top_k
            results.sort(key=lambda x: x["similarity_score"], reverse=True)
            return results[:top_k]
            
        except Exception as e:
            logger.error(f"Error searching similar embeddings: {e}")
            return []
    
    def get_embedding(self, chunk_id: str) -> Optional[List[float]]:
        """Get embedding by chunk ID"""
        try:
            cursor = self.conn.cursor()
            cursor.execute("SELECT embedding FROM embeddings WHERE chunk_id = ?", (chunk_id,))
            row = cursor.fetchone()
            
            if row:
                return self.json.loads(row[0])
            return None
            
        except Exception as e:
            logger.error(f"Error getting embedding: {e}")
            return None


# Test the embedding and vector store
if __name__ == "__main__":
    # Test embedding generation
    print("Testing embedding generation...")
    embedding_gen = EmbeddingGenerator()
    
    test_texts = [
        "This is a test legal document about contract law.",
        "The Sale of Goods Act 1979 governs contracts for the sale of goods.",
        "Employment law covers the rights and duties between employers and employees."
    ]
    
    embeddings = embedding_gen.generate_embeddings_batch(test_texts)
    print(f"Generated {len(embeddings)} embeddings with dimension {len(embeddings[0])}")
    
    # Test vector store
    print("\nTesting SQLite vector store...")
    vector_store = SQLiteVectorStore()
    
    # Create test chunks
    test_chunks = [
        {
            "chunk_id": f"test_chunk_{i}",
            "text": text,
            "metadata": {"title": f"Test Document {i}", "source": "test"},
            "chunk_index": i
        }
        for i, text in enumerate(test_texts)
    ]
    
    # Store embeddings
    success = vector_store.store_embeddings(test_chunks, embeddings)
    print(f"Stored embeddings: {success}")
    
    # Test similarity search
    query_text = "What is contract law?"
    query_embedding = embedding_gen.generate_embedding(query_text)
    
    similar_chunks = vector_store.search_similar(query_embedding, top_k=2)
    print(f"\nFound {len(similar_chunks)} similar chunks:")
    for chunk in similar_chunks:
        print(f"- {chunk['chunk_id']}: {chunk['similarity_score']:.3f}")
        print(f"  Text: {chunk['text'][:50]}...")
        print()


