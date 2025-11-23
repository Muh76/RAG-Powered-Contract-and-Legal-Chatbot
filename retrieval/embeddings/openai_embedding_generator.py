# Legal Chatbot - OpenAI Embedding Generator
# No PyTorch required - uses OpenAI API

import os
from typing import List, Optional
import logging
from dataclasses import dataclass
import requests
import time

logger = logging.getLogger(__name__)


@dataclass
class OpenAIEmbeddingConfig:
    """Configuration for OpenAI embeddings"""
    api_key: str
    model: str = "text-embedding-3-small"  # Default: fast and cheap
    dimension: Optional[int] = None  # None = use model default
    batch_size: int = 100  # OpenAI allows up to 2048
    max_retries: int = 3
    timeout: int = 30


class OpenAIEmbeddingGenerator:
    """Generate embeddings using OpenAI API (no PyTorch required)"""
    
    def __init__(self, config: OpenAIEmbeddingConfig = None):
        self.config = config or OpenAIEmbeddingConfig(api_key=os.getenv("OPENAI_API_KEY", ""))
        self.model_name = self.config.model
        self.dimension = self.config.dimension
        
        if not self.config.api_key:
            raise ValueError("OPENAI_API_KEY is required for OpenAI embeddings")
        
        logger.info(f"✅ OpenAI EmbeddingGenerator initialized (model: {self.config.model})")
        logger.info("✅ No PyTorch required - using OpenAI API")
    
    @property
    def model(self):
        """Return model info for compatibility with existing code"""
        # Return a dummy object that indicates model is available
        return self if self.config.api_key else None
    
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text using OpenAI API"""
        if not text or not text.strip():
            logger.warning("Empty text provided for embedding")
            return [0.0] * (self.dimension or 1536)  # Default dimension
        
        try:
            embeddings = self.generate_embeddings_batch([text])
            return embeddings[0] if embeddings else []
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}", exc_info=True)
            raise
    
    def generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts using OpenAI API"""
        if not texts:
            return []
        
        # Filter empty texts
        valid_texts = [t.strip() if t else "" for t in texts]
        valid_indices = [i for i, t in enumerate(valid_texts) if t]
        
        if not valid_indices:
            logger.warning("No valid texts provided for embedding")
            default_dim = self.dimension or 1536
            return [[0.0] * default_dim for _ in texts]
        
        # Prepare request
        url = "https://api.openai.com/v1/embeddings"
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json"
        }
        
        # Process in batches
        all_embeddings = [None] * len(texts)
        batch_size = self.config.batch_size
        
        for batch_start in range(0, len(valid_indices), batch_size):
            batch_end = min(batch_start + batch_size, len(valid_indices))
            batch_indices = valid_indices[batch_start:batch_end]
            batch_texts = [valid_texts[i] for i in batch_indices]
            
            # Retry logic
            for attempt in range(self.config.max_retries):
                try:
                    payload = {
                        "model": self.config.model,
                        "input": batch_texts
                    }
                    
                    # Add dimension if specified (for text-embedding-3 models)
                    if self.config.dimension:
                        payload["dimensions"] = self.config.dimension
                    
                    response = requests.post(
                        url,
                        headers=headers,
                        json=payload,
                        timeout=self.config.timeout
                    )
                    response.raise_for_status()
                    
                    result = response.json()
                    batch_embeddings = [item["embedding"] for item in result["data"]]
                    
                    # Map embeddings back to original indices
                    for i, embedding in zip(batch_indices, batch_embeddings):
                        all_embeddings[i] = embedding
                    
                    break  # Success, exit retry loop
                    
                except requests.exceptions.HTTPError as e:
                    if e.response.status_code == 429:
                        # Rate limit - wait and retry
                        wait_time = 2 ** attempt
                        logger.warning(f"Rate limited, waiting {wait_time}s before retry...")
                        time.sleep(wait_time)
                        continue
                    else:
                        logger.error(f"OpenAI API error: {e.response.status_code} - {e.response.text}")
                        raise
                        
                except requests.exceptions.RequestException as e:
                    if attempt < self.config.max_retries - 1:
                        wait_time = 2 ** attempt
                        logger.warning(f"Request failed, retrying in {wait_time}s... ({e})")
                        time.sleep(wait_time)
                        continue
                    else:
                        logger.error(f"Failed to generate embeddings after {self.config.max_retries} attempts: {e}")
                        raise
            
            # Small delay between batches to avoid rate limits
            if batch_end < len(valid_indices):
                time.sleep(0.1)
        
        # Fill in None values with zero vectors (for empty texts)
        default_dim = self.dimension or 1536
        for i, emb in enumerate(all_embeddings):
            if emb is None:
                all_embeddings[i] = [0.0] * default_dim
        
        logger.info(f"✅ Generated {len([e for e in all_embeddings if e])} embeddings using OpenAI API")
        return all_embeddings

