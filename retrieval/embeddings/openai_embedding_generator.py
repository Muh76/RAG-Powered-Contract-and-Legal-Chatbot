# Legal Chatbot - OpenAI Embedding Generator
# No PyTorch required - uses OpenAI API

import os
from typing import List, Optional
import logging
from dataclasses import dataclass
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import time

logger = logging.getLogger(__name__)


@dataclass
class OpenAIEmbeddingConfig:
    """Configuration for OpenAI embeddings"""
    api_key: str
    model: str = "text-embedding-3-large"
    dimension: Optional[int] = 3072  # text-embedding-3-large outputs 3072D
    batch_size: int = 50  # Smaller batches to avoid rate limits
    max_retries: int = 5  # More retries for network issues
    timeout: int = 60  # Longer timeout for large batches
    requests_per_minute: int = 20  # VERY conservative rate limit for large datasets


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
    
    def _get_rate_limit_delay(self, attempt: int, base_delay: float = 1.0) -> float:
        """Calculate exponential backoff delay for rate limits"""
        # Exponential backoff: 1s, 2s, 4s, 8s, max 60s
        delay = min(base_delay * (2 ** attempt), 60.0)
        return delay
    
    @property
    def model(self):
        """Return model info for compatibility with existing code"""
        # Return a dummy object that indicates model is available
        return self if self.config.api_key else None
    
    def get_embedding_dimension(self) -> int:
        """Return embedding dimension for compatibility with DocumentService and search."""
        return self.config.dimension or 3072
    
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text using OpenAI API"""
        if not text or not text.strip():
            logger.warning("Empty text provided for embedding")
            return [0.0] * (self.dimension or 3072)
        
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
            default_dim = self.dimension or 3072
            return [[0.0] * default_dim for _ in texts]
        
        # Prepare request with better SSL handling
        url = "https://api.openai.com/v1/embeddings"
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
            "User-Agent": "Legal-Chatbot/1.0"
        }
        
        # Create session with better retry handling
        session = requests.Session()
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry
        
        # Retry strategy for network errors (NOT 429 rate limits - handle those manually)
        retry_strategy = Retry(
            total=2,  # Fewer automatic retries
            backoff_factor=1.5,
            status_forcelist=[500, 502, 503, 504],  # Exclude 429 - handle manually
            allowed_methods=["POST"],
            respect_retry_after_header=True
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("https://", adapter)
        
        # Process in batches with progress tracking
        all_embeddings = [None] * len(texts)
        batch_size = self.config.batch_size
        total_batches = (len(valid_indices) + batch_size - 1) // batch_size
        current_batch = 0
        start_time = time.time()
        
        # Initial delay only for multi-batch (ingestion). Skipped for single-query runtime (chat)
        # to avoid adding 5s latency to every RAG query. Removed from chat path per requirement.
        if total_batches > 1:
            logger.info("   ⏸️  Waiting 5 seconds before starting to avoid initial rate limit...")
            time.sleep(5.0)
        
        for batch_start in range(0, len(valid_indices), batch_size):
            current_batch += 1
            batch_end = min(batch_start + batch_size, len(valid_indices))
            batch_indices = valid_indices[batch_start:batch_end]
            batch_texts = [valid_texts[i] for i in batch_indices]
            
            # Progress indicator (every 10 batches or every 5%)
            if total_batches > 10:
                progress_pct = (current_batch / total_batches) * 100
                if current_batch == 1 or current_batch % max(1, total_batches // 20) == 0 or current_batch == total_batches:
                    elapsed = time.time() - start_time
                    rate = current_batch / elapsed if elapsed > 0 else 0
                    eta = (total_batches - current_batch) / rate if rate > 0 else 0
                    logger.info(f"   📊 Progress: {current_batch}/{total_batches} batches ({progress_pct:.1f}%) | "
                              f"ETA: {eta/60:.1f} min | Rate: {rate:.2f} batches/sec")
            
            # Retry logic with improved rate limit handling
            for attempt in range(self.config.max_retries):
                try:
                    payload = {
                        "model": self.config.model,
                        "input": batch_texts
                    }
                    
                    # Add dimension if specified (for text-embedding-3 models)
                    if self.config.dimension:
                        payload["dimensions"] = self.config.dimension
                    
                    # Use session for better connection handling
                    response = session.post(
                        url,
                        headers=headers,
                        json=payload,
                        timeout=self.config.timeout,
                        verify=True  # SSL verification
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
                        # Rate limit - use exponential backoff with MUCH longer delays
                        wait_time = self._get_rate_limit_delay(attempt, base_delay=5.0)  # Start with 5s
                        
                        # Check if response has Retry-After header - respect it!
                        retry_after = e.response.headers.get("Retry-After")
                        if retry_after:
                            try:
                                wait_time = max(float(retry_after), wait_time)  # Use the longer of the two
                                logger.info(f"   📋 OpenAI requested wait: {wait_time:.1f}s (respecting Retry-After header)")
                            except:
                                pass
                        
                        logger.warning(f"   ⚠️  Rate limited (batch {current_batch}/{total_batches}), waiting {wait_time:.1f}s before retry (attempt {attempt + 1}/{self.config.max_retries})...")
                        time.sleep(wait_time)
                        # Recreate session to clear connection pool
                        session.close()
                        session = requests.Session()
                        adapter = HTTPAdapter(max_retries=retry_strategy)
                        session.mount("https://", adapter)
                        continue
                    else:
                        logger.error(f"OpenAI API error: {e.response.status_code} - {e.response.text}")
                        raise
                        
                except (requests.exceptions.RequestException, requests.exceptions.SSLError) as e:
                    # Handle SSL errors and other network issues
                    error_type = type(e).__name__
                    if attempt < self.config.max_retries - 1:
                        wait_time = self._get_rate_limit_delay(attempt, base_delay=3.0)  # Longer delay for SSL errors
                        logger.warning(f"Network error ({error_type}), retrying in {wait_time:.1f}s... ({str(e)[:100]})")
                        time.sleep(wait_time)
                        # Recreate session on SSL errors
                        if "SSL" in error_type or "SSLError" in error_type:
                            session.close()
                            session = requests.Session()
                            adapter = HTTPAdapter(max_retries=retry_strategy)
                            session.mount("https://", adapter)
                        continue
                    else:
                        logger.error(f"Failed to generate embeddings after {self.config.max_retries} attempts: {error_type}: {e}")
                        raise
            
            # Longer delay between batches to respect rate limits
            if batch_end < len(valid_indices):
                # Calculate delay to stay under requests_per_minute limit
                # If we have requests_per_minute = 20, that's 3s per request minimum
                min_delay = 60.0 / self.config.requests_per_minute if self.config.requests_per_minute > 0 else 3.0
                # Add a buffer for safety - be extra conservative
                time.sleep(max(min_delay, 3.0))
        
        # Fill in None values with zero vectors (for empty texts)
        default_dim = self.dimension or 3072
        for i, emb in enumerate(all_embeddings):
            if emb is None:
                all_embeddings[i] = [0.0] * default_dim
        
        total_generated = len([e for e in all_embeddings if e])
        elapsed_total = time.time() - start_time
        logger.info(f"✅ Generated {total_generated} embeddings using OpenAI API (took {elapsed_total/60:.1f} minutes)")
        return all_embeddings

