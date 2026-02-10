"""
Legal Chatbot - Health Checker with Dependency Monitoring
Phase 4.2: Monitoring and Observability
"""

import time
from typing import Dict, Any, Optional
from datetime import datetime
from loguru import logger
import asyncio

try:
    import psycopg2
    PSYCOPG2_AVAILABLE = True
except ImportError:
    PSYCOPG2_AVAILABLE = False

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

try:
    import qdrant_client
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

from app.core.config import settings


class HealthChecker:
    """Health checker for dependencies and services"""
    
    def __init__(self):
        self._cache = {}
        self._cache_ttl = 30  # Cache results for 30 seconds
    
    def _db_error_message(self, e: Exception) -> str:
        """Return a clearer, actionable message for common DB failures."""
        s = str(e).lower()
        if "connection" in s and ("refused" in s or "not permitted" in s or "could not connect" in s):
            return (
                "PostgreSQL not reachable (is it running on host:port?). "
                f"Original: {e}"
            )
        if "password authentication failed" in s or "authentication failed" in s:
            return (
                "Authentication failed (check user/password in DATABASE_URL). "
                f"Original: {e}"
            )
        if 'database "legal_chatbot" does not exist' in s:
            return (
                "Database 'legal_chatbot' does not exist. "
                "Create it: CREATE DATABASE legal_chatbot; "
                f"Original: {e}"
            )
        return f"Database connection failed: {e}"

    async def check_database(self) -> Dict[str, Any]:
        """Check PostgreSQL database health via connect + SELECT 1."""
        cache_key = "database"
        if cache_key in self._cache:
            cached_time, cached_result = self._cache[cache_key]
            if (time.time() - cached_time) < self._cache_ttl:
                return cached_result
        
        try:
            if not PSYCOPG2_AVAILABLE:
                result = {
                    "status": "unknown",
                    "message": "psycopg2 not available",
                    "response_time_ms": 0,
                }
            else:
                start_time = time.time()
                conn = None
                try:
                    conn = psycopg2.connect(
                        settings.DATABASE_URL,
                        connect_timeout=5,
                    )
                    cur = conn.cursor()
                    try:
                        cur.execute("SELECT 1")
                        cur.fetchone()
                    finally:
                        cur.close()
                finally:
                    if conn is not None:
                        conn.close()
                response_time = (time.time() - start_time) * 1000
                result = {
                    "status": "healthy",
                    "message": "Database connection and query OK",
                    "response_time_ms": round(response_time, 2),
                }
            
            self._cache[cache_key] = (time.time(), result)
            return result
            
        except Exception as e:
            msg = self._db_error_message(e)
            result = {
                "status": "unhealthy",
                "message": msg,
                "response_time_ms": 0,
            }
            logger.warning(f"Database health check failed: {e}")
            self._cache[cache_key] = (time.time(), result)
            return result
    
    async def check_redis(self) -> Dict[str, Any]:
        """Check Redis cache health"""
        cache_key = "redis"
        if cache_key in self._cache:
            cached_time, cached_result = self._cache[cache_key]
            if (time.time() - cached_time) < self._cache_ttl:
                return cached_result
        
        try:
            if not REDIS_AVAILABLE or not (settings.REDIS_URL or "").strip():
                result = {
                    "status": "unknown",
                    "message": "redis not available (REDIS_URL not set)",
                    "response_time_ms": 0,
                }
            else:
                start_time = time.time()
                r = redis.from_url(settings.REDIS_URL, socket_connect_timeout=5)
                r.ping()
                response_time = (time.time() - start_time) * 1000
                
                result = {
                    "status": "healthy",
                    "message": "Redis connection successful",
                    "response_time_ms": round(response_time, 2),
                }
            
            self._cache[cache_key] = (time.time(), result)
            return result
            
        except Exception as e:
            result = {
                "status": "unhealthy",
                "message": f"Redis connection failed: {str(e)}",
                "response_time_ms": 0,
            }
            logger.warning(f"Redis health check failed: {e}")
            self._cache[cache_key] = (time.time(), result)
            return result
    
    async def check_vector_store(self) -> Dict[str, Any]:
        """Check FAISS vector store health (system uses FAISS, not Qdrant)"""
        cache_key = "vector_store"
        if cache_key in self._cache:
            cached_time, cached_result = self._cache[cache_key]
            if (time.time() - cached_time) < self._cache_ttl:
                return cached_result
        
        try:
            start_time = time.time()
            from pathlib import Path
            
            # Check if FAISS index file exists (system uses FAISS, not Qdrant)
            possible_paths = [
                Path("data/faiss_index.bin"),
                Path("data/processed/faiss_index.bin"),
                Path("notebooks/phase1/data/faiss_index.bin"),
                Path.cwd() / "data" / "faiss_index.bin",
            ]
            
            faiss_found = any(path.exists() for path in possible_paths)
            response_time = (time.time() - start_time) * 1000
            
            if faiss_found:
                result = {
                    "status": "healthy",
                    "message": "FAISS index file found",
                    "response_time_ms": round(response_time, 2),
                }
            else:
                # Check if Qdrant is available (optional, for future use)
                if QDRANT_AVAILABLE and hasattr(settings, 'VECTOR_DB_URL') and settings.VECTOR_DB_URL:
                    try:
                        from qdrant_client import QdrantClient
                        client = QdrantClient(url=settings.VECTOR_DB_URL, timeout=5)
                        collections = client.get_collections()
                        result = {
                            "status": "healthy",
                            "message": "Qdrant vector store connection successful",
                            "response_time_ms": round(response_time, 2),
                            "collections_count": len(collections.collections) if collections else 0,
                        }
                    except Exception as qdrant_error:
                        result = {
                            "status": "unknown",
                            "message": f"FAISS index not found and Qdrant unavailable: {str(qdrant_error)}",
                            "response_time_ms": round(response_time, 2),
                        }
                else:
                    result = {
                        "status": "unknown",
                        "message": "FAISS index file not found (system may use in-memory index)",
                        "response_time_ms": round(response_time, 2),
                    }
            
            self._cache[cache_key] = (time.time(), result)
            return result
            
        except Exception as e:
            result = {
                "status": "unknown",
                "message": f"Vector store check failed: {str(e)}",
                "response_time_ms": 0,
            }
            logger.warning(f"Vector store health check failed: {e}")
            self._cache[cache_key] = (time.time(), result)
            return result
    
    async def check_llm_api(self) -> Dict[str, Any]:
        """Check OpenAI API health"""
        cache_key = "llm_api"
        if cache_key in self._cache:
            cached_time, cached_result = self._cache[cache_key]
            if (time.time() - cached_time) < self._cache_ttl:
                return cached_result
        
        try:
            if not OPENAI_AVAILABLE or not settings.OPENAI_API_KEY:
                result = {
                    "status": "unknown",
                    "message": "OpenAI API key not configured",
                    "response_time_ms": 0,
                }
            else:
                start_time = time.time()
                # Simple API key validation by checking models endpoint
                client = openai.OpenAI(api_key=settings.OPENAI_API_KEY, timeout=10)
                list(client.models.list())  # Minimal check - list() doesn't accept limit parameter
                response_time = (time.time() - start_time) * 1000
                
                result = {
                    "status": "healthy",
                    "message": "OpenAI API connection successful",
                    "response_time_ms": round(response_time, 2),
                }
            
            self._cache[cache_key] = (time.time(), result)
            return result
            
        except Exception as e:
            result = {
                "status": "unhealthy",
                "message": f"OpenAI API check failed: {str(e)}",
                "response_time_ms": 0,
            }
            logger.warning(f"OpenAI API health check failed: {e}")
            self._cache[cache_key] = (time.time(), result)
            return result
    
    async def check_all_dependencies(self) -> Dict[str, Any]:
        """Check all dependencies in parallel"""
        database_check = self.check_database()
        redis_check = self.check_redis()
        vector_store_check = self.check_vector_store()
        llm_api_check = self.check_llm_api()
        
        results = await asyncio.gather(
            database_check,
            redis_check,
            vector_store_check,
            llm_api_check,
            return_exceptions=True,
        )
        
        return {
            "database": results[0] if not isinstance(results[0], Exception) else {
                "status": "error",
                "message": str(results[0]),
            },
            "redis": results[1] if not isinstance(results[1], Exception) else {
                "status": "error",
                "message": str(results[1]),
            },
            "vector_store": results[2] if not isinstance(results[2], Exception) else {
                "status": "error",
                "message": str(results[2]),
            },
            "llm_api": results[3] if not isinstance(results[3], Exception) else {
                "status": "error",
                "message": str(results[3]),
            },
        }


# Global health checker instance
health_checker = HealthChecker()

