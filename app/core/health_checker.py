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
    
    async def check_database(self) -> Dict[str, Any]:
        """Check PostgreSQL database health"""
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
                conn = psycopg2.connect(
                    settings.DATABASE_URL,
                    connect_timeout=5,
                )
                conn.close()
                response_time = (time.time() - start_time) * 1000
                
                result = {
                    "status": "healthy",
                    "message": "Database connection successful",
                    "response_time_ms": round(response_time, 2),
                }
            
            self._cache[cache_key] = (time.time(), result)
            return result
            
        except Exception as e:
            result = {
                "status": "unhealthy",
                "message": f"Database connection failed: {str(e)}",
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
            if not REDIS_AVAILABLE:
                result = {
                    "status": "unknown",
                    "message": "redis not available",
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
        """Check Qdrant vector store health"""
        cache_key = "vector_store"
        if cache_key in self._cache:
            cached_time, cached_result = self._cache[cache_key]
            if (time.time() - cached_time) < self._cache_ttl:
                return cached_result
        
        try:
            if not QDRANT_AVAILABLE:
                result = {
                    "status": "unknown",
                    "message": "qdrant-client not available",
                    "response_time_ms": 0,
                }
            else:
                start_time = time.time()
                from qdrant_client import QdrantClient
                client = QdrantClient(url=settings.VECTOR_DB_URL, timeout=5)
                # Try to get collections to verify connection
                collections = client.get_collections()
                response_time = (time.time() - start_time) * 1000
                
                result = {
                    "status": "healthy",
                    "message": "Vector store connection successful",
                    "response_time_ms": round(response_time, 2),
                    "collections_count": len(collections.collections) if collections else 0,
                }
            
            self._cache[cache_key] = (time.time(), result)
            return result
            
        except Exception as e:
            result = {
                "status": "unhealthy",
                "message": f"Vector store connection failed: {str(e)}",
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
                client.models.list(limit=1)  # Minimal check
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

