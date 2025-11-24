# Legal Chatbot - Configuration Settings

from pydantic_settings import BaseSettings
from typing import List, Optional
import os


class Settings(BaseSettings):
    """Application settings"""
    
    # API Configuration
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_WORKERS: int = 1
    API_RELOAD: bool = True
    
    # Database Configuration
    DATABASE_URL: str = "postgresql://postgres:password@localhost:5432/legal_chatbot"
    REDIS_URL: str = "redis://localhost:6379/0"
    
    # Vector Database
    VECTOR_DB_URL: str = "http://localhost:6333"
    VECTOR_DB_COLLECTION: str = "legal_documents"
    
    # OpenAI Configuration
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_MODEL: str = "gpt-4-turbo-preview"
    OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-small"  # Default: fast and cheap
    
    # Embedding Configuration
    # Primary: Use OpenAI embeddings (no PyTorch required - eliminates segfaults!)
    USE_OPENAI_EMBEDDINGS: bool = True  # Set to False to use sentence-transformers (PyTorch)
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"  # Fallback if OpenAI not available
    EMBEDDING_DIMENSION: int = 1536  # text-embedding-3-small default (1536D), or 384 for all-MiniLM-L6-v2
    EMBEDDING_BATCH_SIZE: int = 32
    
    # Retrieval Configuration
    TOP_K_RETRIEVAL: int = 10
    SIMILARITY_THRESHOLD: float = 0.7
    RERANK_TOP_K: int = 5
    ENABLE_RERANKING: bool = True  # Enable cross-encoder reranking for better results
    RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    RERANKER_BATCH_SIZE: int = 32
    
    # Hybrid Search Configuration
    HYBRID_SEARCH_BM25_WEIGHT: float = 0.5  # Increased from 0.4 - BM25 is good for exact matches
    HYBRID_SEARCH_SEMANTIC_WEIGHT: float = 0.5  # Decreased from 0.6 - balance with BM25
    HYBRID_SEARCH_FUSION_STRATEGY: str = "rrf"  # "rrf" or "weighted"
    HYBRID_SEARCH_TOP_K_BM25: int = 30  # Increased from 20 - retrieve more candidates for better fusion
    HYBRID_SEARCH_TOP_K_SEMANTIC: int = 30  # Increased from 20 - retrieve more candidates for better fusion
    HYBRID_SEARCH_TOP_K_FINAL: int = 15  # Increased from 10 - return more results for better coverage
    HYBRID_SEARCH_RRF_K: int = 60  # RRF parameter (standard value)
    
    # Security Configuration
    SECRET_KEY: str = "your-secret-key-change-in-production"
    JWT_SECRET_KEY: str = "your-jwt-secret-key-change-in-production"
    JWT_ALGORITHM: str = "HS256"
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    JWT_REFRESH_TOKEN_EXPIRE_DAYS: int = 7
    
    # OAuth2 Configuration
    OAUTH_GOOGLE_CLIENT_ID: Optional[str] = None
    OAUTH_GOOGLE_CLIENT_SECRET: Optional[str] = None
    OAUTH_GITHUB_CLIENT_ID: Optional[str] = None
    OAUTH_GITHUB_CLIENT_SECRET: Optional[str] = None
    OAUTH_MICROSOFT_CLIENT_ID: Optional[str] = None
    OAUTH_MICROSOFT_CLIENT_SECRET: Optional[str] = None
    OAUTH_MICROSOFT_TENANT_ID: str = "common"  # "common" for multi-tenant, specific tenant ID for single-tenant
    OAUTH_REDIRECT_URI: str = "http://localhost:8501/auth/callback"  # Default redirect URI
    
    # CORS Configuration
    CORS_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:8501"]
    CORS_METHODS: List[str] = ["GET", "POST", "PUT", "DELETE"]
    CORS_HEADERS: List[str] = ["*"]
    
    # Logging Configuration
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "json"
    LOG_FILE: str = "logs/legal_chatbot.log"
    
    # Monitoring Configuration
    PROMETHEUS_PORT: int = 9090
    GRAFANA_PORT: int = 3000
    
    # File Upload Configuration
    MAX_FILE_SIZE: int = 10485760  # 10MB
    ALLOWED_FILE_TYPES: List[str] = ["pdf", "docx", "txt"]
    DOCUMENT_STORAGE_PATH: str = "data/documents"  # Base path for document storage
    
    # Rate Limiting
    RATE_LIMIT_REQUESTS: int = 100
    RATE_LIMIT_WINDOW: int = 3600  # 1 hour
    
    # Privacy Configuration
    PII_REDACTION: bool = True
    DATA_RETENTION_DAYS: int = 90
    ANONYMIZE_LOGS: bool = True
    
    # Development Configuration
    DEBUG: bool = True
    ENVIRONMENT: str = "development"
    
    class Config:
        env_file = ".env"
        case_sensitive = True


# Create settings instance
settings = Settings()
