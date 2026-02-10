# Legal Chatbot - Database Connection

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool
from typing import Generator, Optional, Any
from app.core.config import settings

# Create engine only when DATABASE_URL is set (no hardcoded credentials)
_db_url = (settings.DATABASE_URL or "").strip()
engine: Optional[Engine] = None
SessionLocal: Any = None

if _db_url:
    engine = create_engine(
        _db_url,
        poolclass=StaticPool if "sqlite" in _db_url else None,
        connect_args={"check_same_thread": False} if "sqlite" in _db_url else {},
        echo=settings.DEBUG,
    )
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

_ERR_DB = "DATABASE_URL environment variable is required. Set it to your PostgreSQL connection string."


def get_db() -> Generator[Session, None, None]:
    """Database dependency for FastAPI. Fails fast if DATABASE_URL is not set."""
    if SessionLocal is None:
        raise RuntimeError(_ERR_DB)
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """Initialize database tables"""
    if engine is None:
        raise RuntimeError(_ERR_DB)
    from app.auth.models import Base
    Base.metadata.create_all(bind=engine)

