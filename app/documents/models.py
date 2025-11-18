# Legal Chatbot - Document Database Models

from sqlalchemy import Column, Integer, String, Boolean, DateTime, ForeignKey, Text, Float
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import ENUM as PostgreSQL_ENUM
from datetime import datetime
import enum

from app.auth.models import Base
from app.auth.models import User


class DocumentStatus(str, enum.Enum):
    """Document processing status"""
    UPLOADED = "uploaded"
    PARSING = "parsing"
    PROCESSING = "processing"
    INDEXING = "indexing"
    COMPLETED = "completed"
    FAILED = "failed"


class DocumentType(str, enum.Enum):
    """Document types"""
    PDF = "pdf"
    DOCX = "docx"
    TXT = "txt"
    MD = "md"


class Document(Base):
    """Document model for user-uploaded documents"""
    __tablename__ = "documents"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    
    # Document metadata
    filename = Column(String(255), nullable=False)
    original_filename = Column(String(255), nullable=False)
    file_type = Column(PostgreSQL_ENUM(DocumentType, name='documenttype', create_type=False, values_callable=lambda obj: [e.value for e in obj]), nullable=False)
    file_size = Column(Integer, nullable=False)  # Size in bytes
    file_path = Column(String(500), nullable=False)  # Storage path
    storage_path = Column(String(500), nullable=True)  # Full storage path
    
    # Document content
    title = Column(String(255), nullable=True)
    description = Column(Text, nullable=True)
    extracted_text = Column(Text, nullable=True)  # Full extracted text
    
    # Processing status
    status = Column(PostgreSQL_ENUM(DocumentStatus, name='documentstatus', create_type=False, values_callable=lambda obj: [e.value for e in obj]), default=DocumentStatus.UPLOADED, nullable=False)
    chunks_count = Column(Integer, default=0)
    processing_error = Column(Text, nullable=True)
    
    # Metadata
    additional_metadata = Column(Text, nullable=True)  # JSON string for additional metadata
    jurisdiction = Column(String(50), default="UK", nullable=True)
    tags = Column(String(500), nullable=True)  # Comma-separated tags
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    processed_at = Column(DateTime, nullable=True)
    
    # Relationships
    user = relationship("User", backref="documents")
    chunks = relationship("DocumentChunk", back_populates="document", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Document(id={self.id}, filename={self.filename}, user_id={self.user_id}, status={self.status.value})>"


class DocumentChunk(Base):
    """Document chunk model for user document chunks"""
    __tablename__ = "document_chunks"
    
    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.id"), nullable=False, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    
    # Chunk content
    chunk_id = Column(String(255), unique=True, nullable=False, index=True)  # Unique chunk identifier
    text = Column(Text, nullable=False)
    chunk_index = Column(Integer, nullable=False)  # Index within document
    start_char = Column(Integer, nullable=False)
    end_char = Column(Integer, nullable=False)
    
    # Embedding and indexing
    embedding = Column(Text, nullable=True)  # JSON string of embedding vector
    embedding_dimension = Column(Integer, nullable=True)
    
    # Metadata
    section = Column(String(255), nullable=True)
    title = Column(String(255), nullable=True)
    additional_metadata = Column(Text, nullable=True)  # JSON string for additional metadata
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    # Relationships
    document = relationship("Document", back_populates="chunks")
    user = relationship("User", backref="document_chunks")
    
    def __repr__(self):
        return f"<DocumentChunk(id={self.id}, chunk_id={self.chunk_id}, document_id={self.document_id}, user_id={self.user_id})>"

