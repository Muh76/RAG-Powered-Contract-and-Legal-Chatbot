# Legal Chatbot - Document Schemas

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum


class DocumentStatus(str, Enum):
    """Document processing status"""
    UPLOADED = "uploaded"
    PARSING = "parsing"
    PROCESSING = "processing"
    INDEXING = "indexing"
    COMPLETED = "completed"
    FAILED = "failed"


class DocumentType(str, Enum):
    """Document types"""
    PDF = "pdf"
    DOCX = "docx"
    TXT = "txt"
    MD = "md"


class DocumentBase(BaseModel):
    """Base document schema"""
    title: Optional[str] = None
    description: Optional[str] = None
    jurisdiction: Optional[str] = "UK"
    tags: Optional[str] = None


class DocumentCreate(DocumentBase):
    """Document creation schema"""
    filename: str
    file_type: DocumentType
    file_size: int


class DocumentUpdate(BaseModel):
    """Document update schema"""
    title: Optional[str] = None
    description: Optional[str] = None
    tags: Optional[str] = None


class DocumentChunkResponse(BaseModel):
    """Document chunk response schema"""
    id: int
    chunk_id: str
    text: str
    chunk_index: int
    start_char: int
    end_char: int
    section: Optional[str] = None
    title: Optional[str] = None
    
    class Config:
        from_attributes = True


class DocumentResponse(BaseModel):
    """Document response schema"""
    id: int
    user_id: int
    filename: str
    original_filename: str
    file_type: DocumentType
    file_size: int
    title: Optional[str] = None
    description: Optional[str] = None
    status: DocumentStatus
    chunks_count: int
    processing_error: Optional[str] = None
    jurisdiction: Optional[str] = None
    tags: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    processed_at: Optional[datetime] = None
    
    class Config:
        from_attributes = True


class DocumentListResponse(BaseModel):
    """Document list response schema"""
    documents: List[DocumentResponse]
    total: int
    page: int
    page_size: int


class DocumentDetailResponse(DocumentResponse):
    """Document detail response with chunks"""
    chunks: List[DocumentChunkResponse] = Field(default_factory=list)

