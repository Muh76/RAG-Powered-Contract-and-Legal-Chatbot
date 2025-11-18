# Legal Chatbot - Document Parsers

import io
from pathlib import Path
from typing import List, Optional
import logging

from ingestion.loaders.document_loaders import DocumentChunk, DocumentMetadata
from ingestion.loaders.document_loaders import PDFLoader, TextLoader
from app.documents.models import DocumentType

logger = logging.getLogger(__name__)


class DOCXLoader:
    """DOCX document loader"""
    
    def __init__(self):
        try:
            from docx import Document as DocxDocument
            self.DocxDocument = DocxDocument
        except ImportError:
            logger.error("python-docx not installed. Install with: pip install python-docx")
            raise
    
    def load_documents(self, file_path: str) -> List[DocumentChunk]:
        """Load DOCX documents and return chunks"""
        chunks = []
        
        try:
            doc = self.DocxDocument(file_path)
            
            # Extract text from all paragraphs
            text_parts = []
            for paragraph in doc.paragraphs:
                if paragraph.text.strip():
                    text_parts.append(paragraph.text.strip())
            
            full_text = "\n\n".join(text_parts)
            
            if full_text.strip():
                chunk = DocumentChunk(
                    chunk_id=f"docx_{Path(file_path).stem}_full",
                    text=full_text.strip(),
                    metadata=DocumentMetadata(
                        title=Path(file_path).stem,
                        source="DOCX",
                        jurisdiction="UK",
                        document_type="DOCX",
                        file_path=file_path
                    ),
                    chunk_index=0,
                    start_char=0,
                    end_char=len(full_text)
                )
                chunks.append(chunk)
        except Exception as e:
            logger.error(f"Error loading DOCX {file_path}: {e}")
            
        return chunks
    
    def load_from_bytes(self, content: bytes) -> List[DocumentChunk]:
        """Load DOCX from bytes"""
        import tempfile
        
        with tempfile.NamedTemporaryFile(suffix='.docx', delete=False) as tmp_file:
            tmp_file.write(content)
            tmp_path = tmp_file.name
        
        try:
            chunks = self.load_documents(tmp_path)
        finally:
            # Clean up temp file
            Path(tmp_path).unlink()
        
        return chunks


class DocumentParser:
    """Document parser for different file types"""
    
    def __init__(self):
        self.pdf_loader = PDFLoader()
        self.text_loader = TextLoader()
        self.docx_loader = DOCXLoader()
    
    def parse_document(
        self,
        content: bytes,
        filename: str,
        file_type: DocumentType,
        user_id: int,
        document_id: int
    ) -> List[DocumentChunk]:
        """
        Parse document from bytes.
        
        Args:
            content: File content as bytes
            filename: Original filename
            file_type: Document type
            user_id: User ID
            document_id: Document ID
            
        Returns:
            List of document chunks
        """
        import tempfile
        
        # Create temporary file for parsing
        suffix = {
            DocumentType.PDF: '.pdf',
            DocumentType.DOCX: '.docx',
            DocumentType.TXT: '.txt',
            DocumentType.MD: '.txt'
        }.get(file_type, '.txt')
        
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp_file:
            tmp_file.write(content)
            tmp_path = tmp_file.name
        
        try:
            chunks = []
            
            if file_type == DocumentType.PDF:
                chunks = self.pdf_loader.load_documents(tmp_path)
            elif file_type == DocumentType.DOCX:
                # DOCX can be loaded directly from bytes
                chunks = self.docx_loader.load_from_bytes(content)
            elif file_type in [DocumentType.TXT, DocumentType.MD]:
                chunks = self.text_loader.load_documents(tmp_path)
            else:
                raise ValueError(f"Unsupported file type: {file_type}")
            
            # Update chunk IDs to include user and document info
            for chunk in chunks:
                chunk.chunk_id = f"user_{user_id}_doc_{document_id}_{chunk.chunk_id}"
                chunk.metadata.file_path = filename
            
            return chunks
        finally:
            # Clean up temp file
            if Path(tmp_path).exists():
                Path(tmp_path).unlink()
    
    def extract_text(
        self,
        content: bytes,
        filename: str,
        file_type: DocumentType
    ) -> str:
        """
        Extract plain text from document.
        
        Args:
            content: File content as bytes
            filename: Original filename
            file_type: Document type
            
        Returns:
            Extracted text
        """
        chunks = self.parse_document(content, filename, file_type, user_id=0, document_id=0)
        
        if not chunks:
            return ""
        
        # Combine all chunks into single text
        text_parts = [chunk.text for chunk in chunks]
        return "\n\n".join(text_parts)

