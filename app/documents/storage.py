# Legal Chatbot - Document Storage Service

import os
import shutil
import hashlib
from pathlib import Path
from typing import Optional, BinaryIO
import logging
from datetime import datetime

from app.core.config import settings

logger = logging.getLogger(__name__)


class DocumentStorage:
    """Document storage service for file system storage"""
    
    def __init__(self, base_path: Optional[str] = None):
        """
        Initialize document storage.
        
        Args:
            base_path: Base path for document storage. Defaults to settings.DOCUMENT_STORAGE_PATH
        """
        self.base_path = Path(base_path or settings.DOCUMENT_STORAGE_PATH)
        self.base_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Document storage initialized at: {self.base_path}")
    
    def get_user_directory(self, user_id: int) -> Path:
        """Get storage directory for a specific user"""
        user_dir = self.base_path / f"user_{user_id}"
        user_dir.mkdir(parents=True, exist_ok=True)
        return user_dir
    
    def get_document_path(self, user_id: int, document_id: int, filename: str) -> Path:
        """Get full storage path for a document"""
        user_dir = self.get_user_directory(user_id)
        return user_dir / f"{document_id}_{filename}"
    
    def save_document(
        self,
        user_id: int,
        document_id: int,
        filename: str,
        content: bytes
    ) -> Path:
        """
        Save document to storage.
        
        Args:
            user_id: User ID
            document_id: Document ID
            filename: Original filename
            content: File content as bytes
            
        Returns:
            Path to saved file
        """
        file_path = self.get_document_path(user_id, document_id, filename)
        
        try:
            with open(file_path, 'wb') as f:
                f.write(content)
            
            logger.info(f"Document saved: {file_path} ({len(content)} bytes)")
            return file_path
        except Exception as e:
            logger.error(f"Error saving document {filename}: {e}")
            raise
    
    def read_document(self, user_id: int, document_id: int, filename: str) -> bytes:
        """
        Read document from storage.
        
        Args:
            user_id: User ID
            document_id: Document ID
            filename: Filename
            
        Returns:
            File content as bytes
        """
        file_path = self.get_document_path(user_id, document_id, filename)
        
        try:
            with open(file_path, 'rb') as f:
                content = f.read()
            return content
        except FileNotFoundError:
            logger.error(f"Document not found: {file_path}")
            raise
        except Exception as e:
            logger.error(f"Error reading document {filename}: {e}")
            raise
    
    def delete_document(self, user_id: int, document_id: int, filename: str) -> bool:
        """
        Delete document from storage.
        
        Args:
            user_id: User ID
            document_id: Document ID
            filename: Filename
            
        Returns:
            True if deleted successfully
        """
        file_path = self.get_document_path(user_id, document_id, filename)
        
        try:
            if file_path.exists():
                file_path.unlink()
                logger.info(f"Document deleted: {file_path}")
                return True
            else:
                logger.warning(f"Document not found for deletion: {file_path}")
                return False
        except Exception as e:
            logger.error(f"Error deleting document {filename}: {e}")
            raise
    
    def get_file_hash(self, content: bytes) -> str:
        """Get SHA256 hash of file content"""
        return hashlib.sha256(content).hexdigest()
    
    def get_file_size(self, user_id: int, document_id: int, filename: str) -> int:
        """Get file size in bytes"""
        file_path = self.get_document_path(user_id, document_id, filename)
        if file_path.exists():
            return file_path.stat().st_size
        return 0

