# Legal Chatbot - Document Service

import json
import logging
from datetime import datetime
from typing import List, Optional, Dict, Any, Tuple
from sqlalchemy.orm import Session
from sqlalchemy import and_

from app.documents.models import Document, DocumentChunk, DocumentStatus, DocumentType
from app.documents.schemas import DocumentCreate, DocumentUpdate
from app.documents.storage import DocumentStorage
from app.documents.parsers import DocumentParser
from ingestion.chunkers.document_chunker import DocumentChunker, ChunkingConfig
from ingestion.loaders.document_loaders import DocumentChunk as IngestionDocumentChunk
from retrieval.embeddings.embedding_generator import EmbeddingGenerator, EmbeddingConfig
from app.core.config import settings
from app.core.errors import NotFoundError, AuthenticationError

logger = logging.getLogger(__name__)


class DocumentService:
    """Document service for processing and managing user documents"""
    
    def __init__(self):
        self.storage = DocumentStorage()
        self.parser = DocumentParser()
        self.chunker = DocumentChunker(ChunkingConfig(
            chunk_size=1000,
            overlap_size=200,
            min_chunk_size=100,
            max_chunk_size=1500,
            preserve_sentences=True
        ))
        
        # Initialize embedding generator
        try:
            embedding_config = EmbeddingConfig(
                model_name=settings.EMBEDDING_MODEL,
                dimension=settings.EMBEDDING_DIMENSION,
                batch_size=settings.EMBEDDING_BATCH_SIZE,
                max_length=512
            )
            self.embedding_gen = EmbeddingGenerator(embedding_config)
            if self.embedding_gen.model is None:
                logger.warning("Embedding generator not available - documents will be stored without embeddings")
                self.embedding_gen = None
        except Exception as e:
            logger.warning(f"Failed to initialize embedding generator: {e}")
            self.embedding_gen = None
    
    def create_document(
        self,
        db: Session,
        user_id: int,
        filename: str,
        content: bytes,
        document_data: Optional[DocumentCreate] = None
    ) -> Document:
        """
        Create and process a new document.
        
        Args:
            db: Database session
            user_id: User ID
            filename: Original filename
            content: File content as bytes
            document_data: Optional document metadata
            
        Returns:
            Created document
        """
        # Validate file size
        if len(content) > settings.MAX_FILE_SIZE:
            raise ValueError(f"File size exceeds maximum allowed size of {settings.MAX_FILE_SIZE} bytes")
        
        # Determine file type
        file_ext = filename.split('.')[-1].lower() if '.' in filename else ''
        if file_ext not in settings.ALLOWED_FILE_TYPES:
            raise ValueError(f"File type '{file_ext}' not allowed. Allowed types: {settings.ALLOWED_FILE_TYPES}")
        
        try:
            file_type = DocumentType(file_ext)
        except ValueError:
            raise ValueError(f"Unsupported file type: {file_ext}")
        
        # Create document record
        document = Document(
            user_id=user_id,
            filename=f"{user_id}_{filename}",
            original_filename=filename,
            file_type=file_type,
            file_size=len(content),
            file_path="",  # Will be set after saving
            status=DocumentStatus.UPLOADED,
            title=document_data.title if document_data else None,
            description=document_data.description if document_data else None,
            jurisdiction=document_data.jurisdiction if document_data else "UK",
            tags=document_data.tags if document_data else None
        )
        
        db.add(document)
        db.flush()  # Get document.id
        
        try:
            # Save file to storage
            storage_path = self.storage.save_document(user_id, document.id, document.filename, content)
            document.file_path = str(storage_path.relative_to(settings.DOCUMENT_STORAGE_PATH))
            document.storage_path = str(storage_path)
            
            # Start processing
            db.commit()
            
            # Process document asynchronously (in real app, use background tasks)
            try:
                self._process_document(db, document, content)
            except Exception as e:
                logger.error(f"Error processing document {document.id}: {e}", exc_info=True)
                document.status = DocumentStatus.FAILED
                document.processing_error = str(e)
                db.commit()
            
            db.refresh(document)
            return document
            
        except Exception as e:
            db.rollback()
            logger.error(f"Error creating document: {e}", exc_info=True)
            raise
    
    def _process_document(self, db: Session, document: Document, content: bytes):
        """Process document: parse, chunk, and index"""
        try:
            document.status = DocumentStatus.PARSING
            db.commit()
            
            # Parse document
            logger.info(f"Parsing document {document.id}: {document.original_filename}")
            parsed_chunks = self.parser.parse_document(
                content=content,
                filename=document.original_filename,
                file_type=document.file_type,
                user_id=document.user_id,
                document_id=document.id
            )
            
            if not parsed_chunks:
                raise ValueError("No content extracted from document")
            
            # Extract full text
            full_text = "\n\n".join([chunk.text for chunk in parsed_chunks])
            document.extracted_text = full_text
            
            document.status = DocumentStatus.PROCESSING
            db.commit()
            
            # Chunk documents
            logger.info(f"Chunking document {document.id}: {len(parsed_chunks)} initial chunks")
            all_chunks = []
            
            for parsed_chunk in parsed_chunks:
                # Convert to ingestion format
                ingestion_chunk = IngestionDocumentChunk(
                    chunk_id=parsed_chunk.chunk_id,
                    text=parsed_chunk.text,
                    metadata=parsed_chunk.metadata,
                    chunk_index=parsed_chunk.chunk_index,
                    start_char=parsed_chunk.start_char,
                    end_char=parsed_chunk.end_char
                )
                
                # Chunk if needed
                chunked = self.chunker.chunk_document(ingestion_chunk)
                all_chunks.extend(chunked)
            
            document.status = DocumentStatus.INDEXING
            db.commit()
            
            # Generate embeddings and store chunks
            logger.info(f"Indexing {len(all_chunks)} chunks for document {document.id}")
            chunks_created = 0
            
            if self.embedding_gen:
                # Generate embeddings in batches
                texts = [chunk.text for chunk in all_chunks]
                
                try:
                    embeddings = self.embedding_gen.generate_embeddings_batch(texts)
                    embedding_dim = self.embedding_gen.get_embedding_dimension()
                except Exception as e:
                    logger.error(f"Error generating embeddings: {e}")
                    embeddings = None
                    embedding_dim = None
            else:
                embeddings = None
                embedding_dim = None
            
            # Store chunks in database
            for idx, chunk in enumerate(all_chunks):
                embedding = embeddings[idx] if embeddings else None
                
                db_chunk = DocumentChunk(
                    document_id=document.id,
                    user_id=document.user_id,
                    chunk_id=chunk.chunk_id,
                    text=chunk.text,
                    chunk_index=chunk.chunk_index,
                    start_char=chunk.start_char,
                    end_char=chunk.end_char,
                    section=chunk.metadata.section,
                    title=chunk.metadata.title,
                    embedding=json.dumps(embedding) if embedding else None,
                    embedding_dimension=embedding_dim,
                    additional_metadata=json.dumps({
                        "source": chunk.metadata.source,
                        "jurisdiction": chunk.metadata.jurisdiction,
                        "document_type": chunk.metadata.document_type,
                        "file_path": chunk.metadata.file_path
                    })
                )
                
                db.add(db_chunk)
                chunks_created += 1
            
            document.chunks_count = chunks_created
            document.status = DocumentStatus.COMPLETED
            document.processed_at = datetime.utcnow()
            
            db.commit()
            logger.info(f"✅ Document {document.id} processed successfully: {chunks_created} chunks created")
            
        except Exception as e:
            logger.error(f"Error processing document {document.id}: {e}", exc_info=True)
            document.status = DocumentStatus.FAILED
            document.processing_error = str(e)
            db.commit()
            raise
    
    def get_document(self, db: Session, document_id: int, user_id: int) -> Document:
        """Get a document by ID (user must own it or be admin)"""
        document = db.query(Document).filter(Document.id == document_id).first()
        
        if not document:
            raise NotFoundError("Document not found")
        
        if document.user_id != user_id:
            # Check if user is admin (this should be done via dependency in routes)
            raise AuthenticationError("Access denied")
        
        return document
    
    def list_documents(
        self,
        db: Session,
        user_id: int,
        skip: int = 0,
        limit: int = 100,
        status: Optional[DocumentStatus] = None,
        file_type: Optional[DocumentType] = None
    ) -> Tuple[List[Document], int]:
        """List documents for a user"""
        query = db.query(Document).filter(Document.user_id == user_id)
        
        if status:
            query = query.filter(Document.status == status)
        if file_type:
            query = query.filter(Document.file_type == file_type)
        
        total = query.count()
        documents = query.order_by(Document.created_at.desc()).offset(skip).limit(limit).all()
        
        return documents, total
    
    def update_document(
        self,
        db: Session,
        document_id: int,
        user_id: int,
        document_data: DocumentUpdate
    ) -> Document:
        """Update document metadata"""
        document = self.get_document(db, document_id, user_id)
        
        update_data = document_data.model_dump(exclude_unset=True)
        for key, value in update_data.items():
            setattr(document, key, value)
        
        document.updated_at = datetime.utcnow()
        db.commit()
        db.refresh(document)
        
        return document
    
    def delete_document(self, db: Session, document_id: int, user_id: int) -> bool:
        """Delete a document and all its chunks"""
        document = self.get_document(db, document_id, user_id)
        
        try:
            # Delete file from storage
            self.storage.delete_document(user_id, document_id, document.filename)
            
            # Delete document (cascades to chunks)
            db.delete(document)
            db.commit()
            
            logger.info(f"Document {document_id} deleted successfully")
            return True
        except Exception as e:
            logger.error(f"Error deleting document {document_id}: {e}", exc_info=True)
            db.rollback()
            raise
    
    def get_document_chunks(
        self,
        db: Session,
        document_id: int,
        user_id: int
    ) -> List[DocumentChunk]:
        """Get all chunks for a document"""
        document = self.get_document(db, document_id, user_id)
        
        chunks = db.query(DocumentChunk).filter(
            DocumentChunk.document_id == document_id,
            DocumentChunk.user_id == user_id
        ).order_by(DocumentChunk.chunk_index).all()
        
        return chunks
    
    def search_user_documents(
        self,
        db: Session,
        user_id: int,
        query: str,
        top_k: int = 10,
        similarity_threshold: float = 0.7
    ) -> List[Dict[str, Any]]:
        """
        Search user's private documents.
        
        Args:
            db: Database session
            user_id: User ID
            query: Search query
            top_k: Number of results to return
            similarity_threshold: Minimum similarity score
            
        Returns:
            List of search results with similarity scores
        """
        if not self.embedding_gen or self.embedding_gen.model is None:
            logger.warning("Embedding generator not available - cannot search documents")
            return []
        
        try:
            # Generate query embedding
            query_embedding = self.embedding_gen.generate_embedding(query)
            embedding_dim = self.embedding_gen.get_embedding_dimension()
            
            # Get all user's document chunks with embeddings
            chunks = db.query(DocumentChunk).filter(
                and_(
                    DocumentChunk.user_id == user_id,
                    DocumentChunk.embedding.isnot(None),
                    DocumentChunk.embedding_dimension == embedding_dim
                )
            ).all()
            
            if not chunks:
                return []
            
            # Calculate similarities
            import numpy as np
            
            results = []
            query_vec = np.array(query_embedding, dtype=np.float32)
            
            for chunk in chunks:
                try:
                    chunk_embedding = json.loads(chunk.embedding)
                    chunk_vec = np.array(chunk_embedding, dtype=np.float32)
                    
                    # Cosine similarity
                    similarity = np.dot(query_vec, chunk_vec) / (
                        np.linalg.norm(query_vec) * np.linalg.norm(chunk_vec)
                    )
                    
                    if similarity >= similarity_threshold:
                        results.append({
                            "chunk_id": chunk.chunk_id,
                            "text": chunk.text,
                            "similarity_score": float(similarity),
                            "document_id": chunk.document_id,
                            "user_id": chunk.user_id,
                            "metadata": {
                                "section": chunk.section,
                                "title": chunk.title,
                                "chunk_index": chunk.chunk_index
                            }
                        })
                except Exception as e:
                    logger.warning(f"Error processing chunk {chunk.id}: {e}")
                    continue
            
            # Sort by similarity and return top_k
            results.sort(key=lambda x: x["similarity_score"], reverse=True)
            return results[:top_k]
            
        except Exception as e:
            logger.error(f"Error searching user documents: {e}", exc_info=True)
            return []

