# Legal Chatbot - Documents Route

from fastapi import APIRouter, HTTPException, UploadFile, File, Depends, Query, status
from sqlalchemy.orm import Session
from typing import Optional, List

from app.documents.service import DocumentService
from app.documents.schemas import (
    DocumentResponse, DocumentListResponse, DocumentDetailResponse,
    DocumentCreate, DocumentUpdate, DocumentStatus, DocumentType,
    DocumentChunkResponse
)
from app.auth.dependencies import require_solicitor_or_admin, get_current_active_user
from app.auth.models import User
from app.core.database import get_db
from app.core.config import settings
from app.core.errors import NotFoundError, AuthenticationError
import logging

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/documents/upload", response_model=DocumentResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(
    file: UploadFile = File(...),
    title: Optional[str] = Query(None, description="Document title"),
    description: Optional[str] = Query(None, description="Document description"),
    jurisdiction: Optional[str] = Query("UK", description="Document jurisdiction"),
    tags: Optional[str] = Query(None, description="Comma-separated tags"),
    current_user: User = Depends(require_solicitor_or_admin),
    db: Session = Depends(get_db)
):
    """Upload document endpoint (requires Solicitor or Admin role)"""
    try:
        # Validate file type
        file_ext = file.filename.split('.')[-1].lower() if '.' in file.filename else ''
        if file_ext not in settings.ALLOWED_FILE_TYPES:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"File type '{file_ext}' not allowed. Allowed types: {settings.ALLOWED_FILE_TYPES}"
            )
        
        # Read file content
        content = await file.read()
        
        # Validate file size
        if len(content) > settings.MAX_FILE_SIZE:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"File size ({len(content)} bytes) exceeds maximum allowed size ({settings.MAX_FILE_SIZE} bytes)"
            )
        
        # Create document data
        document_data = DocumentCreate(
            title=title or file.filename,
            description=description,
            jurisdiction=jurisdiction,
            tags=tags
        )
        
        # Create document service
        doc_service = DocumentService()
        
        # Create and process document
        document = doc_service.create_document(
            db=db,
            user_id=current_user.id,
            filename=file.filename,
            content=content,
            document_data=document_data
        )
        
        logger.info(
            f"Document upload: user_id={current_user.id}, "
            f"user_email={current_user.email}, role={current_user.role.value}, "
            f"document_id={document.id}, filename={file.filename}, size={len(content)} bytes"
        )
        
        return DocumentResponse.model_validate(document)
        
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"Upload error for user {current_user.email}: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Upload error: {str(e)}")


@router.get("/documents", response_model=DocumentListResponse)
async def list_documents(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    status: Optional[DocumentStatus] = Query(None),
    file_type: Optional[DocumentType] = Query(None),
    current_user: User = Depends(require_solicitor_or_admin),
    db: Session = Depends(get_db)
):
    """List uploaded documents (requires Solicitor or Admin role)"""
    try:
        doc_service = DocumentService()
        documents, total = doc_service.list_documents(
            db=db,
            user_id=current_user.id,
            skip=skip,
            limit=limit,
            status=status,
            file_type=file_type
        )
        
        logger.info(
            f"Document list requested: user_id={current_user.id}, "
            f"user_email={current_user.email}, role={current_user.role.value}, "
            f"total={total}, returned={len(documents)}"
        )
        
        return DocumentListResponse(
            documents=[DocumentResponse.model_validate(doc) for doc in documents],
            total=total,
            page=skip // limit + 1 if limit > 0 else 1,
            page_size=limit
        )
    except Exception as e:
        logger.error(f"Error listing documents: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.get("/documents/{document_id}", response_model=DocumentDetailResponse)
async def get_document(
    document_id: int,
    current_user: User = Depends(require_solicitor_or_admin),
    db: Session = Depends(get_db)
):
    """Get a document by ID (requires Solicitor or Admin role)"""
    try:
        doc_service = DocumentService()
        document = doc_service.get_document(db, document_id, current_user.id)
        chunks = doc_service.get_document_chunks(db, document_id, current_user.id)
        
        response = DocumentDetailResponse.model_validate(document)
        response.chunks = [DocumentChunkResponse.model_validate(chunk) for chunk in chunks]
        
        return response
    except NotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except AuthenticationError as e:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(e))
    except Exception as e:
        logger.error(f"Error getting document {document_id}: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.put("/documents/{document_id}", response_model=DocumentResponse)
async def update_document(
    document_id: int,
    document_data: DocumentUpdate,
    current_user: User = Depends(require_solicitor_or_admin),
    db: Session = Depends(get_db)
):
    """Update document metadata (requires Solicitor or Admin role)"""
    try:
        doc_service = DocumentService()
        document = doc_service.update_document(db, document_id, current_user.id, document_data)
        return DocumentResponse.model_validate(document)
    except NotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except AuthenticationError as e:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(e))
    except Exception as e:
        logger.error(f"Error updating document {document_id}: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.delete("/documents/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(
    document_id: int,
    current_user: User = Depends(require_solicitor_or_admin),
    db: Session = Depends(get_db)
):
    """Delete a document (requires Solicitor or Admin role)"""
    try:
        doc_service = DocumentService()
        doc_service.delete_document(db, document_id, current_user.id)
        logger.info(f"Document {document_id} deleted by user {current_user.id}")
        return None
    except NotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except AuthenticationError as e:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(e))
    except Exception as e:
        logger.error(f"Error deleting document {document_id}: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.post("/documents/{document_id}/reprocess", response_model=DocumentResponse)
async def reprocess_document(
    document_id: int,
    current_user: User = Depends(require_solicitor_or_admin),
    db: Session = Depends(get_db)
):
    """Reprocess a document (requires Solicitor or Admin role)"""
    try:
        doc_service = DocumentService()
        document = doc_service.get_document(db, document_id, current_user.id)
        
        # Read file content
        content = doc_service.storage.read_document(
            current_user.id,
            document_id,
            document.filename
        )
        
        # Reprocess document
        doc_service._process_document(db, document, content)
        
        db.refresh(document)
        return DocumentResponse.model_validate(document)
    except NotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except AuthenticationError as e:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(e))
    except Exception as e:
        logger.error(f"Error reprocessing document {document_id}: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
