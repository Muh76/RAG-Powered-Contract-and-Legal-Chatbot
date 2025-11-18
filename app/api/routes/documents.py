# Legal Chatbot - Documents Route

from fastapi import APIRouter, HTTPException, UploadFile, File, Depends
from app.models.schemas import DocumentUploadRequest, DocumentUploadResponse
from typing import List
from app.auth.dependencies import require_solicitor_or_admin
from app.auth.models import User

router = APIRouter()


@router.post("/documents/upload", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    current_user: User = Depends(require_solicitor_or_admin)
):
    """Upload document endpoint (requires Solicitor or Admin role)"""
    try:
        # Mock implementation for Phase 1
        content = await file.read()
        
        # Log upload with user info
        import logging
        logger = logging.getLogger(__name__)
        logger.info(
            f"Document upload: user_id={current_user.id}, "
            f"user_email={current_user.email}, role={current_user.role.value}, "
            f"filename={file.filename}, size={len(content)}"
        )
        
        return DocumentUploadResponse(
            document_id=f"doc_{file.filename}_{len(content)}",
            status="uploaded",
            message=f"Document {file.filename} uploaded successfully",
            chunks_created=5  # Mock value
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload error: {str(e)}")


@router.get("/documents")
async def list_documents(
    current_user: User = Depends(require_solicitor_or_admin)
):
    """List uploaded documents (requires Solicitor or Admin role)"""
    # Mock implementation for Phase 1
    import logging
    logger = logging.getLogger(__name__)
    logger.info(
        f"Document list requested: user_id={current_user.id}, "
        f"user_email={current_user.email}, role={current_user.role.value}"
    )
    
    return {
        "documents": [
            {
                "id": "doc_1",
                "name": "sample_contract.pdf",
                "upload_date": "2024-01-01",
                "status": "processed"
            }
        ]
    }
