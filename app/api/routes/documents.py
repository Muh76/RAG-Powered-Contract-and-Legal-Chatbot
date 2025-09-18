# Legal Chatbot - Documents Route

from fastapi import APIRouter, HTTPException, UploadFile, File
from app.models.schemas import DocumentUploadRequest, DocumentUploadResponse
from typing import List

router = APIRouter()


@router.post("/documents/upload", response_model=DocumentUploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """Upload document endpoint"""
    try:
        # Mock implementation for Phase 1
        content = await file.read()
        
        return DocumentUploadResponse(
            document_id=f"doc_{file.filename}_{len(content)}",
            status="uploaded",
            message=f"Document {file.filename} uploaded successfully",
            chunks_created=5  # Mock value
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload error: {str(e)}")


@router.get("/documents")
async def list_documents():
    """List uploaded documents"""
    # Mock implementation for Phase 1
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
