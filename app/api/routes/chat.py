# Legal Chatbot - Chat Route

from fastapi import APIRouter, HTTPException
from app.models.schemas import ChatRequest, ChatResponse, Source, SafetyReport, LatencyAndScores
from datetime import datetime
import time

router = APIRouter()


async def call_chat_api(request: ChatRequest) -> ChatResponse:
    """Mock chat API implementation for Phase 1"""
    start_time = time.time()
    
    # Mock response for Phase 1
    mock_answer = f"This is a mock response for your query: '{request.query}'. In Phase 1, we'll implement the actual RAG pipeline with UK legal corpus."
    
    mock_sources = [
        Source(
            chunk_id="mock_chunk_1",
            title="Mock Legal Document",
            url="https://example.com/mock-doc",
            text_snippet="This is a mock legal text snippet for demonstration purposes.",
            similarity_score=0.85,
            metadata={"source": "mock", "jurisdiction": "UK"}
        )
    ]
    
    mock_safety = SafetyReport(
        is_safe=True,
        flags=[],
        confidence=0.95,
        reasoning="Query appears to be a legal question"
    )
    
    end_time = time.time()
    total_time = (end_time - start_time) * 1000
    
    mock_metrics = LatencyAndScores(
        retrieval_time_ms=50.0,
        generation_time_ms=total_time - 50.0,
        total_time_ms=total_time,
        retrieval_score=0.85,
        answer_relevance_score=0.80
    )
    
    return ChatResponse(
        answer=mock_answer,
        sources=mock_sources,
        safety=mock_safety,
        metrics=mock_metrics,
        confidence_score=0.80,
        legal_jurisdiction="UK"
    )


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Chat endpoint for legal queries"""
    try:
        response = await call_chat_api(request)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat service error: {str(e)}")
