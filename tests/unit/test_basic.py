# Legal Chatbot - Basic Tests

import pytest
import asyncio
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch

from app.api.main import app
from app.models.schemas import ChatRequest, ChatResponse, ChatMode


client = TestClient(app)


class TestHealthEndpoint:
    """Test health check endpoint"""
    
    def test_health_check(self):
        """Test health check endpoint"""
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"


class TestChatEndpoint:
    """Test chat endpoint"""
    
    @patch('app.api.routes.chat.call_chat_api')
    def test_chat_request_validation(self, mock_chat):
        """Test chat request validation"""
        # Test empty query
        response = client.post("/api/v1/chat", json={"query": ""})
        assert response.status_code == 422
        
        # Test query too long
        long_query = "a" * 1001
        response = client.post("/api/v1/chat", json={"query": long_query})
        assert response.status_code == 422
    
    @patch('app.api.routes.chat.call_chat_api')
    def test_chat_mode_validation(self, mock_chat):
        """Test chat mode validation"""
        mock_chat.return_value = {
            "answer": "Test response",
            "sources": [],
            "safety": {"is_safe": True, "flags": [], "confidence": 1.0, "reasoning": ""},
            "metrics": {"retrieval_time_ms": 100, "generation_time_ms": 500, "total_time_ms": 600, "retrieval_score": 0.8, "answer_relevance_score": 0.9},
            "confidence_score": 0.8,
            "legal_jurisdiction": "UK"
        }
        
        # Test valid modes
        for mode in ["public", "solicitor"]:
            response = client.post("/api/v1/chat", json={
                "query": "What is contract law?",
                "mode": mode
            })
            assert response.status_code == 200
    
    @patch('app.api.routes.chat.call_chat_api')
    def test_chat_top_k_validation(self, mock_chat):
        """Test top_k parameter validation"""
        mock_chat.return_value = {
            "answer": "Test response",
            "sources": [],
            "safety": {"is_safe": True, "flags": [], "confidence": 1.0, "reasoning": ""},
            "metrics": {"retrieval_time_ms": 100, "generation_time_ms": 500, "total_time_ms": 600, "retrieval_score": 0.8, "answer_relevance_score": 0.9},
            "confidence_score": 0.8,
            "legal_jurisdiction": "UK"
        }
        
        # Test valid top_k values
        for top_k in [1, 10, 20]:
            response = client.post("/api/v1/chat", json={
                "query": "What is contract law?",
                "top_k": top_k
            })
            assert response.status_code == 200
        
        # Test invalid top_k values
        for top_k in [0, 21]:
            response = client.post("/api/v1/chat", json={
                "query": "What is contract law?",
                "top_k": top_k
            })
            assert response.status_code == 422


class TestModels:
    """Test Pydantic models"""
    
    def test_chat_request_model(self):
        """Test ChatRequest model"""
        # Valid request
        request = ChatRequest(
            query="What is contract law?",
            mode=ChatMode.PUBLIC,
            top_k=10
        )
        assert request.query == "What is contract law?"
        assert request.mode == ChatMode.PUBLIC
        assert request.top_k == 10
    
    def test_chat_response_model(self):
        """Test ChatResponse model"""
        response = ChatResponse(
            answer="Contract law governs agreements between parties.",
            sources=[],
            safety={"is_safe": True, "flags": [], "confidence": 1.0, "reasoning": ""},
            metrics={"retrieval_time_ms": 100, "generation_time_ms": 500, "total_time_ms": 600, "retrieval_score": 0.8, "answer_relevance_score": 0.9},
            confidence_score=0.8,
            legal_jurisdiction="UK"
        )
        assert response.answer == "Contract law governs agreements between parties."
        assert response.confidence_score == 0.8
        assert response.legal_jurisdiction == "UK"


class TestConfiguration:
    """Test configuration settings"""
    
    def test_settings_loading(self):
        """Test settings loading"""
        from app.core.config import settings
        
        assert settings.API_HOST == "0.0.0.0"
        assert settings.API_PORT == 8000
        assert settings.TOP_K_RETRIEVAL == 10
        assert settings.SIMILARITY_THRESHOLD == 0.7


if __name__ == "__main__":
    pytest.main([__file__])
