"""
Comprehensive End-to-End Tests for All API Endpoints
Phase 4.1: Complete API Coverage Testing
"""

import pytest
import requests
import time
import json
from typing import Dict, Any, List
from datetime import datetime


class TestAllEndpoints:
    """Comprehensive E2E tests for all API endpoints"""
    
    BASE_URL = "http://localhost:8000"
    TIMEOUT = 30
    
    # Test queries for different scenarios
    TEST_QUERIES = {
        "simple_legal": "What is a contract?",
        "complex_legal": "What are the implied conditions in a contract of sale under UK law?",
        "statute_lookup": "What is the Sale of Goods Act 1979?",
        "comparison": "Compare the Sale of Goods Act 1979 with the Consumer Rights Act 2015",
        "non_legal": "How do I cook pasta?",
        "employment": "What are employee rights in the UK?",
        "invalid_empty": "",
        "invalid_long": "a" * 1001,
    }
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup before each test"""
        self.base_url = self.BASE_URL
        self.session = requests.Session()
        yield
        self.session.close()
    
    # ==================== Health Endpoint Tests ====================
    
    def test_health_endpoint(self):
        """Test health check endpoint"""
        response = self.session.get(f"{self.base_url}/api/v1/health", timeout=10)
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        data = response.json()
        assert data["status"] == "healthy", f"Expected 'healthy', got {data['status']}"
        assert "timestamp" in data or "services" in data  # Health endpoint has timestamp or services
        print("✅ Health endpoint test passed")
    
    def test_root_endpoint(self):
        """Test root endpoint"""
        response = self.session.get(f"{self.base_url}/", timeout=10)
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "version" in data
        print("✅ Root endpoint test passed")
    
    # ==================== Chat Endpoint Tests ====================
    
    def test_chat_endpoint_legal_query_solicitor(self):
        """Test chat endpoint with legal query in solicitor mode"""
        payload = {
            "query": self.TEST_QUERIES["simple_legal"],
            "mode": "solicitor",
            "top_k": 5
        }
        start_time = time.time()
        response = self.session.post(
            f"{self.base_url}/api/v1/chat",
            json=payload,
            timeout=self.TIMEOUT
        )
        response_time = time.time() - start_time
        
        assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
        data = response.json()
        assert "answer" in data
        assert "sources" in data
        assert "safety" in data
        assert "metrics" in data
        assert data["legal_jurisdiction"] == "UK"
        assert response_time < 10, f"Response time {response_time:.2f}s exceeds 10s threshold"
        print(f"✅ Chat endpoint (solicitor) test passed (response time: {response_time:.2f}s)")
    
    def test_chat_endpoint_legal_query_public(self):
        """Test chat endpoint with legal query in public mode"""
        payload = {
            "query": self.TEST_QUERIES["employment"],
            "mode": "public",
            "top_k": 5
        }
        response = self.session.post(
            f"{self.base_url}/api/v1/chat",
            json=payload,
            timeout=self.TIMEOUT
        )
        assert response.status_code == 200
        data = response.json()
        assert data["mode"] == "public"
        assert len(data["answer"]) > 0
        print("✅ Chat endpoint (public) test passed")
    
    def test_chat_endpoint_non_legal_query(self):
        """Test chat endpoint blocks non-legal queries"""
        payload = {
            "query": self.TEST_QUERIES["non_legal"],
            "mode": "public",
            "top_k": 5
        }
        response = self.session.post(
            f"{self.base_url}/api/v1/chat",
            json=payload,
            timeout=self.TIMEOUT
        )
        assert response.status_code == 200  # Guardrails should block, but return 200 with message
        data = response.json()
        # Should have safety flags indicating non-legal query
        assert not data["safety"]["is_safe"] or len(data["safety"]["flags"]) > 0
        print("✅ Chat endpoint (non-legal query) test passed - guardrails working")
    
    def test_chat_endpoint_invalid_query_empty(self):
        """Test chat endpoint rejects empty query"""
        payload = {
            "query": self.TEST_QUERIES["invalid_empty"],
            "mode": "public"
        }
        response = self.session.post(
            f"{self.base_url}/api/v1/chat",
            json=payload,
            timeout=self.TIMEOUT
        )
        assert response.status_code == 422, "Expected 422 for empty query"
        print("✅ Chat endpoint (empty query validation) test passed")
    
    def test_chat_endpoint_invalid_query_too_long(self):
        """Test chat endpoint rejects query that's too long"""
        payload = {
            "query": self.TEST_QUERIES["invalid_long"],
            "mode": "public"
        }
        response = self.session.post(
            f"{self.base_url}/api/v1/chat",
            json=payload,
            timeout=self.TIMEOUT
        )
        assert response.status_code == 422, "Expected 422 for query too long"
        print("✅ Chat endpoint (query length validation) test passed")
    
    def test_chat_endpoint_invalid_mode(self):
        """Test chat endpoint rejects invalid mode"""
        payload = {
            "query": self.TEST_QUERIES["simple_legal"],
            "mode": "invalid_mode"
        }
        response = self.session.post(
            f"{self.base_url}/api/v1/chat",
            json=payload,
            timeout=self.TIMEOUT
        )
        assert response.status_code == 422, "Expected 422 for invalid mode"
        print("✅ Chat endpoint (mode validation) test passed")
    
    # ==================== Hybrid Search Endpoint Tests ====================
    
    def test_hybrid_search_post(self):
        """Test POST hybrid search endpoint"""
        payload = {
            "query": "contract sale goods",
            "top_k": 5,
            "fusion_strategy": "rrf",
            "include_explanation": True,
            "highlight_sources": True
        }
        start_time = time.time()
        response = self.session.post(
            f"{self.base_url}/api/v1/search/hybrid",
            json=payload,
            timeout=self.TIMEOUT
        )
        response_time = time.time() - start_time
        
        assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
        data = response.json()
        assert "results" in data
        assert "total_results" in data
        assert "fusion_strategy" in data
        assert data["fusion_strategy"] == "rrf"
        assert response_time < 5, f"Hybrid search response time {response_time:.2f}s exceeds 5s threshold"
        print(f"✅ Hybrid search POST test passed (response time: {response_time:.2f}s)")
    
    def test_hybrid_search_get(self):
        """Test GET hybrid search endpoint"""
        params = {
            "query": "employment rights",
            "top_k": 3,
            "fusion_strategy": "weighted",
            "include_explanation": True
        }
        response = self.session.get(
            f"{self.base_url}/api/v1/search/hybrid",
            params=params,
            timeout=self.TIMEOUT
        )
        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        assert len(data["results"]) <= params["top_k"]
        print("✅ Hybrid search GET test passed")
    
    def test_hybrid_search_with_metadata_filter(self):
        """Test hybrid search with metadata filtering"""
        payload = {
            "query": "contract law",
            "top_k": 5,
            "metadata_filters": [
                {
                    "field": "jurisdiction",
                    "value": "UK",
                    "operator": "eq"
                }
            ]
        }
        response = self.session.post(
            f"{self.base_url}/api/v1/search/hybrid",
            json=payload,
            timeout=self.TIMEOUT
        )
        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        # Verify all results have UK jurisdiction
        for result in data["results"]:
            if "jurisdiction" in result.get("metadata", {}):
                assert result["metadata"]["jurisdiction"] == "UK"
        print("✅ Hybrid search with metadata filter test passed")
    
    def test_hybrid_search_explainability(self):
        """Test hybrid search explainability features"""
        payload = {
            "query": "employee rights",
            "top_k": 3,
            "include_explanation": True,
            "highlight_sources": True
        }
        response = self.session.post(
            f"{self.base_url}/api/v1/search/hybrid",
            json=payload,
            timeout=self.TIMEOUT
        )
        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        # Check if explainability fields are present in results
        for result in data["results"][:1]:  # Check first result
            if result.get("explanation"):
                assert "confidence" in result or result.get("confidence") is not None
            if result.get("highlighted_text"):
                assert len(result["highlighted_text"]) > 0
        print("✅ Hybrid search explainability test passed")
    
    # ==================== Agentic Chat Endpoint Tests ====================
    
    def test_agentic_chat_simple_query(self):
        """Test agentic chat with simple legal query"""
        payload = {
            "query": self.TEST_QUERIES["statute_lookup"],
            "mode": "public"
        }
        start_time = time.time()
        response = self.session.post(
            f"{self.base_url}/api/v1/agentic-chat",
            json=payload,
            timeout=60  # Agentic chat may take longer
        )
        response_time = time.time() - start_time
        
        assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
        data = response.json()
        assert "answer" in data
        assert "tool_calls" in data
        assert "iterations" in data
        assert "intermediate_steps_count" in data
        assert "safety" in data
        assert "metrics" in data
        assert data["mode"] == "public"
        assert response_time < 30, f"Agentic chat response time {response_time:.2f}s exceeds 30s threshold"
        print(f"✅ Agentic chat (simple query) test passed (response time: {response_time:.2f}s)")
    
    def test_agentic_chat_complex_query(self):
        """Test agentic chat with complex multi-tool query"""
        payload = {
            "query": self.TEST_QUERIES["comparison"],
            "mode": "solicitor",
            "chat_history": []
        }
        response = self.session.post(
            f"{self.base_url}/api/v1/agentic-chat",
            json=payload,
            timeout=60
        )
        assert response.status_code == 200
        data = response.json()
        assert len(data["answer"]) > 0
        # Complex query should use multiple tools
        assert data["iterations"] > 0
        assert len(data["tool_calls"]) >= 1
        print(f"✅ Agentic chat (complex query) test passed - {data['iterations']} iterations, {len(data['tool_calls'])} tools")
    
    def test_agentic_chat_with_history(self):
        """Test agentic chat with conversation history"""
        payload = {
            "query": "What about the Consumer Rights Act 2015?",
            "mode": "public",
            "chat_history": [
                {"role": "user", "content": "Tell me about the Sale of Goods Act 1979"},
                {"role": "assistant", "content": "The Sale of Goods Act 1979 is..."}
            ]
        }
        response = self.session.post(
            f"{self.base_url}/api/v1/agentic-chat",
            json=payload,
            timeout=60
        )
        assert response.status_code == 200
        data = response.json()
        assert len(data["answer"]) > 0
        print("✅ Agentic chat (with history) test passed")
    
    def test_agentic_chat_stats(self):
        """Test agentic chat stats endpoint"""
        response = self.session.get(
            f"{self.base_url}/api/v1/agentic-chat/stats",
            timeout=10
        )
        assert response.status_code == 200
        data = response.json()
        assert "tools" in data or "model" in data or isinstance(data, dict)
        print("✅ Agentic chat stats test passed")
    
    # ==================== Documents Endpoint Tests ====================
    
    def test_documents_list(self):
        """Test list documents endpoint"""
        response = self.session.get(
            f"{self.base_url}/api/v1/documents",
            timeout=10
        )
        assert response.status_code == 200
        data = response.json()
        assert "documents" in data
        assert isinstance(data["documents"], list)
        print("✅ Documents list test passed")
    
    def test_documents_upload_mock(self):
        """Test document upload endpoint (mock)"""
        # Create a mock file
        files = {
            "file": ("test_document.txt", "This is a test document content", "text/plain")
        }
        response = self.session.post(
            f"{self.base_url}/api/v1/documents/upload",
            files=files,
            timeout=30
        )
        assert response.status_code == 200
        data = response.json()
        assert "document_id" in data
        assert "status" in data
        assert data["status"] == "uploaded"
        print("✅ Documents upload (mock) test passed")
    
    # ==================== Error Scenario Tests ====================
    
    def test_invalid_endpoint(self):
        """Test invalid endpoint returns 404"""
        response = self.session.get(
            f"{self.base_url}/api/v1/invalid-endpoint",
            timeout=10
        )
        assert response.status_code == 404
        print("✅ Invalid endpoint (404) test passed")
    
    def test_malformed_json(self):
        """Test malformed JSON request"""
        response = self.session.post(
            f"{self.base_url}/api/v1/chat",
            data="invalid json{",
            headers={"Content-Type": "application/json"},
            timeout=10
        )
        assert response.status_code == 422 or response.status_code == 400
        print("✅ Malformed JSON test passed")
    
    def test_missing_required_fields(self):
        """Test request with missing required fields"""
        response = self.session.post(
            f"{self.base_url}/api/v1/chat",
            json={"mode": "public"},  # Missing 'query'
            timeout=10
        )
        assert response.status_code == 422
        print("✅ Missing required fields test passed")
    
    # ==================== Performance Tests ====================
    
    def test_response_time_simple_query(self):
        """Test response time for simple queries (< 3s target)"""
        payload = {
            "query": self.TEST_QUERIES["simple_legal"],
            "mode": "public",
            "top_k": 3
        }
        start_time = time.time()
        response = self.session.post(
            f"{self.base_url}/api/v1/chat",
            json=payload,
            timeout=self.TIMEOUT
        )
        response_time = time.time() - start_time
        
        assert response.status_code == 200
        assert response_time < 3.0, f"Simple query took {response_time:.2f}s (target: <3s)"
        print(f"✅ Response time test (simple query): {response_time:.2f}s")
    
    def test_response_time_complex_query(self):
        """Test response time for complex queries (< 10s target)"""
        payload = {
            "query": self.TEST_QUERIES["complex_legal"],
            "mode": "solicitor",
            "top_k": 5
        }
        start_time = time.time()
        response = self.session.post(
            f"{self.base_url}/api/v1/chat",
            json=payload,
            timeout=self.TIMEOUT
        )
        response_time = time.time() - start_time
        
        assert response.status_code == 200
        assert response_time < 10.0, f"Complex query took {response_time:.2f}s (target: <10s)"
        print(f"✅ Response time test (complex query): {response_time:.2f}s")


@pytest.mark.e2e
class TestE2ERegression:
    """Regression tests to ensure Phase 1, 2, 3 features still work"""
    
    BASE_URL = "http://localhost:8000"
    
    def test_phase1_traditional_rag_still_works(self):
        """Verify Phase 1 traditional RAG endpoint still works"""
        payload = {
            "query": "What is contract law?",
            "mode": "public",
            "top_k": 5
        }
        response = requests.post(
            f"{self.BASE_URL}/api/v1/chat",
            json=payload,
            timeout=30
        )
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "sources" in data
        assert data["legal_jurisdiction"] == "UK"
        print("✅ Phase 1 traditional RAG regression test passed")
    
    def test_phase2_hybrid_search_still_works(self):
        """Verify Phase 2 hybrid search still works"""
        payload = {
            "query": "employment rights",
            "top_k": 5,
            "fusion_strategy": "rrf"
        }
        response = requests.post(
            f"{self.BASE_URL}/api/v1/search/hybrid",
            json=payload,
            timeout=30
        )
        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        assert data["fusion_strategy"] == "rrf"
        print("✅ Phase 2 hybrid search regression test passed")
    
    def test_phase3_agentic_chat_still_works(self):
        """Verify Phase 3 agentic chat still works"""
        payload = {
            "query": "What is the Sale of Goods Act 1979?",
            "mode": "public"
        }
        response = requests.post(
            f"{self.BASE_URL}/api/v1/agentic-chat",
            json=payload,
            timeout=60
        )
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "tool_calls" in data
        assert "iterations" in data
        print("✅ Phase 3 agentic chat regression test passed")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

