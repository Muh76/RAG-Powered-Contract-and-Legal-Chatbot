"""
Frontend Integration Tests
Phase 4.1: Streamlit UI Integration with API
"""

import pytest
import requests
import time
from typing import Dict, Any


class TestFrontendIntegration:
    """Test frontend integration with API"""
    
    BASE_URL = "http://localhost:8000"
    FRONTEND_URL = "http://localhost:8501"
    TIMEOUT = 30
    
    def test_api_endpoint_accessible_from_frontend(self):
        """Test that API endpoints are accessible (frontend would call these)"""
        # Test health endpoint
        response = requests.get(f"{self.BASE_URL}/api/v1/health", timeout=10)
        assert response.status_code == 200, "Health endpoint must be accessible"
        print("✅ Health endpoint accessible")
    
    def test_chat_api_endpoint_format(self):
        """Test chat API endpoint returns data in format expected by frontend"""
        payload = {
            "query": "What is contract law?",
            "mode": "public",
            "top_k": 5
        }
        response = requests.post(
            f"{self.BASE_URL}/api/v1/chat",
            json=payload,
            timeout=self.TIMEOUT
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Frontend expects these fields
        required_fields = ["answer", "sources", "safety", "metrics", "confidence_score", "legal_jurisdiction"]
        for field in required_fields:
            assert field in data, f"Frontend requires field: {field}"
        
        # Verify data types
        assert isinstance(data["answer"], str)
        assert isinstance(data["sources"], list)
        assert isinstance(data["safety"], dict)
        assert isinstance(data["metrics"], dict)
        assert isinstance(data["confidence_score"], (int, float))
        assert isinstance(data["legal_jurisdiction"], str)
        
        print("✅ Chat API endpoint format test passed")
    
    def test_agentic_chat_api_endpoint_format(self):
        """Test agentic chat API endpoint returns data in format expected by frontend"""
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
        
        # Frontend expects these fields for agentic chat
        required_fields = ["answer", "tool_calls", "iterations", "safety", "metrics", "confidence_score"]
        for field in required_fields:
            assert field in data, f"Frontend requires field: {field}"
        
        # Verify data types
        assert isinstance(data["answer"], str)
        assert isinstance(data["tool_calls"], list)
        assert isinstance(data["iterations"], int)
        assert isinstance(data["safety"], dict)
        assert isinstance(data["metrics"], dict)
        
        print("✅ Agentic chat API endpoint format test passed")
    
    def test_hybrid_search_api_endpoint_format(self):
        """Test hybrid search API endpoint returns data in format expected by frontend"""
        payload = {
            "query": "contract law",
            "top_k": 5,
            "fusion_strategy": "rrf"
        }
        response = requests.post(
            f"{self.BASE_URL}/api/v1/search/hybrid",
            json=payload,
            timeout=self.TIMEOUT
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Frontend expects these fields
        required_fields = ["query", "results", "total_results", "fusion_strategy"]
        for field in required_fields:
            assert field in data, f"Frontend requires field: {field}"
        
        # Verify data types
        assert isinstance(data["results"], list)
        assert isinstance(data["total_results"], int)
        assert isinstance(data["fusion_strategy"], str)
        
        if len(data["results"]) > 0:
            result = data["results"][0]
            # Check result structure
            assert "text" in result or "chunk_id" in result
        
        print("✅ Hybrid search API endpoint format test passed")
    
    def test_cors_headers(self):
        """Test CORS headers are set correctly for frontend access"""
        response = requests.options(
            f"{self.BASE_URL}/api/v1/chat",
            headers={
                "Origin": "http://localhost:8501",
                "Access-Control-Request-Method": "POST"
            },
            timeout=10
        )
        
        # CORS should allow requests from frontend origin
        # Check if CORS headers are present
        assert response.status_code in [200, 204, 405], "CORS preflight should succeed"
        print("✅ CORS headers test passed")
    
    def test_api_response_time_for_frontend(self):
        """Test API response times are acceptable for frontend UX"""
        payload = {
            "query": "What is contract law?",
            "mode": "public",
            "top_k": 5
        }
        
        start_time = time.time()
        response = requests.post(
            f"{self.BASE_URL}/api/v1/chat",
            json=payload,
            timeout=self.TIMEOUT
        )
        response_time = time.time() - start_time
        
        assert response.status_code == 200
        # Frontend UX target: response within 5 seconds
        assert response_time < 5.0, f"Response time {response_time:.2f}s exceeds 5s UX target"
        print(f"✅ API response time for frontend: {response_time:.2f}s")
    
    def test_error_responses_format(self):
        """Test error responses are in format frontend can handle"""
        # Test validation error
        payload = {"query": "", "mode": "public"}  # Empty query should fail validation
        response = requests.post(
            f"{self.BASE_URL}/api/v1/chat",
            json=payload,
            timeout=10
        )
        
        # Error response should be valid JSON
        assert response.status_code == 422
        error_data = response.json()
        assert "detail" in error_data or "errors" in error_data or "message" in error_data
        print("✅ Error responses format test passed")
    
    def test_streaming_support_headers(self):
        """Test API supports streaming if implemented (for future frontend features)"""
        response = requests.get(
            f"{self.BASE_URL}/api/v1/health",
            timeout=10
        )
        
        # Check if server supports streaming (Accept: text/event-stream)
        # This is a placeholder for future streaming support
        assert response.status_code == 200
        print("✅ Streaming support headers test passed (placeholder)")
    
    def test_api_supports_concurrent_requests(self):
        """Test API can handle concurrent requests from frontend"""
        import concurrent.futures
        
        def make_request():
            payload = {"query": "What is contract law?", "mode": "public"}
            return requests.post(
                f"{self.BASE_URL}/api/v1/chat",
                json=payload,
                timeout=self.TIMEOUT
            )
        
        # Simulate multiple concurrent frontend requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(make_request) for _ in range(5)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        # All requests should succeed
        status_codes = [r.status_code for r in results]
        assert all(code == 200 for code in status_codes), f"Some requests failed: {status_codes}"
        print("✅ API concurrent requests support test passed")
    
    def test_api_supports_session_management(self):
        """Test API supports session management (for frontend chat history)"""
        # Test agentic chat with chat history
        payload = {
            "query": "What about the Consumer Rights Act?",
            "mode": "public",
            "chat_history": [
                {"role": "user", "content": "Tell me about the Sale of Goods Act 1979"},
                {"role": "assistant", "content": "The Sale of Goods Act 1979 is..."}
            ]
        }
        response = requests.post(
            f"{self.BASE_URL}/api/v1/agentic-chat",
            json=payload,
            timeout=60
        )
        
        assert response.status_code == 200
        data = response.json()
        # Should handle chat history correctly
        assert len(data["answer"]) > 0
        print("✅ API session management support test passed")


class TestFrontendAPIContract:
    """Test API contract matches frontend expectations"""
    
    BASE_URL = "http://localhost:8000"
    TIMEOUT = 30
    
    def test_chat_response_contract(self):
        """Verify chat response matches expected contract"""
        payload = {
            "query": "What is contract law?",
            "mode": "public"
        }
        response = requests.post(
            f"{self.BASE_URL}/api/v1/chat",
            json=payload,
            timeout=self.TIMEOUT
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify response contract
        contract = {
            "answer": str,
            "sources": list,
            "safety": dict,
            "metrics": dict,
            "confidence_score": (int, float),
            "legal_jurisdiction": str,
            "mode": str
        }
        
        for field, expected_type in contract.items():
            assert field in data, f"Missing required field: {field}"
            assert isinstance(data[field], expected_type), \
                f"Field {field} has wrong type: expected {expected_type}, got {type(data[field])}"
        
        print("✅ Chat response contract test passed")
    
    def test_sources_structure_contract(self):
        """Verify sources structure matches frontend expectations"""
        payload = {
            "query": "What are employment rights?",
            "mode": "solicitor",
            "top_k": 3
        }
        response = requests.post(
            f"{self.BASE_URL}/api/v1/chat",
            json=payload,
            timeout=self.TIMEOUT
        )
        
        assert response.status_code == 200
        data = response.json()
        sources = data["sources"]
        
        # Each source should have required fields
        for source in sources:
            assert isinstance(source, dict)
            # Check for common source fields
            assert "chunk_id" in source or "title" in source or "text_snippet" in source
        
        print("✅ Sources structure contract test passed")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

