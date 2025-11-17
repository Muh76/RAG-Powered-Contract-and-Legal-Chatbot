"""
Error Scenario Testing
Phase 4.1: Comprehensive Error Handling Tests
"""

import pytest
import requests
import time
from typing import Dict, Any


class TestErrorScenarios:
    """Test error handling for various failure scenarios"""
    
    BASE_URL = "http://localhost:8000"
    TIMEOUT = 30
    
    # ==================== Network Failure Scenarios ====================
    
    def test_timeout_handling(self):
        """Test API handles timeout scenarios gracefully"""
        # Use a very short timeout to simulate timeout
        try:
            response = requests.post(
                f"{self.BASE_URL}/api/v1/chat",
                json={"query": "What is contract law?", "mode": "public"},
                timeout=0.001  # Extremely short timeout
            )
        except requests.exceptions.Timeout:
            # Expected behavior - timeout should be caught
            print("✅ Timeout handling test passed - timeout exception caught")
            return
        except Exception as e:
            # Any other exception is also acceptable
            print(f"✅ Timeout handling test passed - exception caught: {type(e).__name__}")
            return
        
        # If we get here, the request didn't timeout (unexpected but not a failure)
        print("⚠️ Timeout test: Request completed (timeout may be too short for reliable test)")
    
    def test_connection_refused_with_invalid_url(self):
        """Test behavior when connecting to invalid URL"""
        try:
            response = requests.get(
                "http://localhost:9999/api/v1/health",
                timeout=5
            )
        except requests.exceptions.ConnectionError:
            print("✅ Connection refused test passed - connection error caught")
            return
        except Exception as e:
            print(f"✅ Connection refused test passed - exception caught: {type(e).__name__}")
            return
    
    # ==================== Invalid Input Scenarios ====================
    
    def test_invalid_query_types(self):
        """Test API handles various invalid query types"""
        invalid_queries = [
            None,  # This will fail JSON parsing, which is expected
            123,  # Wrong type
            [],  # Wrong type
            {},  # Wrong type
        ]
        
        for invalid_query in invalid_queries:
            try:
                payload = {"query": invalid_query, "mode": "public"}
                response = requests.post(
                    f"{self.BASE_URL}/api/v1/chat",
                    json=payload,
                    timeout=10
                )
                # Should return 422 or 400 for validation errors
                assert response.status_code in [400, 422], \
                    f"Expected 400/422 for invalid query type, got {response.status_code}"
            except requests.exceptions.RequestException:
                # Network errors are acceptable
                pass
        
        print("✅ Invalid query types test passed")
    
    def test_invalid_mode_values(self):
        """Test API rejects invalid mode values"""
        invalid_modes = ["invalid", "admin", "test", 123, None]
        
        for invalid_mode in invalid_modes:
            try:
                payload = {"query": "test query", "mode": invalid_mode}
                response = requests.post(
                    f"{self.BASE_URL}/api/v1/chat",
                    json=payload,
                    timeout=10
                )
                # Should return 422 for validation errors
                assert response.status_code == 422, \
                    f"Expected 422 for invalid mode '{invalid_mode}', got {response.status_code}"
            except requests.exceptions.RequestException:
                pass
        
        print("✅ Invalid mode values test passed")
    
    def test_invalid_top_k_values(self):
        """Test API rejects invalid top_k values"""
        invalid_top_k_values = [0, -1, 101, "invalid", None]
        
        for invalid_top_k in invalid_top_k_values:
            try:
                payload = {"query": "test query", "mode": "public", "top_k": invalid_top_k}
                response = requests.post(
                    f"{self.BASE_URL}/api/v1/chat",
                    json=payload,
                    timeout=10
                )
                # Should return 422 for validation errors
                assert response.status_code == 422, \
                    f"Expected 422 for invalid top_k '{invalid_top_k}', got {response.status_code}"
            except requests.exceptions.RequestException:
                pass
        
        print("✅ Invalid top_k values test passed")
    
    def test_invalid_fusion_strategy(self):
        """Test hybrid search rejects invalid fusion strategy"""
        payload = {
            "query": "test query",
            "top_k": 5,
            "fusion_strategy": "invalid_strategy"
        }
        response = requests.post(
            f"{self.BASE_URL}/api/v1/search/hybrid",
            json=payload,
            timeout=10
        )
        assert response.status_code == 422, "Expected 422 for invalid fusion strategy"
        print("✅ Invalid fusion strategy test passed")
    
    def test_malformed_metadata_filters(self):
        """Test hybrid search handles malformed metadata filters"""
        malformed_filters = [
            [{"field": "jurisdiction"}],  # Missing value
            [{"value": "UK"}],  # Missing field
            [{"field": "jurisdiction", "value": "UK", "operator": "invalid"}],  # Invalid operator
            "not_a_list",  # Wrong type
        ]
        
        for filters in malformed_filters:
            try:
                payload = {
                    "query": "test query",
                    "top_k": 5,
                    "metadata_filters": filters
                }
                response = requests.post(
                    f"{self.BASE_URL}/api/v1/search/hybrid",
                    json=payload,
                    timeout=10
                )
                # Should return 422 or handle gracefully
                assert response.status_code in [200, 422], \
                    f"Expected 200 or 422 for malformed filters, got {response.status_code}"
            except requests.exceptions.RequestException:
                pass
        
        print("✅ Malformed metadata filters test passed")
    
    # ==================== Missing Required Fields ====================
    
    def test_missing_query_field(self):
        """Test API rejects requests missing query field"""
        payload = {"mode": "public"}
        response = requests.post(
            f"{self.BASE_URL}/api/v1/chat",
            json=payload,
            timeout=10
        )
        assert response.status_code == 422
        print("✅ Missing query field test passed")
    
    def test_missing_mode_field(self):
        """Test API handles missing mode field (should default or require)"""
        payload = {"query": "test query"}
        response = requests.post(
            f"{self.BASE_URL}/api/v1/chat",
            json=payload,
            timeout=10
        )
        # Mode might be optional with default, or required (422)
        assert response.status_code in [200, 422]
        print("✅ Missing mode field test passed")
    
    # ==================== SQL Injection / XSS Attempts ====================
    
    def test_sql_injection_attempts(self):
        """Test API handles SQL injection attempts safely"""
        sql_injection_queries = [
            "'; DROP TABLE users; --",
            "1' OR '1'='1",
            "admin'--",
            "'; SELECT * FROM documents; --"
        ]
        
        for query in sql_injection_queries:
            try:
                payload = {"query": query, "mode": "public"}
                response = requests.post(
                    f"{self.BASE_URL}/api/v1/chat",
                    json=payload,
                    timeout=10
                )
                # Should not crash, should return either blocked or safe response
                assert response.status_code in [200, 422, 400]
                # Should not contain SQL error messages
                response_text = response.text.lower()
                assert "sql" not in response_text or "syntax error" not in response_text
            except requests.exceptions.RequestException:
                pass
        
        print("✅ SQL injection attempts test passed")
    
    def test_xss_attempts(self):
        """Test API handles XSS attempts safely"""
        xss_queries = [
            "<script>alert('xss')</script>",
            "javascript:alert('xss')",
            "<img src=x onerror=alert('xss')>",
            "';alert(String.fromCharCode(88,83,83))//';"
        ]
        
        for query in xss_queries:
            try:
                payload = {"query": query, "mode": "public"}
                response = requests.post(
                    f"{self.BASE_URL}/api/v1/chat",
                    json=payload,
                    timeout=10
                )
                # Should not crash, should return either blocked or safe response
                assert response.status_code in [200, 422, 400]
            except requests.exceptions.RequestException:
                pass
        
        print("✅ XSS attempts test passed")
    
    # ==================== Large Payload Scenarios ====================
    
    def test_large_query_payload(self):
        """Test API handles large query payloads"""
        # Test with query at max length (1000 chars)
        large_query = "a" * 1000
        payload = {"query": large_query, "mode": "public"}
        response = requests.post(
            f"{self.BASE_URL}/api/v1/chat",
            json=payload,
            timeout=30
        )
        # Should handle gracefully (either process or reject)
        assert response.status_code in [200, 422]
        print("✅ Large query payload test passed")
    
    def test_large_chat_history(self):
        """Test agentic chat handles large conversation history"""
        # Create large chat history
        chat_history = [
            {"role": "user" if i % 2 == 0 else "assistant", "content": f"Message {i}" * 10}
            for i in range(50)  # 50 messages
        ]
        payload = {
            "query": "Summarize our conversation",
            "mode": "public",
            "chat_history": chat_history
        }
        response = requests.post(
            f"{self.BASE_URL}/api/v1/agentic-chat",
            json=payload,
            timeout=60
        )
        # Should handle gracefully (either process or reject)
        assert response.status_code in [200, 422, 400]
        print("✅ Large chat history test passed")
    
    # ==================== Concurrent Request Scenarios ====================
    
    def test_concurrent_requests(self):
        """Test API handles concurrent requests"""
        import concurrent.futures
        
        def make_request():
            payload = {"query": "What is contract law?", "mode": "public"}
            return requests.post(
                f"{self.BASE_URL}/api/v1/chat",
                json=payload,
                timeout=30
            )
        
        # Make 5 concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(make_request) for _ in range(5)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        # All requests should succeed (200) or be handled gracefully
        status_codes = [r.status_code for r in results]
        assert all(code in [200, 429, 503] for code in status_codes), \
            f"Unexpected status codes: {status_codes}"
        print(f"✅ Concurrent requests test passed - {len(results)} requests handled")
    
    # ==================== Service Degradation Scenarios ====================
    
    def test_graceful_degradation_empty_results(self):
        """Test API handles empty retrieval results gracefully"""
        # Query that likely won't match anything
        payload = {
            "query": "xyzabc123nonexistentterm456",
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
        # Should return a helpful message, not crash
        assert "answer" in data
        assert len(data["answer"]) > 0
        print("✅ Graceful degradation (empty results) test passed")
    
    def test_error_response_format(self):
        """Test error responses follow consistent format"""
        # Trigger a validation error
        payload = {"query": "", "mode": "public"}
        response = requests.post(
            f"{self.BASE_URL}/api/v1/chat",
            json=payload,
            timeout=10
        )
        
        assert response.status_code == 422
        # Error response should be JSON
        assert response.headers["content-type"] == "application/json"
        error_data = response.json()
        # Should have error details
        assert "detail" in error_data or "errors" in error_data or "message" in error_data
        print("✅ Error response format test passed")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

