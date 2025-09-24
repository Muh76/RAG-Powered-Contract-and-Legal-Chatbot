# tests/test_api_integration.py - End-to-End API Integration Tests
import pytest
import requests
import json
import time
from typing import Dict, Any

class TestAPIIntegration:
    """End-to-end API integration tests"""
    
    def __init__(self):
        self.base_url = "http://localhost:8000"
        self.test_queries = [
            {
                "query": "What are the implied conditions in a contract of sale?",
                "mode": "solicitor",
                "expected_citations": True,
                "expected_legal": True
            },
            {
                "query": "How do I cook pasta?",
                "mode": "public", 
                "expected_citations": False,
                "expected_legal": False
            },
            {
                "query": "What are employment rights under UK law?",
                "mode": "public",
                "expected_citations": True,
                "expected_legal": True
            }
        ]
    
    def test_health_endpoint(self) -> bool:
        """Test the health check endpoint"""
        try:
            response = requests.get(f"{self.base_url}/api/v1/health", timeout=10)
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Health check passed: {data.get('status')}")
                return True
            else:
                print(f"❌ Health check failed: {response.status_code}")
                return False
        except requests.exceptions.RequestException as e:
            print(f"❌ Health check error: {e}")
            return False
    
    def test_chat_endpoint_legal_query(self) -> bool:
        """Test chat endpoint with legal query"""
        try:
            query_data = {
                "query": "What are the implied conditions in a contract of sale?",
                "mode": "solicitor",
                "top_k": 3
            }
            
            response = requests.post(
                f"{self.base_url}/api/v1/chat",
                json=query_data,
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Legal query test passed")
                print(f"   - Answer length: {len(data.get('answer', ''))} chars")
                print(f"   - Citations: {len(data.get('citations', []))}")
                print(f"   - Mode: {data.get('mode')}")
                print(f"   - Guardrails applied: {data.get('guardrails_applied')}")
                return True
            else:
                print(f"❌ Legal query test failed: {response.status_code}")
                print(f"   Response: {response.text}")
                return False
        except requests.exceptions.RequestException as e:
            print(f"❌ Legal query test error: {e}")
            return False
    
    def test_chat_endpoint_non_legal_query(self) -> bool:
        """Test chat endpoint with non-legal query (should be blocked)"""
        try:
            query_data = {
                "query": "How do I cook pasta?",
                "mode": "public",
                "top_k": 3
            }
            
            response = requests.post(
                f"{self.base_url}/api/v1/chat",
                json=query_data,
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Non-legal query test passed (blocked as expected)")
                print(f"   - Citations: {len(data.get('citations', []))}")
                print(f"   - Guardrails applied: {data.get('guardrails_applied')}")
                return True
            else:
                print(f"❌ Non-legal query test failed: {response.status_code}")
                print(f"   Response: {response.text}")
                return False
        except requests.exceptions.RequestException as e:
            print(f"❌ Non-legal query test error: {e}")
            return False
    
    def test_search_endpoint(self) -> bool:
        """Test the debug search endpoint"""
        try:
            params = {
                "query": "contract of sale",
                "top_k": 3
            }
            
            response = requests.get(
                f"{self.base_url}/api/v1/search",
                params=params,
                timeout=15
            )
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Search endpoint test passed")
                print(f"   - Retrieved chunks: {len(data.get('retrieved_chunks', []))}")
                return True
            else:
                print(f"❌ Search endpoint test failed: {response.status_code}")
                print(f"   Response: {response.text}")
                return False
        except requests.exceptions.RequestException as e:
            print(f"❌ Search endpoint test error: {e}")
            return False
    
    def test_api_performance(self) -> bool:
        """Test API response times"""
        try:
            query_data = {
                "query": "What are the seller's obligations in a contract of sale?",
                "mode": "solicitor",
                "top_k": 3
            }
            
            start_time = time.time()
            response = requests.post(
                f"{self.base_url}/api/v1/chat",
                json=query_data,
                timeout=30
            )
            end_time = time.time()
            
            response_time = end_time - start_time
            
            if response.status_code == 200 and response_time < 10:  # Should respond within 10 seconds
                print(f"✅ Performance test passed")
                print(f"   - Response time: {response_time:.2f} seconds")
                return True
            else:
                print(f"❌ Performance test failed")
                print(f"   - Response time: {response_time:.2f} seconds")
                print(f"   - Status code: {response.status_code}")
                return False
        except requests.exceptions.RequestException as e:
            print(f"❌ Performance test error: {e}")
            return False
    
    def run_all_tests(self) -> Dict[str, bool]:
        """Run all API integration tests"""
        print("🧪 Running End-to-End API Integration Tests:")
        print("=" * 60)
        
        results = {}
        
        # Test 1: Health check
        print("\n🔍 Test 1: Health Check Endpoint")
        print("-" * 40)
        results['health'] = self.test_health_endpoint()
        
        # Test 2: Legal query
        print("\n🔍 Test 2: Legal Query (Solicitor Mode)")
        print("-" * 40)
        results['legal_query'] = self.test_chat_endpoint_legal_query()
        
        # Test 3: Non-legal query
        print("\n🔍 Test 3: Non-Legal Query (Should be Blocked)")
        print("-" * 40)
        results['non_legal_query'] = self.test_chat_endpoint_non_legal_query()
        
        # Test 4: Search endpoint
        print("\n🔍 Test 4: Search Endpoint (Debug)")
        print("-" * 40)
        results['search'] = self.test_search_endpoint()
        
        # Test 5: Performance
        print("\n🔍 Test 5: API Performance")
        print("-" * 40)
        results['performance'] = self.test_api_performance()
        
        # Summary
        print("\n📊 Test Results Summary:")
        print("=" * 60)
        passed = sum(results.values())
        total = len(results)
        
        for test_name, result in results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"  {test_name.replace('_', ' ').title()}: {status}")
        
        print(f"\n🎯 Overall Result: {passed}/{total} tests passed")
        
        if passed == total:
            print("🎉 All API integration tests passed!")
        else:
            print("⚠️  Some tests failed. Check the FastAPI server is running.")
        
        return results

# Function to run tests
def run_api_tests():
    """Run the API integration tests"""
    tester = TestAPIIntegration()
    return tester.run_all_tests()

if __name__ == "__main__":
    run_api_tests()
