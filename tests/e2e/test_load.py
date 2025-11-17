"""
Load Testing - Concurrent User Simulation
Phase 4.1: Performance Under Load
"""

import pytest
import requests
import time
import concurrent.futures
import statistics
from typing import List, Dict, Any
from datetime import datetime


class TestLoadTesting:
    """Load testing to simulate concurrent users"""
    
    BASE_URL = "http://localhost:8000"
    TIMEOUT = 30
    
    def make_chat_request(self, query: str, mode: str = "public") -> Dict[str, Any]:
        """Make a single chat request"""
        start_time = time.time()
        try:
            payload = {"query": query, "mode": mode, "top_k": 5}
            response = requests.post(
                f"{self.BASE_URL}/api/v1/chat",
                json=payload,
                timeout=self.TIMEOUT
            )
            response_time = time.time() - start_time
            
            return {
                "status_code": response.status_code,
                "response_time": response_time,
                "success": response.status_code == 200,
                "error": None if response.status_code == 200 else response.text[:100]
            }
        except Exception as e:
            response_time = time.time() - start_time
            return {
                "status_code": None,
                "response_time": response_time,
                "success": False,
                "error": str(e)[:100]
            }
    
    def test_light_load(self):
        """Test under light load (5 concurrent users)"""
        queries = [
            "What is contract law?",
            "What are employment rights?",
            "What is the Sale of Goods Act?",
            "What are consumer rights?",
            "What is GDPR?"
        ]
        
        start_time = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [
                executor.submit(self.make_chat_request, query, "public")
                for query in queries
            ]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        total_time = time.time() - start_time
        
        # Analyze results
        successful = [r for r in results if r["success"]]
        failed = [r for r in results if not r["success"]]
        response_times = [r["response_time"] for r in successful]
        
        success_rate = len(successful) / len(results) * 100
        avg_response_time = statistics.mean(response_times) if response_times else 0
        max_response_time = max(response_times) if response_times else 0
        
        print(f"\n📊 Light Load Test Results:")
        print(f"   Total requests: {len(results)}")
        print(f"   Successful: {len(successful)} ({success_rate:.1f}%)")
        print(f"   Failed: {len(failed)}")
        print(f"   Average response time: {avg_response_time:.2f}s")
        print(f"   Max response time: {max_response_time:.2f}s")
        print(f"   Total time: {total_time:.2f}s")
        
        # Assertions
        assert success_rate >= 80, f"Success rate {success_rate:.1f}% below 80% threshold"
        assert avg_response_time < 5.0, f"Average response time {avg_response_time:.2f}s exceeds 5s"
        print("✅ Light load test passed")
    
    def test_medium_load(self):
        """Test under medium load (10 concurrent users)"""
        queries = [
            "What is contract law?",
            "What are employment rights?",
            "What is the Sale of Goods Act?",
            "What are consumer rights?",
            "What is GDPR?",
            "What is data protection?",
            "What are director duties?",
            "What is breach of contract?",
            "What are implied terms?",
            "What is unfair dismissal?"
        ]
        
        start_time = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [
                executor.submit(self.make_chat_request, query, "public")
                for query in queries
            ]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        total_time = time.time() - start_time
        
        # Analyze results
        successful = [r for r in results if r["success"]]
        failed = [r for r in results if not r["success"]]
        response_times = [r["response_time"] for r in successful]
        
        success_rate = len(successful) / len(results) * 100
        avg_response_time = statistics.mean(response_times) if response_times else 0
        max_response_time = max(response_times) if response_times else 0
        
        print(f"\n📊 Medium Load Test Results:")
        print(f"   Total requests: {len(results)}")
        print(f"   Successful: {len(successful)} ({success_rate:.1f}%)")
        print(f"   Failed: {len(failed)}")
        print(f"   Average response time: {avg_response_time:.2f}s")
        print(f"   Max response time: {max_response_time:.2f}s")
        print(f"   Total time: {total_time:.2f}s")
        
        # Assertions
        assert success_rate >= 70, f"Success rate {success_rate:.1f}% below 70% threshold"
        assert avg_response_time < 8.0, f"Average response time {avg_response_time:.2f}s exceeds 8s"
        print("✅ Medium load test passed")
    
    def test_heavy_load(self):
        """Test under heavy load (20 concurrent users)"""
        base_queries = [
            "What is contract law?",
            "What are employment rights?",
            "What is the Sale of Goods Act?",
            "What are consumer rights?",
            "What is GDPR?"
        ]
        # Repeat queries to reach 20 requests
        queries = (base_queries * 4)[:20]
        
        start_time = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
            futures = [
                executor.submit(self.make_chat_request, query, "public")
                for query in queries
            ]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        total_time = time.time() - start_time
        
        # Analyze results
        successful = [r for r in results if r["success"]]
        failed = [r for r in results if not r["success"]]
        response_times = [r["response_time"] for r in successful]
        
        success_rate = len(successful) / len(results) * 100
        avg_response_time = statistics.mean(response_times) if response_times else 0
        max_response_time = max(response_times) if response_times else 0
        
        print(f"\n📊 Heavy Load Test Results:")
        print(f"   Total requests: {len(results)}")
        print(f"   Successful: {len(successful)} ({success_rate:.1f}%)")
        print(f"   Failed: {len(failed)}")
        print(f"   Average response time: {avg_response_time:.2f}s")
        print(f"   Max response time: {max_response_time:.2f}s")
        print(f"   Total time: {total_time:.2f}s")
        
        # Assertions - more lenient for heavy load
        assert success_rate >= 60, f"Success rate {success_rate:.1f}% below 60% threshold"
        assert avg_response_time < 12.0, f"Average response time {avg_response_time:.2f}s exceeds 12s"
        print("✅ Heavy load test passed")
    
    def test_sustained_load(self):
        """Test sustained load over time (10 requests over 30 seconds)"""
        query = "What is contract law?"
        results = []
        
        start_time = time.time()
        end_time = start_time + 30  # 30 seconds
        
        while time.time() < end_time and len(results) < 10:
            result = self.make_chat_request(query, "public")
            results.append(result)
            time.sleep(3)  # Wait 3 seconds between requests
        
        # Analyze results
        successful = [r for r in results if r["success"]]
        failed = [r for r in results if not r["success"]]
        response_times = [r["response_time"] for r in successful]
        
        success_rate = len(successful) / len(results) * 100 if results else 0
        avg_response_time = statistics.mean(response_times) if response_times else 0
        
        print(f"\n📊 Sustained Load Test Results:")
        print(f"   Total requests: {len(results)}")
        print(f"   Successful: {len(successful)} ({success_rate:.1f}%)")
        print(f"   Failed: {len(failed)}")
        print(f"   Average response time: {avg_response_time:.2f}s")
        
        # Assertions
        assert success_rate >= 80, f"Success rate {success_rate:.1f}% below 80% threshold"
        assert avg_response_time < 5.0, f"Average response time {avg_response_time:.2f}s exceeds 5s"
        print("✅ Sustained load test passed")
    
    def test_mixed_endpoints_load(self):
        """Test load across multiple endpoints simultaneously"""
        def make_hybrid_search_request():
            start_time = time.time()
            try:
                payload = {"query": "contract law", "top_k": 5}
                response = requests.post(
                    f"{self.BASE_URL}/api/v1/search/hybrid",
                    json=payload,
                    timeout=self.TIMEOUT
                )
                return {
                    "status_code": response.status_code,
                    "response_time": time.time() - start_time,
                    "success": response.status_code == 200
                }
            except Exception as e:
                return {
                    "status_code": None,
                    "response_time": time.time() - start_time,
                    "success": False,
                    "error": str(e)[:100]
                }
        
        def make_agentic_chat_request():
            start_time = time.time()
            try:
                payload = {"query": "What is the Sale of Goods Act?", "mode": "public"}
                response = requests.post(
                    f"{self.BASE_URL}/api/v1/agentic-chat",
                    json=payload,
                    timeout=60
                )
                return {
                    "status_code": response.status_code,
                    "response_time": time.time() - start_time,
                    "success": response.status_code == 200
                }
            except Exception as e:
                return {
                    "status_code": None,
                    "response_time": time.time() - start_time,
                    "success": False,
                    "error": str(e)[:100]
                }
        
        # Mix of different endpoint requests
        tasks = [
            self.make_chat_request("What is contract law?", "public") for _ in range(3)
        ] + [
            make_hybrid_search_request() for _ in range(2)
        ] + [
            make_agentic_chat_request() for _ in range(2)
        ]
        
        start_time = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=7) as executor:
            futures = [executor.submit(task) for task in tasks]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        total_time = time.time() - start_time
        
        # Analyze results
        successful = [r for r in results if r["success"]]
        response_times = [r["response_time"] for r in successful]
        
        success_rate = len(successful) / len(results) * 100
        avg_response_time = statistics.mean(response_times) if response_times else 0
        
        print(f"\n📊 Mixed Endpoints Load Test Results:")
        print(f"   Total requests: {len(results)}")
        print(f"   Successful: {len(successful)} ({success_rate:.1f}%)")
        print(f"   Average response time: {avg_response_time:.2f}s")
        print(f"   Total time: {total_time:.2f}s")
        
        # Assertions
        assert success_rate >= 70, f"Success rate {success_rate:.1f}% below 70% threshold"
        print("✅ Mixed endpoints load test passed")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

