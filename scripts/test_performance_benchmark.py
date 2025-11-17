#!/usr/bin/env python3
"""
Performance Benchmarking Script
Phase 4.1: Response Time Targets and Performance Analysis
"""

import requests
import time
import statistics
from typing import List, Dict, Any
from datetime import datetime
import json


class PerformanceBenchmark:
    """Performance benchmarking for all endpoints"""
    
    BASE_URL = "http://localhost:8000"
    
    # Performance targets
    TARGET_SIMPLE_QUERY_TIME = 3.0  # seconds
    TARGET_COMPLEX_QUERY_TIME = 10.0  # seconds
    TARGET_HYBRID_SEARCH_TIME = 5.0  # seconds
    TARGET_AGENTIC_CHAT_TIME = 30.0  # seconds
    
    def __init__(self):
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "tests": []
        }
    
    def measure_chat_response_time(self, query: str, mode: str = "public", top_k: int = 5) -> Dict[str, Any]:
        """Measure response time for chat endpoint"""
        payload = {
            "query": query,
            "mode": mode,
            "top_k": top_k
        }
        
        start_time = time.time()
        try:
            response = requests.post(
                f"{self.BASE_URL}/api/v1/chat",
                json=payload,
                timeout=30
            )
            response_time = time.time() - start_time
            
            return {
                "endpoint": "/api/v1/chat",
                "query": query[:50],
                "mode": mode,
                "status_code": response.status_code,
                "response_time": response_time,
                "success": response.status_code == 200,
                "target_met": response_time < self.TARGET_SIMPLE_QUERY_TIME if len(query) < 100 else response_time < self.TARGET_COMPLEX_QUERY_TIME,
                "data_size": len(response.text) if response.status_code == 200 else 0
            }
        except Exception as e:
            response_time = time.time() - start_time
            return {
                "endpoint": "/api/v1/chat",
                "query": query[:50],
                "mode": mode,
                "status_code": None,
                "response_time": response_time,
                "success": False,
                "error": str(e)[:100],
                "target_met": False
            }
    
    def measure_hybrid_search_time(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """Measure response time for hybrid search"""
        payload = {
            "query": query,
            "top_k": top_k,
            "fusion_strategy": "rrf"
        }
        
        start_time = time.time()
        try:
            response = requests.post(
                f"{self.BASE_URL}/api/v1/search/hybrid",
                json=payload,
                timeout=30
            )
            response_time = time.time() - start_time
            
            return {
                "endpoint": "/api/v1/search/hybrid",
                "query": query[:50],
                "status_code": response.status_code,
                "response_time": response_time,
                "success": response.status_code == 200,
                "target_met": response_time < self.TARGET_HYBRID_SEARCH_TIME,
                "data_size": len(response.text) if response.status_code == 200 else 0
            }
        except Exception as e:
            response_time = time.time() - start_time
            return {
                "endpoint": "/api/v1/search/hybrid",
                "query": query[:50],
                "status_code": None,
                "response_time": response_time,
                "success": False,
                "error": str(e)[:100],
                "target_met": False
            }
    
    def measure_agentic_chat_time(self, query: str, mode: str = "public") -> Dict[str, Any]:
        """Measure response time for agentic chat"""
        payload = {
            "query": query,
            "mode": mode
        }
        
        start_time = time.time()
        try:
            response = requests.post(
                f"{self.BASE_URL}/api/v1/agentic-chat",
                json=payload,
                timeout=60
            )
            response_time = time.time() - start_time
            
            return {
                "endpoint": "/api/v1/agentic-chat",
                "query": query[:50],
                "mode": mode,
                "status_code": response.status_code,
                "response_time": response_time,
                "success": response.status_code == 200,
                "target_met": response_time < self.TARGET_AGENTIC_CHAT_TIME,
                "data_size": len(response.text) if response.status_code == 200 else 0
            }
        except Exception as e:
            response_time = time.time() - start_time
            return {
                "endpoint": "/api/v1/agentic-chat",
                "query": query[:50],
                "mode": mode,
                "status_code": None,
                "response_time": response_time,
                "success": False,
                "error": str(e)[:100],
                "target_met": False
            }
    
    def run_benchmark_suite(self, iterations: int = 5) -> Dict[str, Any]:
        """Run comprehensive performance benchmark suite"""
        print("🚀 Starting Performance Benchmark Suite")
        print("=" * 60)
        
        # Test queries
        simple_queries = [
            "What is contract law?",
            "What are employment rights?",
            "What is GDPR?"
        ]
        
        complex_queries = [
            "What are the implied conditions in a contract of sale under UK law?",
            "What are the key differences between the Sale of Goods Act 1979 and the Consumer Rights Act 2015?",
            "What are employee rights regarding unfair dismissal in the UK?"
        ]
        
        # 1. Simple query performance (chat endpoint)
        print("\n📊 Testing Simple Query Performance (Chat Endpoint)")
        print("-" * 60)
        simple_results = []
        for query in simple_queries:
            for _ in range(iterations):
                result = self.measure_chat_response_time(query, mode="public")
                simple_results.append(result)
                self.results["tests"].append(result)
                status = "✅" if result["success"] and result["target_met"] else "❌"
                print(f"  {status} {result['query']}: {result['response_time']:.2f}s "
                      f"(target: <{self.TARGET_SIMPLE_QUERY_TIME}s)")
        
        # 2. Complex query performance (chat endpoint)
        print("\n📊 Testing Complex Query Performance (Chat Endpoint)")
        print("-" * 60)
        complex_results = []
        for query in complex_queries:
            for _ in range(iterations):
                result = self.measure_chat_response_time(query, mode="solicitor")
                complex_results.append(result)
                self.results["tests"].append(result)
                status = "✅" if result["success"] and result["target_met"] else "❌"
                print(f"  {status} {result['query']}: {result['response_time']:.2f}s "
                      f"(target: <{self.TARGET_COMPLEX_QUERY_TIME}s)")
        
        # 3. Hybrid search performance
        print("\n📊 Testing Hybrid Search Performance")
        print("-" * 60)
        hybrid_results = []
        for query in simple_queries:
            for _ in range(iterations):
                result = self.measure_hybrid_search_time(query)
                hybrid_results.append(result)
                self.results["tests"].append(result)
                status = "✅" if result["success"] and result["target_met"] else "❌"
                print(f"  {status} {result['query']}: {result['response_time']:.2f}s "
                      f"(target: <{self.TARGET_HYBRID_SEARCH_TIME}s)")
        
        # 4. Agentic chat performance
        print("\n📊 Testing Agentic Chat Performance")
        print("-" * 60)
        agentic_results = []
        test_queries = [
            "What is the Sale of Goods Act 1979?",
            "Compare the Sale of Goods Act 1979 with the Consumer Rights Act 2015"
        ]
        for query in test_queries:
            for _ in range(iterations):
                result = self.measure_agentic_chat_time(query, mode="public")
                agentic_results.append(result)
                self.results["tests"].append(result)
                status = "✅" if result["success"] and result["target_met"] else "❌"
                print(f"  {status} {result['query']}: {result['response_time']:.2f}s "
                      f"(target: <{self.TARGET_AGENTIC_CHAT_TIME}s)")
        
        # Calculate statistics
        self.results["statistics"] = self._calculate_statistics()
        
        # Print summary
        self._print_summary(simple_results, complex_results, hybrid_results, agentic_results)
        
        return self.results
    
    def _calculate_statistics(self) -> Dict[str, Any]:
        """Calculate performance statistics"""
        successful_tests = [t for t in self.results["tests"] if t["success"]]
        failed_tests = [t for t in self.results["tests"] if not t["success"]]
        
        response_times = [t["response_time"] for t in successful_tests]
        
        stats = {
            "total_tests": len(self.results["tests"]),
            "successful": len(successful_tests),
            "failed": len(failed_tests),
            "success_rate": len(successful_tests) / len(self.results["tests"]) * 100 if self.results["tests"] else 0
        }
        
        if response_times:
            stats["response_time"] = {
                "mean": statistics.mean(response_times),
                "median": statistics.median(response_times),
                "min": min(response_times),
                "max": max(response_times),
                "stdev": statistics.stdev(response_times) if len(response_times) > 1 else 0
            }
        
        # Target met rate
        target_met = [t for t in successful_tests if t.get("target_met", False)]
        stats["target_met_rate"] = len(target_met) / len(successful_tests) * 100 if successful_tests else 0
        
        return stats
    
    def _print_summary(self, simple_results: List, complex_results: List, 
                      hybrid_results: List, agentic_results: List):
        """Print performance summary"""
        print("\n📈 Performance Summary")
        print("=" * 60)
        
        stats = self.results["statistics"]
        
        print(f"\nOverall Statistics:")
        print(f"  Total Tests: {stats['total_tests']}")
        print(f"  Successful: {stats['successful']} ({stats['success_rate']:.1f}%)")
        print(f"  Failed: {stats['failed']}")
        
        if stats.get("response_time"):
            rt = stats["response_time"]
            print(f"\nResponse Time Statistics:")
            print(f"  Mean: {rt['mean']:.2f}s")
            print(f"  Median: {rt['median']:.2f}s")
            print(f"  Min: {rt['min']:.2f}s")
            print(f"  Max: {rt['max']:.2f}s")
            print(f"  Std Dev: {rt['stdev']:.2f}s")
        
        print(f"\nTarget Met Rate: {stats['target_met_rate']:.1f}%")
        
        # Per-endpoint statistics
        endpoints = {
            "/api/v1/chat": simple_results + complex_results,
            "/api/v1/search/hybrid": hybrid_results,
            "/api/v1/agentic-chat": agentic_results
        }
        
        print(f"\nPer-Endpoint Statistics:")
        for endpoint, results in endpoints.items():
            successful = [r for r in results if r["success"]]
            if successful:
                times = [r["response_time"] for r in successful]
                target_met = [r for r in successful if r.get("target_met", False)]
                print(f"  {endpoint}:")
                print(f"    Success Rate: {len(successful)}/{len(results)} ({len(successful)/len(results)*100:.1f}%)")
                print(f"    Avg Response Time: {statistics.mean(times):.2f}s")
                print(f"    Target Met: {len(target_met)}/{len(successful)} ({len(target_met)/len(successful)*100:.1f}%)")
    
    def save_results(self, filename: str = "performance_benchmark_results.json"):
        """Save benchmark results to file"""
        with open(filename, "w") as f:
            json.dump(self.results, f, indent=2)
        print(f"\n💾 Results saved to {filename}")


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Performance Benchmark Suite")
    parser.add_argument("--iterations", type=int, default=5, help="Number of iterations per test")
    parser.add_argument("--output", type=str, default="performance_benchmark_results.json", 
                       help="Output file for results")
    
    args = parser.parse_args()
    
    benchmark = PerformanceBenchmark()
    results = benchmark.run_benchmark_suite(iterations=args.iterations)
    benchmark.save_results(args.output)
    
    # Exit code based on success rate
    stats = results["statistics"]
    if stats["success_rate"] < 80:
        print("\n⚠️  Warning: Success rate below 80%")
        return 1
    elif stats.get("target_met_rate", 0) < 80:
        print("\n⚠️  Warning: Target met rate below 80%")
        return 1
    else:
        print("\n✅ Performance benchmark passed")
        return 0


if __name__ == "__main__":
    exit(main())

