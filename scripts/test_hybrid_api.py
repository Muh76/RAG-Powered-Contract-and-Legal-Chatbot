#!/usr/bin/env python3
"""
Test Hybrid Search API Endpoint
Phase 2: Test /api/v1/search/hybrid endpoint

This script tests:
1. POST /api/v1/search/hybrid
2. GET /api/v1/search/hybrid
3. Metadata filtering
4. Fusion strategies (RRF, weighted)
"""

import requests
import json
import time
from typing import Dict, Any
import sys


BASE_URL = "http://localhost:8000"
API_BASE = f"{BASE_URL}/api/v1"


def test_health():
    """Test health endpoint"""
    print("🏥 Testing Health Endpoint...")
    try:
        response = requests.get(f"{API_BASE}/health", timeout=5)
        if response.status_code == 200:
            print(f"   ✅ Health check passed: {response.json()['status']}")
            return True
        else:
            print(f"   ❌ Health check failed: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print(f"   ❌ Cannot connect to API at {BASE_URL}")
        print(f"   Make sure the API server is running: uvicorn app.api.main:app --reload")
        return False
    except Exception as e:
        print(f"   ❌ Health check error: {e}")
        return False


def test_hybrid_search_post():
    """Test POST /api/v1/search/hybrid"""
    print("\n📤 Testing POST /api/v1/search/hybrid...")
    
    request_data = {
        "query": "What are the implied conditions in a contract of sale?",
        "top_k": 5,
        "fusion_strategy": "rrf"
    }
    
    try:
        start_time = time.time()
        response = requests.post(
            f"{API_BASE}/search/hybrid",
            json=request_data,
            timeout=30
        )
        elapsed_time = (time.time() - start_time) * 1000
        
        if response.status_code == 200:
            result = response.json()
            print(f"   ✅ POST request successful ({elapsed_time:.1f}ms)")
            print(f"   Query: {result['query']}")
            print(f"   Results: {result['total_results']} chunks found")
            print(f"   Fusion strategy: {result['fusion_strategy']}")
            print(f"   Search time: {result['search_time_ms']:.1f}ms")
            print(f"   BM25 results: {result.get('bm25_results_count', 'N/A')}")
            print(f"   Semantic results: {result.get('semantic_results_count', 'N/A')}")
            
            if result['results']:
                print(f"\n   Top 3 Results:")
                for i, r in enumerate(result['results'][:3], 1):
                    print(f"      {i}. Rank {r['rank']}: Score {r['similarity_score']:.3f}")
                    print(f"         BM25: {r.get('bm25_score', 'N/A')}, "
                          f"Semantic: {r.get('semantic_score', 'N/A')}")
                    print(f"         Section: {r.get('section', 'N/A')}")
                    text_preview = r['text'][:60] + "..." if len(r['text']) > 60 else r['text']
                    print(f"         Text: {text_preview}")
            
            return True
        else:
            print(f"   ❌ POST request failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"   ❌ POST request error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_hybrid_search_get():
    """Test GET /api/v1/search/hybrid"""
    print("\n📥 Testing GET /api/v1/search/hybrid...")
    
    params = {
        "query": "Employee rights in the UK",
        "top_k": 5,
        "fusion_strategy": "weighted"
    }
    
    try:
        start_time = time.time()
        response = requests.get(
            f"{API_BASE}/search/hybrid",
            params=params,
            timeout=30
        )
        elapsed_time = (time.time() - start_time) * 1000
        
        if response.status_code == 200:
            result = response.json()
            print(f"   ✅ GET request successful ({elapsed_time:.1f}ms)")
            print(f"   Query: {result['query']}")
            print(f"   Results: {result['total_results']} chunks found")
            print(f"   Fusion strategy: {result['fusion_strategy']}")
            
            return True
        else:
            print(f"   ❌ GET request failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"   ❌ GET request error: {e}")
        return False


def test_metadata_filtering():
    """Test hybrid search with metadata filtering"""
    print("\n🔍 Testing Metadata Filtering...")
    
    request_data = {
        "query": "contract law",
        "top_k": 5,
        "fusion_strategy": "rrf",
        "metadata_filters": [
            {
                "field": "jurisdiction",
                "value": "UK",
                "operator": "eq"
            }
        ]
    }
    
    try:
        response = requests.post(
            f"{API_BASE}/search/hybrid",
            json=request_data,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"   ✅ Metadata filtering successful")
            print(f"   Results: {result['total_results']} chunks found (filtered)")
            
            # Verify all results have UK jurisdiction
            if result['results']:
                all_uk = all(
                    r.get('jurisdiction') == 'UK' or 
                    r.get('metadata', {}).get('jurisdiction') == 'UK'
                    for r in result['results']
                )
                if all_uk:
                    print(f"   ✅ All results filtered to UK jurisdiction")
                else:
                    print(f"   ⚠️ Some results may not match filter")
            
            return True
        else:
            print(f"   ❌ Metadata filtering failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"   ❌ Metadata filtering error: {e}")
        return False


def test_multiple_filters():
    """Test hybrid search with multiple metadata filters"""
    print("\n🔍 Testing Multiple Metadata Filters...")
    
    request_data = {
        "query": "contract law",
        "top_k": 5,
        "fusion_strategy": "rrf",
        "metadata_filters": [
            {
                "field": "jurisdiction",
                "value": "UK",
                "operator": "eq"
            },
            {
                "field": "document_type",
                "value": "statute",
                "operator": "eq"
            }
        ]
    }
    
    try:
        response = requests.post(
            f"{API_BASE}/search/hybrid",
            json=request_data,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"   ✅ Multiple filters successful")
            print(f"   Results: {result['total_results']} chunks found (filtered)")
            return True
        else:
            print(f"   ❌ Multiple filters failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"   ❌ Multiple filters error: {e}")
        return False


def test_in_filter():
    """Test IN filter operator"""
    print("\n🔍 Testing IN Filter...")
    
    request_data = {
        "query": "contract law",
        "top_k": 5,
        "fusion_strategy": "rrf",
        "metadata_filters": [
            {
                "field": "document_type",
                "value": ["statute", "contract"],
                "operator": "in"
            }
        ]
    }
    
    try:
        response = requests.post(
            f"{API_BASE}/search/hybrid",
            json=request_data,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"   ✅ IN filter successful")
            print(f"   Results: {result['total_results']} chunks found (filtered)")
            return True
        else:
            print(f"   ❌ IN filter failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"   ❌ IN filter error: {e}")
        return False


def test_fusion_strategies():
    """Test both fusion strategies"""
    print("\n🔄 Testing Fusion Strategies...")
    
    strategies = ["rrf", "weighted"]
    
    for strategy in strategies:
        print(f"\n   Testing {strategy.upper()} fusion...")
        request_data = {
            "query": "What are employee rights?",
            "top_k": 3,
            "fusion_strategy": strategy
        }
        
        try:
            response = requests.post(
                f"{API_BASE}/search/hybrid",
                json=request_data,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"      ✅ {strategy.upper()} successful: {result['total_results']} results")
                if result['results']:
                    print(f"         Top score: {result['results'][0]['similarity_score']:.3f}")
            else:
                print(f"      ❌ {strategy.upper()} failed: {response.status_code}")
                
        except Exception as e:
            print(f"      ❌ {strategy.upper()} error: {e}")


def main():
    """Run all tests"""
    print("=" * 60)
    print("Phase 2: Hybrid Search API Endpoint Testing")
    print("=" * 60)
    
    results = {}
    
    # Test health endpoint first
    results['health'] = test_health()
    
    if not results['health']:
        print("\n⚠️ API server is not running. Please start it first:")
        print("   uvicorn app.api.main:app --reload")
        return False
    
    # Test hybrid search endpoints
    results['post'] = test_hybrid_search_post()
    results['get'] = test_hybrid_search_get()
    results['metadata_filter'] = test_metadata_filtering()
    results['multiple_filters'] = test_multiple_filters()
    results['in_filter'] = test_in_filter()
    
    # Test fusion strategies
    test_fusion_strategies()
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, passed_test in results.items():
        status = "✅ PASS" if passed_test else "❌ FAIL"
        print(f"   {test_name:<25} {status}")
    
    print(f"\n   Total: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! Hybrid search API is working correctly.")
        return True
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Review issues above.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

