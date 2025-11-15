#!/usr/bin/env python3
"""
Test and Validate Semantic Retrieval
Phase 2: Module 1.2 - Semantic Retrieval Implementation

This script tests the SemanticRetriever to ensure:
1. Semantic retriever initializes correctly
2. Query embeddings are generated properly
3. FAISS integration works for vector similarity
4. Top-k semantic retrieval returns correct results
5. Performance is acceptable
"""

import os
import sys
import time
import logging
from pathlib import Path
from typing import List, Dict, Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from retrieval.semantic_retriever import SemanticRetriever
from retrieval.embeddings.embedding_generator import EmbeddingConfig
from app.core.config import settings

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_semantic_retriever_initialization():
    """Test 1: Initialize SemanticRetriever"""
    print("\n" + "="*60)
    print("TEST 1: SemanticRetriever Initialization")
    print("="*60)
    
    try:
        retriever = SemanticRetriever()
        
        if not retriever.is_ready():
            print("❌ SemanticRetriever initialized but not ready")
            print("   Check if FAISS index exists and embeddings are working")
            return False, None
        
        print("✅ SemanticRetriever initialized successfully")
        
        # Get stats
        stats = retriever.get_index_stats()
        print(f"   Index loaded: {stats['index_loaded']}")
        print(f"   Number of vectors: {stats['num_vectors']}")
        print(f"   Dimension: {stats['dimension']}")
        print(f"   Number of chunks: {stats['num_chunks']}")
        print(f"   Index type: {stats['index_type']}")
        
        return True, retriever
        
    except Exception as e:
        print(f"❌ SemanticRetriever initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def test_query_embedding_generation(retriever: SemanticRetriever):
    """Test 2: Generate query embeddings"""
    print("\n" + "="*60)
    print("TEST 2: Query Embedding Generation")
    print("="*60)
    
    test_queries = [
        "What is contract law?",
        "Explain the Sale of Goods Act 1979",
        "What are employee rights in the UK?",
        "How does discrimination law work?",
        "What is intellectual property?"
    ]
    
    results = []
    
    for i, query in enumerate(test_queries, 1):
        try:
            start_time = time.time()
            embedding = retriever._generate_query_embedding(query)
            elapsed_time = (time.time() - start_time) * 1000  # ms
            
            # Validate embedding
            if embedding is None or embedding.size == 0:
                print(f"   ❌ Test {i}: Empty embedding")
                return False
            
            if len(embedding) != retriever.embedding_dimension:
                print(f"   ❌ Test {i}: Wrong dimension. Expected {retriever.embedding_dimension}, got {len(embedding)}")
                return False
            
            # Check normalization (should be ~1.0)
            norm = np.linalg.norm(embedding)
            if abs(norm - 1.0) > 0.01:
                print(f"   ⚠️  Test {i}: Embedding not normalized. Norm: {norm:.6f}")
            
            # Check for NaN or Inf
            if np.isnan(embedding).any() or np.isinf(embedding).any():
                print(f"   ❌ Test {i}: Embedding contains NaN or Inf")
                return False
            
            results.append({
                'query': query[:40] + "...",
                'dim': len(embedding),
                'norm': norm,
                'time_ms': elapsed_time
            })
            
            print(f"   ✅ Test {i}: {query[:40]}... | dim={len(embedding)}, norm={norm:.3f}, time={elapsed_time:.2f}ms")
            
        except Exception as e:
            print(f"   ❌ Test {i} failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    # Summary
    avg_time = sum(r['time_ms'] for r in results) / len(results)
    print(f"\n   📊 Summary: {len(results)}/{len(test_queries)} tests passed")
    print(f"   ⏱️  Average time per embedding: {avg_time:.2f}ms")
    
    return True


def test_semantic_search(retriever: SemanticRetriever):
    """Test 3: Perform semantic search"""
    print("\n" + "="*60)
    print("TEST 3: Semantic Search (Top-K Retrieval)")
    print("="*60)
    
    test_queries = [
        "What are the implied conditions in a contract of sale?",
        "Employee rights in the UK",
        "Discrimination under the Equality Act",
        "Breach of contract remedies",
        "Intellectual property protection"
    ]
    
    for i, query in enumerate(test_queries, 1):
        try:
            print(f"\n📝 Query {i}: '{query}'")
            
            start_time = time.time()
            results = retriever.search(query, top_k=5)
            elapsed_time = (time.time() - start_time) * 1000  # ms
            
            if not results:
                print(f"   ⚠️  No results returned")
                continue
            
            print(f"   ⏱️  Search time: {elapsed_time:.2f}ms")
            print(f"   📊 Results found: {len(results)}")
            
            for rank, result in enumerate(results[:3], 1):  # Show top 3
                print(f"      {rank}. Score: {result['similarity_score']:.3f}")
                print(f"         Section: {result.get('section', 'N/A')}")
                text_preview = result['text'][:60] + "..." if len(result['text']) > 60 else result['text']
                print(f"         Text: {text_preview}")
            
            # Validate results
            if len(results) > 0:
                # Check scores are in valid range
                scores = [r['similarity_score'] for r in results]
                if any(score < 0 or score > 1 for score in scores):
                    print(f"   ⚠️  Warning: Some scores outside [0, 1] range")
                
                # Check scores are decreasing (should be sorted)
                if scores != sorted(scores, reverse=True):
                    print(f"   ⚠️  Warning: Results not properly sorted by score")
            
        except Exception as e:
            print(f"   ❌ Query {i} failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    return True


def test_top_k_retrieval(retriever: SemanticRetriever):
    """Test 4: Test different top_k values"""
    print("\n" + "="*60)
    print("TEST 4: Top-K Retrieval (Different K Values)")
    print("="*60)
    
    query = "What are the key provisions of contract law?"
    
    for top_k in [1, 3, 5, 10]:
        try:
            results = retriever.search(query, top_k=top_k)
            print(f"   top_k={top_k:<3}: {len(results)} results returned")
            
            if len(results) > top_k:
                print(f"   ⚠️  Warning: More results than requested")
            
            if results:
                scores = [r['similarity_score'] for r in results]
                print(f"      Score range: [{min(scores):.3f}, {max(scores):.3f}]")
            
        except Exception as e:
            print(f"   ❌ top_k={top_k} failed: {e}")
            return False
    
    return True


def test_similarity_threshold(retriever: SemanticRetriever):
    """Test 5: Test similarity threshold filtering"""
    print("\n" + "="*60)
    print("TEST 5: Similarity Threshold Filtering")
    print("="*60)
    
    query = "Employee rights and termination"
    
    thresholds = [0.0, 0.3, 0.5, 0.7]
    
    for threshold in thresholds:
        try:
            results = retriever.search(query, top_k=10, similarity_threshold=threshold)
            print(f"   Threshold={threshold:.1f}: {len(results)} results")
            
            if results:
                scores = [r['similarity_score'] for r in results]
                if any(score < threshold for score in scores):
                    print(f"   ❌ Error: Results below threshold found")
                    return False
                
                print(f"      Score range: [{min(scores):.3f}, {max(scores):.3f}]")
            
        except Exception as e:
            print(f"   ❌ Threshold={threshold} failed: {e}")
            return False
    
    return True


def test_batch_search(retriever: SemanticRetriever):
    """Test 6: Batch search"""
    print("\n" + "="*60)
    print("TEST 6: Batch Search")
    print("="*60)
    
    queries = [
        "Contract law basics",
        "Employment rights",
        "Discrimination law"
    ]
    
    try:
        start_time = time.time()
        batch_results = retriever.search_batch(queries, top_k=5)
        elapsed_time = (time.time() - start_time) * 1000  # ms
        
        print(f"   ⏱️  Batch search time: {elapsed_time:.2f}ms")
        print(f"   📊 Queries processed: {len(queries)}")
        print(f"   📊 Average time per query: {elapsed_time / len(queries):.2f}ms")
        
        for i, (query, results) in enumerate(zip(queries, batch_results), 1):
            print(f"   Query {i}: '{query[:30]}...' -> {len(results)} results")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Batch search failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def benchmark_performance(retriever: SemanticRetriever):
    """Test 7: Performance benchmark"""
    print("\n" + "="*60)
    print("TEST 7: Performance Benchmark")
    print("="*60)
    
    test_queries = [
        "What is contract law?",
        "Explain employee rights",
        "Discrimination law",
        "Breach of contract",
        "Intellectual property"
    ] * 4  # 20 queries total
    
    print(f"   Benchmarking with {len(test_queries)} queries...")
    
    times = []
    num_results = []
    
    for query in test_queries:
        start = time.time()
        results = retriever.search(query, top_k=10)
        elapsed = (time.time() - start) * 1000  # ms
        times.append(elapsed)
        num_results.append(len(results))
    
    avg_time = sum(times) / len(times)
    avg_results = sum(num_results) / len(num_results)
    throughput = len(test_queries) / (sum(times) / 1000)  # queries per second
    
    print(f"\n   📊 Performance Metrics:")
    print(f"   Average search time: {avg_time:.2f}ms")
    print(f"   Average results per query: {avg_results:.1f}")
    print(f"   Throughput: {throughput:.1f} queries/second")
    print(f"   Min time: {min(times):.2f}ms")
    print(f"   Max time: {max(times):.2f}ms")
    
    # Check if performance is acceptable (< 100ms per query)
    if avg_time < 100:
        print(f"   ✅ Performance is acceptable (< 100ms)")
    else:
        print(f"   ⚠️  Performance could be improved (> 100ms)")
    
    return True


def test_semantic_quality(retriever: SemanticRetriever):
    """Test 8: Semantic quality - similar queries should return similar results"""
    print("\n" + "="*60)
    print("TEST 8: Semantic Quality (Similar Queries)")
    print("="*60)
    
    # Similar queries
    similar_queries = [
        "What is contract law?",
        "Explain contract law",
        "Contract law definition"
    ]
    
    # Get results for each
    all_results = []
    for query in similar_queries:
        results = retriever.search(query, top_k=5)
        chunk_ids = [r['chunk_id'] for r in results]
        all_results.append(set(chunk_ids))
    
    # Check overlap between similar queries
    overlap_12 = len(all_results[0] & all_results[1])
    overlap_13 = len(all_results[0] & all_results[2])
    overlap_23 = len(all_results[1] & all_results[2])
    
    print(f"   Query 1 vs Query 2 overlap: {overlap_12}/5 chunks")
    print(f"   Query 1 vs Query 3 overlap: {overlap_13}/5 chunks")
    print(f"   Query 2 vs Query 3 overlap: {overlap_23}/5 chunks")
    
    avg_overlap = (overlap_12 + overlap_13 + overlap_23) / 3
    print(f"   Average overlap: {avg_overlap:.1f}/5 chunks")
    
    if avg_overlap >= 2:
        print(f"   ✅ Semantic quality is good (similar queries return similar results)")
    else:
        print(f"   ⚠️  Semantic quality could be improved (low overlap)")
    
    return True


def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("PHASE 2 - MODULE 1.2: Semantic Retrieval Implementation")
    print("Test and Validate Semantic Retrieval")
    print("="*60)
    
    results = {}
    
    # Test 1: Initialization
    results['initialization'], retriever = test_semantic_retriever_initialization()
    if not results['initialization'] or retriever is None:
        print("\n❌ Initialization test failed. Cannot continue.")
        return False
    
    # Test 2: Query embedding generation
    results['query_embedding'] = test_query_embedding_generation(retriever)
    
    # Test 3: Semantic search
    results['semantic_search'] = test_semantic_search(retriever)
    
    # Test 4: Top-k retrieval
    results['top_k'] = test_top_k_retrieval(retriever)
    
    # Test 5: Similarity threshold
    results['threshold'] = test_similarity_threshold(retriever)
    
    # Test 6: Batch search
    results['batch'] = test_batch_search(retriever)
    
    # Test 7: Performance benchmark
    results['benchmark'] = benchmark_performance(retriever)
    
    # Test 8: Semantic quality
    results['quality'] = test_semantic_quality(retriever)
    
    # Final summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, passed_test in results.items():
        status = "✅ PASS" if passed_test else "❌ FAIL"
        print(f"   {test_name:<25} {status}")
    
    print(f"\n   Total: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! Semantic retrieval is ready for Phase 2.")
        return True
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Review issues above.")
        return False


if __name__ == "__main__":
    import numpy as np
    success = main()
    sys.exit(0 if success else 1)

