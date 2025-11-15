#!/usr/bin/env python3
"""
Test and Validate Hybrid Retrieval System
Phase 2: Module 2 & 3 - Hybrid Search with Metadata Filtering

This script tests:
1. BM25Retriever
2. MetadataFilter
3. AdvancedHybridRetriever with RRF and weighted fusion
4. Integration with metadata filtering
"""

import os
import sys
import time
import pickle
import logging
from pathlib import Path
from typing import List, Dict, Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from retrieval.bm25_retriever import BM25Retriever
from retrieval.semantic_retriever import SemanticRetriever
from retrieval.metadata_filter import MetadataFilter, FilterOperator
from retrieval.hybrid_retriever import AdvancedHybridRetriever, FusionStrategy

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_chunk_metadata() -> List[Dict[str, Any]]:
    """Load chunk metadata from file."""
    possible_paths = [
        project_root / "data" / "chunk_metadata.pkl",
        project_root / "notebooks" / "phase1" / "data" / "chunk_metadata.pkl",
        Path("data/chunk_metadata.pkl"),
        Path("notebooks/phase1/data/chunk_metadata.pkl"),
    ]
    
    for path in possible_paths:
        if path.exists():
            try:
                with open(path, "rb") as f:
                    chunk_metadata = pickle.load(f)
                logger.info(f"Loaded {len(chunk_metadata)} chunks from {path}")
                return chunk_metadata
            except Exception as e:
                logger.error(f"Error loading {path}: {e}")
    
    logger.warning("No chunk_metadata.pkl found, creating sample data")
    # Create sample data for testing
    return [
        {
            "chunk_id": "chunk_0",
            "text": "This is a test document about contract law. It covers various aspects of contracts.",
            "metadata": {
                "title": "Contract Law Guide",
                "source": "test",
                "jurisdiction": "UK",
                "document_type": "statute",
                "section": "Section 1"
            }
        },
        {
            "chunk_id": "chunk_1",
            "text": "Employment law covers the rights and duties between employers and employees.",
            "metadata": {
                "title": "Employment Law",
                "source": "test",
                "jurisdiction": "UK",
                "document_type": "statute",
                "section": "Section 2"
            }
        },
        {
            "chunk_id": "chunk_2",
            "text": "Discrimination law prohibits discrimination based on protected characteristics.",
            "metadata": {
                "title": "Equality Act",
                "source": "test",
                "jurisdiction": "UK",
                "document_type": "statute",
                "section": "Section 3"
            }
        }
    ]


def test_bm25_retriever():
    """Test 1: BM25 Retriever"""
    print("\n" + "="*60)
    print("TEST 1: BM25 Retriever")
    print("="*60)
    
    try:
        chunk_metadata = load_chunk_metadata()
        documents = [chunk.get("text", "") for chunk in chunk_metadata]
        
        print(f"   Initializing BM25 retriever with {len(documents)} documents...")
        bm25 = BM25Retriever(documents)
        
        stats = bm25.get_index_stats()
        print(f"   ✅ BM25 retriever initialized")
        print(f"   Documents: {stats['num_documents']}")
        print(f"   Vocabulary: {stats['vocabulary_size']} terms")
        
        # Test search
        query = "contract law"
        results = bm25.search(query, top_k=3)
        print(f"\n   Query: '{query}'")
        print(f"   Results: {len(results)} chunks found")
        
        for rank, (doc_idx, score) in enumerate(results, 1):
            doc_text = bm25.get_document(doc_idx)
            preview = doc_text[:60] + "..." if len(doc_text) > 60 else doc_text
            print(f"      {rank}. Score: {score:.3f} - {preview}")
        
        return True, bm25
        
    except Exception as e:
        print(f"   ❌ BM25 retriever test failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def test_metadata_filter():
    """Test 2: Metadata Filter"""
    print("\n" + "="*60)
    print("TEST 2: Metadata Filter")
    print("="*60)
    
    try:
        chunk_metadata = load_chunk_metadata()
        
        # Create filter
        metadata_filter = MetadataFilter()
        metadata_filter.add_equals_filter("jurisdiction", "UK")
        
        print(f"   Filtering {len(chunk_metadata)} chunks...")
        filtered = metadata_filter.filter_chunks(chunk_metadata)
        print(f"   ✅ Filtered to {len(filtered)} chunks (jurisdiction=UK)")
        
        # Test multiple filters
        metadata_filter2 = MetadataFilter()
        metadata_filter2.add_equals_filter("jurisdiction", "UK")
        metadata_filter2.add_equals_filter("document_type", "statute")
        
        filtered2 = metadata_filter2.filter_chunks(chunk_metadata)
        print(f"   ✅ Multiple filters: {len(filtered2)} chunks (jurisdiction=UK AND document_type=statute)")
        
        # Test IN filter
        metadata_filter3 = MetadataFilter()
        metadata_filter3.add_in_filter("document_type", ["statute", "contract"])
        
        filtered3 = metadata_filter3.filter_chunks(chunk_metadata)
        print(f"   ✅ IN filter: {len(filtered3)} chunks (document_type IN [statute, contract])")
        
        return True, metadata_filter
        
    except Exception as e:
        print(f"   ❌ Metadata filter test failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def test_hybrid_retriever_rrf():
    """Test 3: Hybrid Retriever with RRF Fusion"""
    print("\n" + "="*60)
    print("TEST 3: Hybrid Retriever (RRF Fusion)")
    print("="*60)
    
    try:
        chunk_metadata = load_chunk_metadata()
        documents = [chunk.get("text", "") for chunk in chunk_metadata]
        
        # Initialize retrievers
        bm25 = BM25Retriever(documents)
        semantic = SemanticRetriever()
        
        if not semantic.is_ready():
            print("   ⚠️  Semantic retriever not ready, skipping test")
            return False, None
        
        # Create hybrid retriever with RRF
        hybrid = AdvancedHybridRetriever(
            bm25_retriever=bm25,
            semantic_retriever=semantic,
            chunk_metadata=chunk_metadata,
            fusion_strategy=FusionStrategy.RRF
        )
        
        print(f"   ✅ Hybrid retriever initialized (RRF)")
        
        # Test search
        query = "contract law"
        print(f"\n   Query: '{query}'")
        
        start_time = time.time()
        results = hybrid.search(query, top_k=3)
        elapsed_time = (time.time() - start_time) * 1000
        
        print(f"   ⏱️  Search time: {elapsed_time:.2f}ms")
        print(f"   Results: {len(results)} chunks found")
        
        for result in results:
            print(f"      Rank {result['rank']}: Score: {result['similarity_score']:.3f}")
            print(f"         BM25: {result.get('bm25_score', 'N/A')}, "
                  f"Semantic: {result.get('semantic_score', 'N/A')}")
            text_preview = result['text'][:60] + "..." if len(result['text']) > 60 else result['text']
            print(f"         Text: {text_preview}")
        
        return True, hybrid
        
    except Exception as e:
        print(f"   ❌ Hybrid retriever (RRF) test failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def test_hybrid_retriever_weighted():
    """Test 4: Hybrid Retriever with Weighted Fusion"""
    print("\n" + "="*60)
    print("TEST 4: Hybrid Retriever (Weighted Fusion)")
    print("="*60)
    
    try:
        chunk_metadata = load_chunk_metadata()
        documents = [chunk.get("text", "") for chunk in chunk_metadata]
        
        # Initialize retrievers
        bm25 = BM25Retriever(documents)
        semantic = SemanticRetriever()
        
        if not semantic.is_ready():
            print("   ⚠️  Semantic retriever not ready, skipping test")
            return False, None
        
        # Create hybrid retriever with weighted fusion
        hybrid = AdvancedHybridRetriever(
            bm25_retriever=bm25,
            semantic_retriever=semantic,
            chunk_metadata=chunk_metadata,
            fusion_strategy=FusionStrategy.WEIGHTED,
            bm25_weight=0.4,
            semantic_weight=0.6
        )
        
        print(f"   ✅ Hybrid retriever initialized (Weighted: 40% BM25, 60% Semantic)")
        
        # Test search
        query = "employment rights"
        print(f"\n   Query: '{query}'")
        
        results = hybrid.search(query, top_k=3)
        print(f"   Results: {len(results)} chunks found")
        
        for result in results:
            print(f"      Rank {result['rank']}: Score: {result['similarity_score']:.3f}")
            print(f"         BM25: {result.get('bm25_score', 'N/A')}, "
                  f"Semantic: {result.get('semantic_score', 'N/A')}")
        
        return True, hybrid
        
    except Exception as e:
        print(f"   ❌ Hybrid retriever (Weighted) test failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def test_hybrid_with_metadata_filter():
    """Test 5: Hybrid Retriever with Metadata Filtering"""
    print("\n" + "="*60)
    print("TEST 5: Hybrid Retriever with Metadata Filtering")
    print("="*60)
    
    try:
        chunk_metadata = load_chunk_metadata()
        documents = [chunk.get("text", "") for chunk in chunk_metadata]
        
        # Initialize retrievers
        bm25 = BM25Retriever(documents)
        semantic = SemanticRetriever()
        
        if not semantic.is_ready():
            print("   ⚠️  Semantic retriever not ready, skipping test")
            return False, None
        
        # Create hybrid retriever
        hybrid = AdvancedHybridRetriever(
            bm25_retriever=bm25,
            semantic_retriever=semantic,
            chunk_metadata=chunk_metadata
        )
        
        # Create metadata filter
        metadata_filter = MetadataFilter()
        metadata_filter.add_equals_filter("jurisdiction", "UK")
        
        print(f"   ✅ Hybrid retriever initialized")
        print(f"   Filter: jurisdiction=UK")
        
        # Test search with post-filter
        query = "contract law"
        print(f"\n   Query: '{query}' (post-filter)")
        
        results = hybrid.search(
            query,
            top_k=5,
            metadata_filter=metadata_filter,
            pre_filter=False
        )
        
        print(f"   Results: {len(results)} chunks found (after filtering)")
        
        for result in results:
            print(f"      Rank {result['rank']}: Score: {result['similarity_score']:.3f}")
            print(f"         Jurisdiction: {result.get('jurisdiction', 'N/A')}")
        
        # Test search with pre-filter
        print(f"\n   Query: '{query}' (pre-filter)")
        
        results2 = hybrid.search(
            query,
            top_k=5,
            metadata_filter=metadata_filter,
            pre_filter=True
        )
        
        print(f"   Results: {len(results2)} chunks found (after pre-filtering)")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Hybrid retriever with metadata filter test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_fusion_strategies_comparison():
    """Test 6: Compare RRF vs Weighted Fusion"""
    print("\n" + "="*60)
    print("TEST 6: Fusion Strategies Comparison")
    print("="*60)
    
    try:
        chunk_metadata = load_chunk_metadata()
        documents = [chunk.get("text", "") for chunk in chunk_metadata]
        
        # Initialize retrievers
        bm25 = BM25Retriever(documents)
        semantic = SemanticRetriever()
        
        if not semantic.is_ready():
            print("   ⚠️  Semantic retriever not ready, skipping test")
            return False
        
        query = "contract law"
        
        # Test RRF
        hybrid_rrf = AdvancedHybridRetriever(
            bm25_retriever=bm25,
            semantic_retriever=semantic,
            chunk_metadata=chunk_metadata,
            fusion_strategy=FusionStrategy.RRF
        )
        
        results_rrf = hybrid_rrf.search(query, top_k=3)
        
        # Test Weighted
        hybrid_weighted = AdvancedHybridRetriever(
            bm25_retriever=bm25,
            semantic_retriever=semantic,
            chunk_metadata=chunk_metadata,
            fusion_strategy=FusionStrategy.WEIGHTED
        )
        
        results_weighted = hybrid_weighted.search(query, top_k=3)
        
        print(f"   Query: '{query}'")
        print(f"\n   RRF Fusion Results:")
        for result in results_rrf:
            print(f"      Rank {result['rank']}: Score: {result['similarity_score']:.3f}")
        
        print(f"\n   Weighted Fusion Results:")
        for result in results_weighted:
            print(f"      Rank {result['rank']}: Score: {result['similarity_score']:.3f}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Fusion strategies comparison failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("PHASE 2 - MODULE 2 & 3: Hybrid Search with Metadata Filtering")
    print("Test and Validate Hybrid Retrieval System")
    print("="*60)
    
    results = {}
    
    # Test 1: BM25 Retriever
    results['bm25'], bm25_retriever = test_bm25_retriever()
    
    # Test 2: Metadata Filter
    results['metadata_filter'], metadata_filter = test_metadata_filter()
    
    # Test 3: Hybrid Retriever (RRF)
    results['hybrid_rrf'], hybrid_rrf = test_hybrid_retriever_rrf()
    
    # Test 4: Hybrid Retriever (Weighted)
    results['hybrid_weighted'], hybrid_weighted = test_hybrid_retriever_weighted()
    
    # Test 5: Hybrid with Metadata Filtering
    results['hybrid_filter'] = test_hybrid_with_metadata_filter()
    
    # Test 6: Fusion Strategies Comparison
    results['fusion_comparison'] = test_fusion_strategies_comparison()
    
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
        print("\n🎉 ALL TESTS PASSED! Hybrid retrieval system is ready for Phase 2.")
        return True
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Review issues above.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

