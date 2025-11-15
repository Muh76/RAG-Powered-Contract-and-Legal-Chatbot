#!/usr/bin/env python3
# Legal Chatbot - Test Cross-Encoder Reranking
# Phase 2: Test Reranking Implementation

import os
import sys
from pathlib import Path
import logging

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from retrieval.rerankers.cross_encoder_reranker import CrossEncoderReranker

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_reranker_initialization():
    """Test reranker initialization"""
    print("=" * 60)
    print("1️⃣ Testing Cross-Encoder Reranker Initialization")
    print("=" * 60)
    
    try:
        reranker = CrossEncoderReranker()
        if reranker.is_ready():
            print("✅ Reranker initialized successfully")
            stats = reranker.get_stats()
            print(f"   Model: {stats['model_name']}")
            print(f"   Device: {stats['device']}")
            print(f"   Batch size: {stats['batch_size']}")
            return reranker
        else:
            print("❌ Reranker not ready")
            return None
    except Exception as e:
        print(f"❌ Failed to initialize reranker: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_simple_reranking(reranker):
    """Test simple reranking"""
    print("\n" + "=" * 60)
    print("2️⃣ Testing Simple Reranking")
    print("=" * 60)
    
    if not reranker or not reranker.is_ready():
        print("⚠️ Skipping: Reranker not available")
        return
    
    query = "What are the implied conditions in a contract of sale?"
    
    documents = [
        "A contract of sale includes implied conditions about the quality and fitness of goods.",
        "The Sale of Goods Act 1979 sets out implied conditions in contracts.",
        "Implied conditions are automatically included in contracts by law.",
        "Cake recipes often include chocolate and vanilla as ingredients.",
        "Contracts must have offer, acceptance, and consideration."
    ]
    
    print(f"Query: {query}")
    print(f"Documents: {len(documents)}")
    
    try:
        reranked = reranker.rerank(query, documents, top_k=3)
        
        print(f"\n✅ Reranking completed")
        print(f"   Top {len(reranked)} results:")
        for i, result in enumerate(reranked, 1):
            print(f"\n   {i}. Rank {result['rank']}: Score {result['score']:.3f}")
            text_preview = result['text'][:80] + "..." if len(result['text']) > 80 else result['text']
            print(f"      Text: {text_preview}")
    except Exception as e:
        print(f"❌ Reranking failed: {e}")
        import traceback
        traceback.print_exc()


def test_reranking_with_metadata(reranker):
    """Test reranking with retrieval results"""
    print("\n" + "=" * 60)
    print("3️⃣ Testing Reranking with Metadata")
    print("=" * 60)
    
    if not reranker or not reranker.is_ready():
        print("⚠️ Skipping: Reranker not available")
        return
    
    query = "What are employee rights in the UK?"
    
    retrieval_results = [
        {
            "chunk_id": "chunk_1",
            "text": "Employees in the UK have various rights under employment law.",
            "similarity_score": 0.75,
            "section": "Employment Rights Act 1996",
            "metadata": {"document_type": "statute", "jurisdiction": "UK"}
        },
        {
            "chunk_id": "chunk_2",
            "text": "Employment contracts must include terms about rights and obligations.",
            "similarity_score": 0.68,
            "section": "Employment Rights",
            "metadata": {"document_type": "contract", "jurisdiction": "UK"}
        },
        {
            "chunk_id": "chunk_3",
            "text": "The weather in the UK is often rainy and cloudy.",
            "similarity_score": 0.45,
            "section": "Weather Info",
            "metadata": {"document_type": "non_legal", "jurisdiction": "UK"}
        }
    ]
    
    print(f"Query: {query}")
    print(f"Retrieval results: {len(retrieval_results)}")
    
    try:
        reranked = reranker.rerank_results(
            query=query,
            retrieval_results=retrieval_results,
            top_k=2,
            preserve_metadata=True
        )
        
        print(f"\n✅ Reranking completed")
        print(f"   Top {len(reranked)} results:")
        for i, result in enumerate(reranked, 1):
            print(f"\n   {i}. Rank {result.get('rerank_rank', result.get('rank', i))}")
            print(f"      Rerank Score: {result.get('rerank_score', 'N/A')}")
            print(f"      Original Score: {result.get('similarity_score', 'N/A')}")
            print(f"      Section: {result.get('section', 'N/A')}")
            text_preview = result['text'][:70] + "..." if len(result['text']) > 70 else result['text']
            print(f"      Text: {text_preview}")
    except Exception as e:
        print(f"❌ Reranking failed: {e}")
        import traceback
        traceback.print_exc()


def test_performance(reranker):
    """Test reranking performance"""
    print("\n" + "=" * 60)
    print("4️⃣ Testing Reranking Performance")
    print("=" * 60)
    
    if not reranker or not reranker.is_ready():
        print("⚠️ Skipping: Reranker not available")
        return
    
    import time
    
    query = "What is contract law?"
    documents = [
        f"Document {i}: This is a test document about contract law and legal matters. " * 5
        for i in range(20)
    ]
    
    print(f"Testing with {len(documents)} documents...")
    
    try:
        start_time = time.time()
        reranked = reranker.rerank(query, documents, top_k=10)
        elapsed_ms = (time.time() - start_time) * 1000
        
        print(f"✅ Reranking completed")
        print(f"   Time: {elapsed_ms:.2f}ms")
        print(f"   Throughput: {len(documents) / (elapsed_ms / 1000):.1f} docs/second")
        print(f"   Results returned: {len(reranked)}")
    except Exception as e:
        print(f"❌ Performance test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("🧪 Cross-Encoder Reranking Test Suite")
    print("=" * 60)
    
    # Test 1: Initialization
    reranker = test_reranker_initialization()
    
    # Test 2: Simple reranking
    test_simple_reranking(reranker)
    
    # Test 3: Reranking with metadata
    test_reranking_with_metadata(reranker)
    
    # Test 4: Performance
    test_performance(reranker)
    
    print("\n" + "=" * 60)
    print("✅ Reranking test suite completed!")
    print("=" * 60)

