#!/usr/bin/env python3
"""
Test and Benchmark Embedding Generation
Phase 2: Module 1.1 - Resolve PyTorch/Embedding Issues

This script tests and benchmarks the EmbeddingGenerator to ensure:
1. PyTorch is working correctly
2. SentenceTransformers loads successfully
3. Embedding generation works (single and batch)
4. Performance benchmarks are acceptable
"""

import os
import sys
import time
import logging
from pathlib import Path
from typing import List
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from retrieval.embeddings.embedding_generator import EmbeddingGenerator, EmbeddingConfig
from app.core.config import settings

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_pytorch_import():
    """Test 1: Verify PyTorch can be imported and used"""
    print("\n" + "="*60)
    print("TEST 1: PyTorch Import & Basic Operations")
    print("="*60)
    
    try:
        import torch
        print(f"✅ PyTorch imported successfully")
        print(f"   Version: {torch.__version__}")
        print(f"   CUDA available: {torch.cuda.is_available()}")
        
        # Test basic tensor operations
        x = torch.randn(3, 4)
        y = torch.randn(4, 5)
        z = torch.matmul(x, y)
        print(f"   ✅ Basic tensor operations work: {z.shape}")
        
        return True
    except Exception as e:
        print(f"❌ PyTorch import failed: {e}")
        return False


def test_sentence_transformers_import():
    """Test 2: Verify SentenceTransformers can be imported"""
    print("\n" + "="*60)
    print("TEST 2: SentenceTransformers Import")
    print("="*60)
    
    try:
        from sentence_transformers import SentenceTransformer
        print(f"✅ SentenceTransformers imported successfully")
        
        # Test model loading
        model_name = settings.EMBEDDING_MODEL
        print(f"   Loading model: {model_name}")
        model = SentenceTransformer(model_name)
        print(f"   ✅ Model loaded successfully")
        print(f"   Model max sequence length: {model.max_seq_length}")
        
        return True, model
    except Exception as e:
        print(f"❌ SentenceTransformers import/loading failed: {e}")
        return False, None


def test_embedding_generator_init():
    """Test 3: Initialize EmbeddingGenerator"""
    print("\n" + "="*60)
    print("TEST 3: EmbeddingGenerator Initialization")
    print("="*60)
    
    try:
        config = EmbeddingConfig(
            model_name=settings.EMBEDDING_MODEL,
            dimension=settings.EMBEDDING_DIMENSION,
            batch_size=settings.EMBEDDING_BATCH_SIZE,
            max_length=512
        )
        
        print(f"   Config: model={config.model_name}, dim={config.dimension}")
        print(f"   Initializing EmbeddingGenerator...")
        
        embedding_gen = EmbeddingGenerator(config)
        
        if embedding_gen.model is None:
            print(f"❌ EmbeddingGenerator initialized but model is None")
            return False, None
        
        print(f"   ✅ EmbeddingGenerator initialized successfully")
        print(f"   Model dimension: {embedding_gen.get_embedding_dimension()}")
        
        return True, embedding_gen
    except Exception as e:
        print(f"❌ EmbeddingGenerator initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def test_single_embedding_generation(embedding_gen: EmbeddingGenerator):
    """Test 4: Generate single embedding"""
    print("\n" + "="*60)
    print("TEST 4: Single Embedding Generation")
    print("="*60)
    
    test_texts = [
        "This is a test legal document about contract law.",
        "The Sale of Goods Act 1979 governs contracts for the sale of goods.",
        "Employment law covers the rights and duties between employers and employees.",
        "What are the implied conditions in a contract of sale?",
        "Discrimination under the Equality Act 2010 is prohibited."
    ]
    
    results = []
    
    for i, text in enumerate(test_texts, 1):
        try:
            start_time = time.time()
            embedding = embedding_gen.generate_embedding(text)
            elapsed_time = (time.time() - start_time) * 1000  # ms
            
            # Validate embedding
            if embedding is None:
                print(f"   ❌ Test {i}: Embedding is None")
                return False
            
            if not isinstance(embedding, list):
                print(f"   ❌ Test {i}: Embedding is not a list")
                return False
            
            if len(embedding) != embedding_gen.get_embedding_dimension():
                print(f"   ❌ Test {i}: Wrong dimension. Expected {embedding_gen.get_embedding_dimension()}, got {len(embedding)}")
                return False
            
            # Check for NaN or Inf
            embedding_array = np.array(embedding)
            if np.isnan(embedding_array).any() or np.isinf(embedding_array).any():
                print(f"   ❌ Test {i}: Embedding contains NaN or Inf values")
                return False
            
            # Check norm (should be reasonable)
            norm = np.linalg.norm(embedding_array)
            if norm == 0 or norm > 100:
                print(f"   ⚠️  Test {i}: Suspicious norm: {norm:.3f}")
            
            results.append({
                'text': text[:50] + "...",
                'dim': len(embedding),
                'norm': norm,
                'time_ms': elapsed_time
            })
            
            print(f"   ✅ Test {i}: {text[:40]}... | dim={len(embedding)}, norm={norm:.3f}, time={elapsed_time:.2f}ms")
            
        except Exception as e:
            print(f"   ❌ Test {i} failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    # Summary
    avg_time = np.mean([r['time_ms'] for r in results])
    print(f"\n   📊 Summary: {len(results)}/5 tests passed")
    print(f"   ⏱️  Average time per embedding: {avg_time:.2f}ms")
    
    return True


def test_batch_embedding_generation(embedding_gen: EmbeddingGenerator):
    """Test 5: Generate embeddings in batch"""
    print("\n" + "="*60)
    print("TEST 5: Batch Embedding Generation")
    print("="*60)
    
    # Create test texts with varying lengths
    test_texts = [
        "Short text.",
        "This is a medium length legal document about contract law and employment rights.",
        "The Sale of Goods Act 1979 governs contracts for the sale of goods in the United Kingdom. This legislation establishes the legal framework for commercial transactions and consumer protection. It covers various aspects including implied terms, conditions, warranties, and remedies for breach of contract.",
        "What are the key provisions of the Equality Act 2010?",
        "Employment law covers a wide range of topics including minimum wage, working hours, health and safety, discrimination, unfair dismissal, redundancy, and collective bargaining rights.",
        "Contract law principles include offer, acceptance, consideration, intention to create legal relations, capacity, and legality of purpose.",
        "Intellectual property law encompasses patents, trademarks, copyrights, and trade secrets.",
        "Data protection is governed by the UK GDPR and Data Protection Act 2018."
    ]
    
    batch_sizes = [1, 2, 4, 8, len(test_texts)]
    results = []
    
    for batch_size in batch_sizes:
        try:
            texts_batch = test_texts[:batch_size]
            
            start_time = time.time()
            embeddings = embedding_gen.generate_embeddings_batch(texts_batch)
            elapsed_time = (time.time() - start_time) * 1000  # ms
            
            # Validate
            if embeddings is None or len(embeddings) != len(texts_batch):
                print(f"   ❌ Batch size {batch_size}: Wrong number of embeddings")
                return False
            
            # Check each embedding
            for i, emb in enumerate(embeddings):
                if len(emb) != embedding_gen.get_embedding_dimension():
                    print(f"   ❌ Batch size {batch_size}, embedding {i}: Wrong dimension")
                    return False
            
            time_per_embedding = elapsed_time / len(texts_batch)
            results.append({
                'batch_size': batch_size,
                'total_time_ms': elapsed_time,
                'time_per_embedding_ms': time_per_embedding
            })
            
            print(f"   ✅ Batch size {batch_size}: {len(embeddings)} embeddings in {elapsed_time:.2f}ms ({time_per_embedding:.2f}ms/embedding)")
            
        except Exception as e:
            print(f"   ❌ Batch size {batch_size} failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    # Summary
    print(f"\n   📊 Batch Processing Summary:")
    print(f"   {'Batch Size':<12} {'Total Time (ms)':<18} {'Time per Embedding (ms)':<25}")
    print(f"   {'-'*12} {'-'*18} {'-'*25}")
    for r in results:
        print(f"   {r['batch_size']:<12} {r['total_time_ms']:<18.2f} {r['time_per_embedding_ms']:<25.2f}")
    
    # Check if batching is more efficient
    single_time = results[0]['time_per_embedding_ms']
    batch_time = results[-1]['time_per_embedding_ms']
    speedup = single_time / batch_time if batch_time > 0 else 0
    print(f"\n   🚀 Batch efficiency: {speedup:.2f}x faster (batch vs single)")
    
    return True


def benchmark_embedding_performance(embedding_gen: EmbeddingGenerator):
    """Test 6: Performance benchmark"""
    print("\n" + "="*60)
    print("TEST 6: Performance Benchmark")
    print("="*60)
    
    # Create diverse test set
    test_texts = [
        "What is contract law?",
        "Explain the Sale of Goods Act 1979",
        "What are employee rights in the UK?",
        "How does discrimination law work?",
        "What is intellectual property?",
        "Explain data protection regulations",
        "What are the terms of this agreement?",
        "How can a contract be terminated?",
        "What remedies exist for breach of contract?",
        "What is unfair dismissal?"
    ] * 10  # 100 texts total
    
    print(f"   Benchmarking with {len(test_texts)} texts...")
    print(f"   Batch size: {embedding_gen.config.batch_size}")
    
    # Single embedding benchmark
    print(f"\n   Single embedding mode:")
    single_times = []
    for i, text in enumerate(test_texts[:10], 1):  # Test first 10
        start = time.time()
        _ = embedding_gen.generate_embedding(text)
        elapsed = (time.time() - start) * 1000
        single_times.append(elapsed)
    avg_single = np.mean(single_times)
    print(f"   Average time per embedding: {avg_single:.2f}ms")
    
    # Batch embedding benchmark
    print(f"\n   Batch embedding mode:")
    batch_times = []
    num_batches = 0
    for i in range(0, len(test_texts), embedding_gen.config.batch_size):
        batch = test_texts[i:i + embedding_gen.config.batch_size]
        start = time.time()
        _ = embedding_gen.generate_embeddings_batch(batch)
        elapsed = (time.time() - start) * 1000
        batch_times.append(elapsed)
        num_batches += 1
    
    avg_batch_total = np.mean(batch_times)
    avg_batch_per_embedding = avg_batch_total / embedding_gen.config.batch_size
    print(f"   Batches processed: {num_batches}")
    print(f"   Average batch time: {avg_batch_total:.2f}ms")
    print(f"   Average time per embedding (in batch): {avg_batch_per_embedding:.2f}ms")
    
    # Throughput
    total_time = sum(batch_times) / 1000  # seconds
    throughput = len(test_texts) / total_time
    print(f"\n   📈 Throughput: {throughput:.1f} embeddings/second")
    print(f"   ⏱️  Total time for {len(test_texts)} embeddings: {total_time:.2f}s")
    
    # Speedup
    speedup = avg_single / avg_batch_per_embedding if avg_batch_per_embedding > 0 else 0
    print(f"   🚀 Batch speedup: {speedup:.2f}x faster")
    
    return True


def test_embedding_consistency(embedding_gen: EmbeddingGenerator):
    """Test 7: Consistency - same text should produce same embedding"""
    print("\n" + "="*60)
    print("TEST 7: Embedding Consistency")
    print("="*60)
    
    test_text = "This is a test for embedding consistency."
    
    # Generate embedding multiple times
    embeddings = []
    for i in range(5):
        emb = embedding_gen.generate_embedding(test_text)
        embeddings.append(np.array(emb))
    
    # Check if all are identical (they should be)
    all_same = True
    for i in range(1, len(embeddings)):
        if not np.allclose(embeddings[0], embeddings[i], atol=1e-6):
            all_same = False
            diff = np.abs(embeddings[0] - embeddings[i]).max()
            print(f"   ❌ Embedding {i} differs from first (max diff: {diff:.2e})")
            return False
    
    if all_same:
        print(f"   ✅ All 5 embeddings are identical (deterministic)")
    
    return True


def test_embedding_similarity(embedding_gen: EmbeddingGenerator):
    """Test 8: Similar texts should have similar embeddings"""
    print("\n" + "="*60)
    print("TEST 8: Embedding Similarity (Semantic Coherence)")
    print("="*60)
    
    # Similar texts
    similar_pairs = [
        ("What is contract law?", "Explain contract law"),
        ("Employee rights in the UK", "Workers' rights in United Kingdom"),
        ("Breach of contract", "Contract violation")
    ]
    
    # Dissimilar texts
    dissimilar_pairs = [
        ("What is contract law?", "How do I cook pasta?"),
        ("Employee rights", "Weather forecast today"),
        ("Breach of contract", "Python programming tutorial")
    ]
    
    print(f"\n   Similar text pairs:")
    for text1, text2 in similar_pairs:
        emb1 = np.array(embedding_gen.generate_embedding(text1))
        emb2 = np.array(embedding_gen.generate_embedding(text2))
        
        # Cosine similarity
        similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        print(f"   '{text1[:30]}...' vs '{text2[:30]}...': {similarity:.3f}")
    
    print(f"\n   Dissimilar text pairs:")
    for text1, text2 in dissimilar_pairs:
        emb1 = np.array(embedding_gen.generate_embedding(text1))
        emb2 = np.array(embedding_gen.generate_embedding(text2))
        
        similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        print(f"   '{text1[:30]}...' vs '{text2[:30]}...': {similarity:.3f}")
    
    print(f"\n   ✅ Embedding similarity test completed (check manually that similar texts have higher similarity)")
    
    return True


def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("PHASE 2 - MODULE 1.1: PyTorch/Embedding Resolution")
    print("Test and Benchmark Embedding Generation")
    print("="*60)
    
    results = {}
    
    # Test 1: PyTorch
    results['pytorch'] = test_pytorch_import()
    if not results['pytorch']:
        print("\n❌ PyTorch test failed. Cannot continue.")
        return False
    
    # Test 2: SentenceTransformers
    results['sentence_transformers'], model = test_sentence_transformers_import()
    if not results['sentence_transformers']:
        print("\n❌ SentenceTransformers test failed. Cannot continue.")
        return False
    
    # Test 3: EmbeddingGenerator
    results['embedding_gen_init'], embedding_gen = test_embedding_generator_init()
    if not results['embedding_gen_init'] or embedding_gen is None:
        print("\n❌ EmbeddingGenerator initialization failed. Cannot continue.")
        return False
    
    # Test 4: Single embedding
    results['single_embedding'] = test_single_embedding_generation(embedding_gen)
    
    # Test 5: Batch embedding
    results['batch_embedding'] = test_batch_embedding_generation(embedding_gen)
    
    # Test 6: Performance benchmark
    results['benchmark'] = benchmark_embedding_performance(embedding_gen)
    
    # Test 7: Consistency
    results['consistency'] = test_embedding_consistency(embedding_gen)
    
    # Test 8: Similarity
    results['similarity'] = test_embedding_similarity(embedding_gen)
    
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
        print("\n🎉 ALL TESTS PASSED! Embedding system is ready for Phase 2.")
        return True
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Review issues above.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

