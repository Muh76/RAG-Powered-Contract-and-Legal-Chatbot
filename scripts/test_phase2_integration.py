#!/usr/bin/env python3
# Legal Chatbot - Phase 2 Integration Tests
# Test all Phase 2 features without model loading (to avoid PyTorch multiprocessing issues)

import os
import sys
from pathlib import Path
import logging

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_imports():
    """Test all Phase 2 imports"""
    print("=" * 60)
    print("1️⃣ Testing Phase 2 Imports")
    print("=" * 60)
    
    try:
        from retrieval.bm25_retriever import BM25Retriever
        print("✅ BM25Retriever imported")
    except Exception as e:
        print(f"❌ BM25Retriever import failed: {e}")
        return False
    
    try:
        from retrieval.semantic_retriever import SemanticRetriever
        print("✅ SemanticRetriever imported")
    except Exception as e:
        print(f"❌ SemanticRetriever import failed: {e}")
        return False
    
    try:
        from retrieval.hybrid_retriever import AdvancedHybridRetriever, FusionStrategy
        print("✅ AdvancedHybridRetriever imported")
        print(f"   FusionStrategy.RRF: {FusionStrategy.RRF}")
        print(f"   FusionStrategy.WEIGHTED: {FusionStrategy.WEIGHTED}")
    except Exception as e:
        print(f"❌ AdvancedHybridRetriever import failed: {e}")
        return False
    
    try:
        from retrieval.metadata_filter import MetadataFilter, FilterOperator
        print("✅ MetadataFilter imported")
    except Exception as e:
        print(f"❌ MetadataFilter import failed: {e}")
        return False
    
    try:
        from retrieval.rerankers.cross_encoder_reranker import CrossEncoderReranker
        print("✅ CrossEncoderReranker imported")
    except Exception as e:
        print(f"❌ CrossEncoderReranker import failed: {e}")
        return False
    
    try:
        from retrieval.explainability import ExplainabilityAnalyzer, RetrievalExplanation
        print("✅ ExplainabilityAnalyzer imported")
    except Exception as e:
        print(f"❌ ExplainabilityAnalyzer import failed: {e}")
        return False
    
    try:
        from retrieval.red_team_tester import RedTeamTester, RedTeamTestResult
        print("✅ RedTeamTester imported")
    except Exception as e:
        print(f"❌ RedTeamTester import failed: {e}")
        return False
    
    try:
        from app.services.rag_service import RAGService
        print("✅ RAGService imported")
    except Exception as e:
        print(f"❌ RAGService import failed: {e}")
        return False
    
    try:
        from app.models.schemas import HybridSearchRequest, HybridSearchResult, FusionStrategy
        print("✅ API schemas imported")
    except Exception as e:
        print(f"❌ API schemas import failed: {e}")
        return False
    
    return True


def test_bm25_retriever():
    """Test BM25 retriever functionality"""
    print("\n" + "=" * 60)
    print("2️⃣ Testing BM25 Retriever")
    print("=" * 60)
    
    try:
        from retrieval.bm25_retriever import BM25Retriever
        
        documents = [
            "A contract of sale includes implied conditions about quality.",
            "Employment law covers employee rights and obligations.",
            "Contract law is fundamental to legal transactions.",
            "The Sale of Goods Act 1979 sets out legal requirements."
        ]
        
        retriever = BM25Retriever(documents)
        results = retriever.search("contract law", top_k=2)
        
        print(f"✅ BM25 retriever initialized with {len(documents)} documents")
        print(f"✅ Search returned {len(results)} results")
        
        if results:
            for i, (idx, score) in enumerate(results, 1):
                print(f"   {i}. Doc {idx}: Score {score:.3f}")
        
        return True
    except Exception as e:
        print(f"❌ BM25 retriever test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_metadata_filter():
    """Test metadata filtering"""
    print("\n" + "=" * 60)
    print("3️⃣ Testing Metadata Filtering")
    print("=" * 60)
    
    try:
        from retrieval.metadata_filter import MetadataFilter
        
        # Test empty filter
        empty_filter = MetadataFilter()
        assert empty_filter.is_empty(), "Empty filter should be empty"
        print("✅ Empty filter check passed")
        
        # Test equals filter
        filter_obj = MetadataFilter()
        filter_obj.add_equals_filter("jurisdiction", "UK")
        assert not filter_obj.is_empty(), "Filter with condition should not be empty"
        print("✅ Equals filter added")
        
        # Test IN filter
        filter_obj.add_in_filter("document_type", ["statute", "contract"])
        print(f"✅ IN filter added (total conditions: {len(filter_obj.filters)})")
        
        # Test chunk filtering
        chunks = [
            {"metadata": {"jurisdiction": "UK", "document_type": "statute"}},
            {"metadata": {"jurisdiction": "US", "document_type": "statute"}},
            {"metadata": {"jurisdiction": "UK", "document_type": "contract"}},
        ]
        
        filtered = filter_obj.filter_chunks(chunks)
        print(f"✅ Filtered {len(chunks)} chunks to {len(filtered)} chunks")
        
        return True
    except Exception as e:
        print(f"❌ Metadata filter test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_explainability():
    """Test explainability features"""
    print("\n" + "=" * 60)
    print("4️⃣ Testing Explainability")
    print("=" * 60)
    
    try:
        from retrieval.explainability import ExplainabilityAnalyzer
        
        analyzer = ExplainabilityAnalyzer()
        query = "What are employee rights in the UK?"
        text = "Employees in the UK have various rights under employment law."
        
        # Test term extraction
        terms = analyzer.extract_query_terms(query)
        print(f"✅ Query terms extracted: {terms}")
        
        # Test highlighting
        highlighted, spans = analyzer.highlight_matched_terms(text, query)
        print(f"✅ Text highlighted: {highlighted[:60]}...")
        print(f"✅ Matched spans: {spans}")
        
        # Test explanation
        result = {
            "chunk_id": "chunk_1",
            "text": text,
            "similarity_score": 0.85,
            "bm25_score": 0.75,
            "semantic_score": 0.90,
            "bm25_rank": 2,
            "semantic_rank": 1,
            "rank": 1
        }
        
        explanation = analyzer.explain_retrieval(result, query)
        print(f"✅ Explanation generated: {explanation.explanation[:80]}...")
        print(f"✅ Confidence: {explanation.confidence:.3f}")
        
        return True
    except Exception as e:
        print(f"❌ Explainability test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_red_team_tester():
    """Test red team tester"""
    print("\n" + "=" * 60)
    print("5️⃣ Testing Red Team Tester")
    print("=" * 60)
    
    try:
        from retrieval.red_team_tester import RedTeamTester
        
        tester = RedTeamTester()
        test_cases = tester.load_test_cases()
        
        print(f"✅ RedTeamTester initialized")
        print(f"✅ Loaded {len(test_cases)} test cases")
        
        # Show categories
        categories = {}
        for tc in test_cases:
            cat = tc.get('category', 'unknown')
            categories[cat] = categories.get(cat, 0) + 1
        
        print(f"✅ Test categories: {', '.join(categories.keys())}")
        
        # Test single test case execution (without guardrails)
        if test_cases:
            test_case = test_cases[0]
            result = tester.run_test_case(test_case, use_rag=False)
            print(f"✅ Test case execution: {result.test_id} - {result.actual_behavior}")
        
        return True
    except Exception as e:
        print(f"❌ Red team tester test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_api_schemas():
    """Test API schemas"""
    print("\n" + "=" * 60)
    print("6️⃣ Testing API Schemas")
    print("=" * 60)
    
    try:
        from app.models.schemas import (
            HybridSearchRequest, HybridSearchResult, FusionStrategy,
            MetadataFilterRequest
        )
        
        # Test request schema
        request = HybridSearchRequest(
            query="What is contract law?",
            top_k=5,
            fusion_strategy=FusionStrategy.RRF,
            include_explanation=True,
            highlight_sources=True,
            metadata_filters=[
                MetadataFilterRequest(field="jurisdiction", value="UK", operator="eq")
            ]
        )
        
        print(f"✅ HybridSearchRequest created")
        print(f"   Query: {request.query}")
        print(f"   Fusion strategy: {request.fusion_strategy.value}")
        print(f"   Include explanation: {request.include_explanation}")
        print(f"   Highlight sources: {request.highlight_sources}")
        
        # Test result schema
        result = HybridSearchResult(
            chunk_id="chunk_1",
            text="Test text",
            similarity_score=0.85,
            bm25_score=0.75,
            semantic_score=0.90,
            rank=1,
            section="Test Section",
            explanation="Test explanation",
            confidence=0.85,
            matched_terms=["contract", "law"],
            highlighted_text="Test **text**"
        )
        
        print(f"✅ HybridSearchResult created with explainability fields")
        
        return True
    except Exception as e:
        print(f"❌ API schemas test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_configuration():
    """Test configuration"""
    print("\n" + "=" * 60)
    print("7️⃣ Testing Configuration")
    print("=" * 60)
    
    try:
        from app.core.config import settings
        
        print(f"✅ Configuration loaded")
        print(f"   EMBEDDING_MODEL: {settings.EMBEDDING_MODEL}")
        print(f"   ENABLE_RERANKING: {settings.ENABLE_RERANKING}")
        print(f"   RERANKER_MODEL: {settings.RERANKER_MODEL}")
        print(f"   HYBRID_SEARCH_FUSION_STRATEGY: {settings.HYBRID_SEARCH_FUSION_STRATEGY}")
        print(f"   HYBRID_SEARCH_BM25_WEIGHT: {settings.HYBRID_SEARCH_BM25_WEIGHT}")
        print(f"   HYBRID_SEARCH_SEMANTIC_WEIGHT: {settings.HYBRID_SEARCH_SEMANTIC_WEIGHT}")
        
        return True
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("🧪 Phase 2 Integration Test Suite")
    print("=" * 60)
    print("Testing all Phase 2 features (without model loading)")
    print("=" * 60)
    
    results = []
    
    # Run all tests
    results.append(("Imports", test_imports()))
    results.append(("BM25 Retriever", test_bm25_retriever()))
    results.append(("Metadata Filter", test_metadata_filter()))
    results.append(("Explainability", test_explainability()))
    results.append(("Red Team Tester", test_red_team_tester()))
    results.append(("API Schemas", test_api_schemas()))
    results.append(("Configuration", test_configuration()))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Test Summary")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {name:<25}: {status}")
    
    print(f"\n   Total: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("\n✅ All Phase 2 integration tests passed!")
        sys.exit(0)
    else:
        print(f"\n⚠️ {total - passed} test(s) failed")
        sys.exit(1)

