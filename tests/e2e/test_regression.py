"""
Regression Tests - Phase 1, 2, 3 Features Verification
Phase 4.1: Ensure backward compatibility and existing features work
"""

import pytest
import requests
from typing import Dict, Any


class TestPhase1Regression:
    """Test Phase 1 (MVP) features still work"""
    
    BASE_URL = "http://localhost:8000"
    TIMEOUT = 30
    
    def test_traditional_rag_endpoint(self):
        """Verify traditional RAG endpoint (/api/v1/chat) still works"""
        payload = {
            "query": "What is contract law?",
            "mode": "public",
            "top_k": 5
        }
        response = requests.post(
            f"{self.BASE_URL}/api/v1/chat",
            json=payload,
            timeout=self.TIMEOUT
        )
        
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        data = response.json()
        
        # Phase 1 features
        assert "answer" in data, "Answer field missing"
        assert "sources" in data, "Sources field missing"
        assert "safety" in data, "Safety field missing"
        assert "metrics" in data, "Metrics field missing"
        assert data["legal_jurisdiction"] == "UK", "Jurisdiction should be UK"
        assert len(data["answer"]) > 0, "Answer should not be empty"
        
        print("✅ Phase 1 traditional RAG endpoint test passed")
    
    def test_dual_mode_responses(self):
        """Verify dual-mode responses (Solicitor/Public) still work"""
        test_query = "What are employee rights?"
        
        # Test public mode
        public_payload = {"query": test_query, "mode": "public"}
        public_response = requests.post(
            f"{self.BASE_URL}/api/v1/chat",
            json=public_payload,
            timeout=self.TIMEOUT
        )
        assert public_response.status_code == 200
        public_data = public_response.json()
        assert public_data["mode"] == "public"
        
        # Test solicitor mode
        solicitor_payload = {"query": test_query, "mode": "solicitor"}
        solicitor_response = requests.post(
            f"{self.BASE_URL}/api/v1/chat",
            json=solicitor_payload,
            timeout=self.TIMEOUT
        )
        assert solicitor_response.status_code == 200
        solicitor_data = solicitor_response.json()
        assert solicitor_data["mode"] == "solicitor"
        
        # Answers should be different (or at least valid)
        assert len(public_data["answer"]) > 0
        assert len(solicitor_data["answer"]) > 0
        
        print("✅ Phase 1 dual-mode responses test passed")
    
    def test_guardrails_domain_gating(self):
        """Verify guardrails domain gating still works"""
        # Legal query should pass
        legal_payload = {"query": "What is contract law?", "mode": "public"}
        legal_response = requests.post(
            f"{self.BASE_URL}/api/v1/chat",
            json=legal_payload,
            timeout=self.TIMEOUT
        )
        assert legal_response.status_code == 200
        
        # Non-legal query should be blocked
        non_legal_payload = {"query": "How do I cook pasta?", "mode": "public"}
        non_legal_response = requests.post(
            f"{self.BASE_URL}/api/v1/chat",
            json=non_legal_payload,
            timeout=self.TIMEOUT
        )
        assert non_legal_response.status_code == 200  # Guardrails return 200 with message
        non_legal_data = non_legal_response.json()
        # Should have safety flags indicating non-legal query
        assert not non_legal_data["safety"]["is_safe"] or len(non_legal_data["safety"]["flags"]) > 0
        
        print("✅ Phase 1 guardrails domain gating test passed")
    
    def test_citations_required(self):
        """Verify citations are included in responses"""
        payload = {
            "query": "What are the implied conditions in a contract of sale?",
            "mode": "solicitor",
            "top_k": 5
        }
        response = requests.post(
            f"{self.BASE_URL}/api/v1/chat",
            json=payload,
            timeout=self.TIMEOUT
        )
        assert response.status_code == 200
        data = response.json()
        
        # Should have sources/citations
        assert "sources" in data
        assert isinstance(data["sources"], list)
        
        print("✅ Phase 1 citations requirement test passed")
    
    def test_health_endpoint(self):
        """Verify health endpoint still works"""
        response = requests.get(f"{self.BASE_URL}/api/v1/health", timeout=10)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        print("✅ Phase 1 health endpoint test passed")


class TestPhase2Regression:
    """Test Phase 2 (Advanced RAG) features still work"""
    
    BASE_URL = "http://localhost:8000"
    TIMEOUT = 30
    
    def test_hybrid_search_endpoint(self):
        """Verify hybrid search endpoint (/api/v1/search/hybrid) still works"""
        payload = {
            "query": "contract sale goods",
            "top_k": 5,
            "fusion_strategy": "rrf"
        }
        response = requests.post(
            f"{self.BASE_URL}/api/v1/search/hybrid",
            json=payload,
            timeout=self.TIMEOUT
        )
        
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        data = response.json()
        
        # Phase 2 features
        assert "results" in data, "Results field missing"
        assert "total_results" in data, "Total results field missing"
        assert "fusion_strategy" in data, "Fusion strategy field missing"
        assert data["fusion_strategy"] == "rrf", "Fusion strategy should be RRF"
        assert isinstance(data["results"], list), "Results should be a list"
        
        # Check if results have hybrid search fields
        if len(data["results"]) > 0:
            result = data["results"][0]
            # Should have BM25 or semantic scores
            assert "bm25_score" in result or "semantic_score" in result or "similarity_score" in result
        
        print("✅ Phase 2 hybrid search endpoint test passed")
    
    def test_hybrid_search_get_endpoint(self):
        """Verify hybrid search GET endpoint still works"""
        params = {
            "query": "employment rights",
            "top_k": 3,
            "fusion_strategy": "weighted"
        }
        response = requests.get(
            f"{self.BASE_URL}/api/v1/search/hybrid",
            params=params,
            timeout=self.TIMEOUT
        )
        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        assert data["fusion_strategy"] == "weighted"
        print("✅ Phase 2 hybrid search GET endpoint test passed")
    
    def test_metadata_filtering(self):
        """Verify metadata filtering still works"""
        payload = {
            "query": "contract law",
            "top_k": 5,
            "metadata_filters": [
                {
                    "field": "jurisdiction",
                    "value": "UK",
                    "operator": "eq"
                }
            ]
        }
        response = requests.post(
            f"{self.BASE_URL}/api/v1/search/hybrid",
            json=payload,
            timeout=self.TIMEOUT
        )
        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        
        # Verify filtered results have UK jurisdiction
        for result in data["results"]:
            if "jurisdiction" in result.get("metadata", {}):
                assert result["metadata"]["jurisdiction"] == "UK"
        
        print("✅ Phase 2 metadata filtering test passed")
    
    def test_explainability_features(self):
        """Verify explainability features still work"""
        payload = {
            "query": "employee rights",
            "top_k": 3,
            "include_explanation": True,
            "highlight_sources": True
        }
        response = requests.post(
            f"{self.BASE_URL}/api/v1/search/hybrid",
            json=payload,
            timeout=self.TIMEOUT
        )
        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        
        # Check if explainability fields are present
        if len(data["results"]) > 0:
            result = data["results"][0]
            # Should have explainability fields when enabled
            if result.get("explanation") or result.get("highlighted_text"):
                assert "confidence" in result or result.get("confidence") is not None
        
        print("✅ Phase 2 explainability features test passed")
    
    def test_fusion_strategies(self):
        """Verify both fusion strategies (RRF and weighted) still work"""
        query = "contract sale goods"
        top_k = 3
        
        # Test RRF strategy
        rrf_payload = {
            "query": query,
            "top_k": top_k,
            "fusion_strategy": "rrf"
        }
        rrf_response = requests.post(
            f"{self.BASE_URL}/api/v1/search/hybrid",
            json=rrf_payload,
            timeout=self.TIMEOUT
        )
        assert rrf_response.status_code == 200
        rrf_data = rrf_response.json()
        assert rrf_data["fusion_strategy"] == "rrf"
        
        # Test weighted strategy
        weighted_payload = {
            "query": query,
            "top_k": top_k,
            "fusion_strategy": "weighted"
        }
        weighted_response = requests.post(
            f"{self.BASE_URL}/api/v1/search/hybrid",
            json=weighted_payload,
            timeout=self.TIMEOUT
        )
        assert weighted_response.status_code == 200
        weighted_data = weighted_response.json()
        assert weighted_data["fusion_strategy"] == "weighted"
        
        print("✅ Phase 2 fusion strategies test passed")


class TestPhase3Regression:
    """Test Phase 3 (Agentic RAG) features still work"""
    
    BASE_URL = "http://localhost:8000"
    TIMEOUT = 60
    
    def test_agentic_chat_endpoint(self):
        """Verify agentic chat endpoint (/api/v1/agentic-chat) still works"""
        payload = {
            "query": "What is the Sale of Goods Act 1979?",
            "mode": "public"
        }
        response = requests.post(
            f"{self.BASE_URL}/api/v1/agentic-chat",
            json=payload,
            timeout=self.TIMEOUT
        )
        
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        data = response.json()
        
        # Phase 3 features
        assert "answer" in data, "Answer field missing"
        assert "tool_calls" in data, "Tool calls field missing"
        assert "iterations" in data, "Iterations field missing"
        assert "intermediate_steps_count" in data, "Intermediate steps count field missing"
        assert "mode" in data, "Mode field missing"
        assert "safety" in data, "Safety field missing"
        assert "metrics" in data, "Metrics field missing"
        assert isinstance(data["tool_calls"], list), "Tool calls should be a list"
        assert isinstance(data["iterations"], int), "Iterations should be an integer"
        assert len(data["answer"]) > 0, "Answer should not be empty"
        
        print("✅ Phase 3 agentic chat endpoint test passed")
    
    def test_tool_calling(self):
        """Verify tool calling still works"""
        payload = {
            "query": "Compare the Sale of Goods Act 1979 with the Consumer Rights Act 2015",
            "mode": "solicitor"
        }
        response = requests.post(
            f"{self.BASE_URL}/api/v1/agentic-chat",
            json=payload,
            timeout=self.TIMEOUT
        )
        assert response.status_code == 200
        data = response.json()
        
        # Complex query should use tools
        assert data["iterations"] > 0, "Agent should perform iterations"
        assert len(data["tool_calls"]) >= 1, "Agent should call at least one tool"
        
        # Verify tool call structure
        for tool_call in data["tool_calls"]:
            assert "tool" in tool_call, "Tool name missing"
            assert "input" in tool_call, "Tool input missing"
            assert "result" in tool_call, "Tool result missing"
        
        print("✅ Phase 3 tool calling test passed")
    
    def test_agentic_chat_with_history(self):
        """Verify agentic chat with conversation history still works"""
        payload = {
            "query": "What about the Consumer Rights Act 2015?",
            "mode": "public",
            "chat_history": [
                {"role": "user", "content": "Tell me about the Sale of Goods Act 1979"},
                {"role": "assistant", "content": "The Sale of Goods Act 1979 is..."}
            ]
        }
        response = requests.post(
            f"{self.BASE_URL}/api/v1/agentic-chat",
            json=payload,
            timeout=self.TIMEOUT
        )
        assert response.status_code == 200
        data = response.json()
        assert len(data["answer"]) > 0
        print("✅ Phase 3 agentic chat with history test passed")
    
    def test_agent_stats_endpoint(self):
        """Verify agent stats endpoint still works"""
        response = requests.get(
            f"{self.BASE_URL}/api/v1/agentic-chat/stats",
            timeout=10
        )
        assert response.status_code == 200
        data = response.json()
        # Should return stats (format may vary)
        assert isinstance(data, dict)
        print("✅ Phase 3 agent stats endpoint test passed")
    
    def test_multi_step_reasoning(self):
        """Verify multi-step reasoning still works"""
        payload = {
            "query": "What are the key differences between the Sale of Goods Act 1979 and the Consumer Rights Act 2015?",
            "mode": "solicitor"
        }
        response = requests.post(
            f"{self.BASE_URL}/api/v1/agentic-chat",
            json=payload,
            timeout=self.TIMEOUT
        )
        assert response.status_code == 200
        data = response.json()
        
        # Complex query should require multiple steps
        assert data["iterations"] > 0
        assert data["intermediate_steps_count"] > 0
        
        print("✅ Phase 3 multi-step reasoning test passed")


class TestBackwardCompatibility:
    """Test backward compatibility between phases"""
    
    BASE_URL = "http://localhost:8000"
    TIMEOUT = 30
    
    def test_traditional_chat_still_works_after_phase2(self):
        """Verify traditional chat endpoint still works after Phase 2"""
        payload = {
            "query": "What is contract law?",
            "mode": "public",
            "top_k": 5
        }
        response = requests.post(
            f"{self.BASE_URL}/api/v1/chat",
            json=payload,
            timeout=self.TIMEOUT
        )
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "sources" in data
        print("✅ Traditional chat backward compatibility test passed")
    
    def test_traditional_chat_still_works_after_phase3(self):
        """Verify traditional chat endpoint still works after Phase 3"""
        payload = {
            "query": "What are employment rights?",
            "mode": "solicitor",
            "top_k": 5
        }
        response = requests.post(
            f"{self.BASE_URL}/api/v1/chat",
            json=payload,
            timeout=self.TIMEOUT
        )
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert data["mode"] == "solicitor"
        print("✅ Traditional chat backward compatibility (Phase 3) test passed")
    
    def test_hybrid_search_still_works_after_phase3(self):
        """Verify hybrid search still works after Phase 3"""
        payload = {
            "query": "contract law",
            "top_k": 5,
            "fusion_strategy": "rrf"
        }
        response = requests.post(
            f"{self.BASE_URL}/api/v1/search/hybrid",
            json=payload,
            timeout=self.TIMEOUT
        )
        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        assert data["fusion_strategy"] == "rrf"
        print("✅ Hybrid search backward compatibility test passed")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

