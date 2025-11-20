# Legal Chatbot - Guardrails Service
# Extracted from Phase 1 notebook

import re
from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)


class GuardrailsService:
    """Guardrails for legal chatbot - extracted from Phase 1 notebook"""
    
    def __init__(self):
        # Legal domain keywords
        self.legal_keywords = [
            'contract', 'sale', 'goods', 'act', 'law', 'legal', 'rights', 'obligations',
            'employment', 'data protection', 'privacy', 'statute', 'section', 'clause',
            'liability', 'breach', 'terms', 'conditions', 'warranty', 'implied',
            'seller', 'buyer', 'employer', 'employee', 'personal data', 'processing'
        ]
        
        # Non-legal keywords that should be refused
        self.non_legal_keywords = [
            'medical', 'health', 'doctor', 'medicine', 'treatment', 'surgery',
            'cooking', 'recipe', 'food', 'restaurant', 'travel', 'vacation',
            'sports', 'game', 'entertainment', 'movie', 'music', 'art'
        ]
        
        # Harmful content patterns
        self.harmful_patterns = [
            r'\b(suicide|self-harm|kill.*self)\b',
            r'\b(bomb|explosive|terrorist)\b',
            r'\b(hate.*speech|racist|discriminat)\b'
        ]
    
    def validate_query(self, query: str) -> Dict[str, Any]:
        """Validate if query is appropriate for legal domain"""
        query_lower = query.lower()
        
        # Check 1: Domain gating (legal vs non-legal)
        legal_score = sum(1 for keyword in self.legal_keywords if keyword in query_lower)
        non_legal_score = sum(1 for keyword in self.non_legal_keywords if keyword in query_lower)
        
        if non_legal_score > legal_score and non_legal_score > 0:
            return {
                "valid": False,
                "reason": "domain_gating",
                "message": "I specialize in legal questions. Please ask about UK law, contracts, employment rights, or other legal matters.",
                "suggestion": "Try rephrasing your question to focus on legal aspects."
            }
        
        # Check 2: Harmful content detection
        for pattern in self.harmful_patterns:
            if re.search(pattern, query_lower):
                return {
                    "valid": False,
                    "reason": "harmful_content",
                    "message": "I cannot provide assistance with harmful or dangerous content.",
                    "suggestion": "Please ask about legal matters instead."
                }
        
        # Check 3: Minimum legal relevance
        if legal_score == 0 and len(query.split()) > 3:
            return {
                "valid": False,
                "reason": "insufficient_legal_relevance",
                "message": "This doesn't appear to be a legal question. I can help with UK law, contracts, employment rights, data protection, and other legal matters.",
                "suggestion": "Could you rephrase this as a legal question?"
            }
        
        return {
            "valid": True,
            "reason": "passed_validation",
            "message": "Query validated successfully",
            "legal_relevance_score": legal_score
        }
    
    def validate_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Validate that response meets quality standards"""
        answer = response.get('answer', '')
        citations = response.get('citations', [])
        validation = response.get('citation_validation', {})
        
        # Check 1: Citation enforcement
        if not validation.get('has_citations', False):
            return {
                "valid": False,
                "reason": "missing_citations",
                "message": "Response must include citations to legal sources",
                "action": "regenerate_with_citations"
            }
        
        # Check 2: Grounding check
        retrieval_info = response.get('retrieval_info', {})
        if retrieval_info.get('num_chunks_retrieved', 0) < 2:
            return {
                "valid": False,
                "reason": "insufficient_grounding",
                "message": "Insufficient legal sources found for this question",
                "action": "suggest_alternatives"
            }
        
        # Check 3: Answer quality
        if len(answer.strip()) < 50:
            return {
                "valid": False,
                "reason": "insufficient_answer",
                "message": "Answer is too brief for a legal question",
                "action": "expand_answer"
            }
        
        return {
            "valid": True,
            "reason": "passed_validation",
            "message": "Response meets quality standards"
        }
    
    def validate_answer(self, answer: str, query: str) -> Dict[str, Any]:
        """Validate answer for safety and quality (backward compatibility)"""
        # Check for harmful content
        answer_lower = answer.lower()
        for pattern in self.harmful_patterns:
            if re.search(pattern, answer_lower):
                return {
                    "valid": False,
                    "reason": "harmful_content",
                    "message": "The response contains inappropriate content.",
                    "suggestion": "Please try a different question."
                }
        
        return {
            "valid": True,
            "reason": "passed_validation",
            "message": "Answer validated successfully"
        }
    
    def check_citations(self, answer: str, num_sources: int) -> Dict[str, Any]:
        """Check if answer contains proper citations"""
        # Find all citation patterns [1], [2], etc.
        citation_pattern = r'\[(\d+)\]'
        found_citations = re.findall(citation_pattern, answer)
        
        if not found_citations:
            return {
                "valid": False,
                "has_citations": False,
                "count": 0,
                "reason": "missing_citations",
                "message": "Response must include citations [1], [2], etc. for each factual claim"
            }
        
        # Check if citations are within valid range
        valid_citations = []
        invalid_citations = []
        for citation in found_citations:
            citation_num = int(citation)
            if 1 <= citation_num <= num_sources:
                valid_citations.append(citation_num)
            else:
                invalid_citations.append(citation_num)
        
        if len(valid_citations) == 0:
            return {
                "valid": False,
                "has_citations": True,
                "count": len(found_citations),
                "valid_count": 0,
                "invalid_citations": invalid_citations,
                "reason": "invalid_citations",
                "message": f"Citations found but none are valid (must be 1-{num_sources})"
            }
        
        # Count sentences and check citation coverage
        sentences = re.split(r'[.!?]+', answer)
        sentences_with_citations = sum(1 for s in sentences if re.search(citation_pattern, s))
        citation_coverage = sentences_with_citations / len(sentences) if sentences else 0
        
        return {
            "valid": True,
            "has_citations": True,
            "count": len(found_citations),
            "valid_count": len(valid_citations),
            "invalid_count": len(invalid_citations),
            "citation_coverage": citation_coverage,
            "valid_citations": sorted(set(valid_citations)),
            "invalid_citations": sorted(set(invalid_citations)),
            "sentences_with_citations": sentences_with_citations,
            "total_sentences": len(sentences)
        }
    
    def check_source_coverage(self, answer: str, retrieved_chunks: List[Dict]) -> Dict[str, Any]:
        """Check if answer content matches retrieved sources"""
        if not retrieved_chunks:
            return {
                "valid": False,
                "reason": "no_sources",
                "message": "No sources were retrieved"
            }
        
        # Extract key phrases from answer (simplified - could use more sophisticated matching)
        answer_lower = answer.lower()
        
        # Check if answer mentions any source-specific information
        source_mentions = []
        for i, chunk in enumerate(retrieved_chunks):
            chunk_text = chunk.get('text', '').lower()[:200]  # First 200 chars
            # Simple keyword overlap check
            chunk_words = set(chunk_text.split()[:10])  # First 10 words
            answer_words = set(answer_lower.split())
            overlap = len(chunk_words.intersection(answer_words))
            if overlap > 2:  # At least 3 words overlap
                source_mentions.append(i + 1)
        
        coverage_ratio = len(source_mentions) / len(retrieved_chunks) if retrieved_chunks else 0
        
        return {
            "valid": coverage_ratio >= 0.5,  # At least 50% of sources should be mentioned
            "coverage_ratio": coverage_ratio,
            "sources_mentioned": source_mentions,
            "total_sources": len(retrieved_chunks),
            "reason": "low_coverage" if coverage_ratio < 0.5 else "good_coverage",
            "message": f"Answer covers {len(source_mentions)}/{len(retrieved_chunks)} sources"
        }
    
    def apply_all_rules(self, query: str, answer: str, retrieved_chunks: List[Dict], 
                       citation_validation: Dict[str, Any] = None) -> Dict[str, Any]:
        """Apply all guardrail rules comprehensively"""
        results = {
            "all_passed": True,
            "rules_applied": [],
            "failures": [],
            "warnings": []
        }
        
        # Rule 1: Domain gating (query validation)
        query_validation = self.validate_query(query)
        results["rules_applied"].append("domain_gating")
        if not query_validation["valid"]:
            results["all_passed"] = False
            results["failures"].append({
                "rule": "domain_gating",
                "reason": query_validation["reason"],
                "message": query_validation["message"]
            })
        
        # Rule 2: Citation enforcement
        num_sources = len(retrieved_chunks)
        if citation_validation:
            citation_check = citation_validation
        else:
            citation_check = self.check_citations(answer, num_sources)
        results["rules_applied"].append("citation_enforcement")
        if not citation_check.get("has_citations", False):
            results["all_passed"] = False
            results["failures"].append({
                "rule": "citation_enforcement",
                "reason": "missing_citations",
                "message": "Response must include citations [1], [2], etc. for each factual claim"
            })
        elif not citation_check.get("valid", True):
            results["all_passed"] = False
            results["failures"].append({
                "rule": "citation_enforcement",
                "reason": citation_check.get("reason", "invalid_citations"),
                "message": citation_check.get("message", "Citations are invalid")
            })
        elif citation_check.get("citation_coverage", 0) < 0.7:  # Less than 70% of sentences have citations
            results["warnings"].append({
                "rule": "citation_coverage",
                "message": f"Only {citation_check.get('citation_coverage', 0)*100:.1f}% of sentences have citations"
            })
        
        # Rule 3: Hallucination check (source coverage)
        source_coverage = self.check_source_coverage(answer, retrieved_chunks)
        results["rules_applied"].append("source_coverage")
        if not source_coverage["valid"]:
            results["all_passed"] = False
            results["failures"].append({
                "rule": "source_coverage",
                "reason": source_coverage["reason"],
                "message": source_coverage["message"]
            })
        
        # Rule 4: Injection detection
        injection_patterns = [
            r'ignore\s+previous\s+instructions',
            r'forget\s+everything',
            r'system\s+prompt',
            r'new\s+instructions'
        ]
        answer_lower = answer.lower()
        query_lower = query.lower()
        for pattern in injection_patterns:
            if re.search(pattern, answer_lower) or re.search(pattern, query_lower):
                results["all_passed"] = False
                results["failures"].append({
                    "rule": "injection_detection",
                    "reason": "suspected_injection",
                    "message": "Potential prompt injection detected"
                })
                break
        results["rules_applied"].append("injection_detection")
        
        # Rule 5: Harmful content check
        harmful_check = self.validate_answer(answer, query)
        results["rules_applied"].append("harmful_content")
        if not harmful_check["valid"]:
            results["all_passed"] = False
            results["failures"].append({
                "rule": "harmful_content",
                "reason": harmful_check["reason"],
                "message": harmful_check["message"]
            })
        
        # Rule 6: Grounding validation (minimum sources)
        if len(retrieved_chunks) < 1:
            results["all_passed"] = False
            results["failures"].append({
                "rule": "grounding_validation",
                "reason": "insufficient_sources",
                "message": "Insufficient sources retrieved for answering"
            })
        results["rules_applied"].append("grounding_validation")
        
        return results
    
    def generate_refusal_response(self, validation_result: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a polite refusal response"""
        return {
            "answer": validation_result["message"],
            "citations": [],
            "citation_validation": {"has_citations": False, "message": "Refusal response"},
            "model_used": "guardrails",
            "mode": "guardrails",
            "query": "N/A",
            "retrieval_info": {"num_chunks_retrieved": 0},
            "guardrails": {
                "applied": True,
                "reason": validation_result["reason"],
                "suggestion": validation_result.get("suggestion", "")
            }
        }