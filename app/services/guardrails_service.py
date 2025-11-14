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