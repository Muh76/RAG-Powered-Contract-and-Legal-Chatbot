# Legal Chatbot - LLM Generation Service
# Extracted from Phase 1 notebook

import os
import sys
import re
from pathlib import Path
from typing import List, Dict, Any
import logging
from openai import OpenAI

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from app.core.config import settings

logger = logging.getLogger(__name__)


class LLMService:
    """LLM service for generating legal answers - extracted from Phase 1 notebook"""
    
    def __init__(self):
        api_key = settings.OPENAI_API_KEY or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set in settings or environment")
        
        self.client = OpenAI(api_key=api_key)
        self.model_name = settings.OPENAI_MODEL
        self.max_tokens = 500
        self.temperature = 0.1  # Low temperature for consistent legal responses
    
    def generate_legal_answer(
        self,
        query: str,
        retrieved_chunks: List[Dict],
        mode: str = "solicitor"
    ) -> Dict[str, Any]:
        """Generate a legal answer with citations based on retrieved chunks"""
        # Prepare context from retrieved chunks
        context_parts = []
        citations = []
        
        for i, chunk in enumerate(retrieved_chunks):
            context_parts.append(f"[{i+1}] {chunk.get('text', '')}")
            citations.append({
                "id": i + 1,
                "chunk_id": chunk.get('chunk_id', f'chunk_{i+1}'),
                "section": chunk.get('section', 'Unknown Section'),
                "title": chunk.get('metadata', {}).get('title', 'Unknown Title'),
                "text_snippet": chunk.get('text', '')[:200] + "..." if len(chunk.get('text', '')) > 200 else chunk.get('text', '')
            })
        
        context = "\n\n".join(context_parts)
        
        # Choose prompt template based on mode
        if mode == "solicitor":
            system_prompt = """You are a legal assistant specializing in UK law. You must:
1. Answer ONLY using the provided legal sources
2. Use precise legal terminology and cite specific sections
3. Include citations in format [1], [2], etc. for each claim
4. If sources are insufficient, clearly state this
5. Maintain professional legal language"""
        else:  # public mode
            system_prompt = """You are a legal assistant helping the general public understand UK law. You must:
1. Answer using the provided legal sources in plain language
2. Explain legal concepts clearly without jargon
3. Include citations in format [1], [2], etc. for each claim
4. If sources are insufficient, clearly state this
5. Use accessible, everyday language"""
        
        user_prompt = f"""SOURCES:
{context}

QUESTION: {query}

Instructions:
- Answer the question using ONLY the provided sources
- Include citations [1], [2], etc. for each factual claim
- If the sources don't contain enough information, say "The provided sources do not contain sufficient information to answer this question completely"
- Keep your answer concise but comprehensive"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )
            
            answer = response.choices[0].message.content
            
            # Validate citations in the answer
            citation_validation = self._validate_citations(answer, len(citations))
            
            return {
                "answer": answer,
                "citations": citations,
                "citation_validation": citation_validation,
                "model_used": self.model_name,
                "mode": mode,
                "query": query
            }
            
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return {
                "answer": f"Error generating response: {str(e)}",
                "citations": [],
                "citation_validation": {"has_citations": False, "error": str(e)},
                "model_used": self.model_name,
                "mode": mode,
                "query": query
            }
    
    def _validate_citations(self, answer: str, num_citations: int) -> Dict[str, Any]:
        """Validate that the answer contains proper citations"""
        # Find all citation patterns [1], [2], etc.
        citation_pattern = r'\[(\d+)\]'
        found_citations = re.findall(citation_pattern, answer)
        
        if not found_citations:
            return {
                "has_citations": False,
                "found_citations": [],
                "valid_citations": False,
                "message": "No citations found in answer"
            }
        
        # Check if citations are within valid range
        valid_citations = []
        for citation in found_citations:
            if 1 <= int(citation) <= num_citations:
                valid_citations.append(int(citation))
        
        return {
            "has_citations": True,
            "found_citations": [int(c) for c in found_citations],
            "valid_citations": len(valid_citations) > 0,
            "valid_citation_numbers": valid_citations,
            "message": f"Found {len(found_citations)} citations, {len(valid_citations)} valid"
        }