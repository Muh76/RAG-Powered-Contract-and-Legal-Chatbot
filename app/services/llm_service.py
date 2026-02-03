# Legal Chatbot - LLM Generation Service
# Extracted from Phase 1 notebook
#
# CODE PATH FOR /api/v1/chat -> OpenAI Chat Completions API:
# 1. app/api/routes/chat.py: chat() endpoint receives request
# 2. Retrieval: rag.search() returns retrieval_result (list of chunk dicts)
# 3. chat.py line 564: llm.generate_legal_answer(query, retrieved_chunks=retrieval_result, mode)
# 4. THIS FILE: LLMService.generate_legal_answer() constructs messages and calls OpenAI
# 5. OpenAI call: self.client.chat.completions.create(messages=[...]) at line ~175

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
        self.max_tokens = 2000  # Increased from 500 to allow longer legal responses
        self.temperature = 0.1  # Low temperature for consistent legal responses
    
    def generate_legal_answer(
        self,
        query: str,
        retrieved_chunks: List[Dict],
        mode: str = "solicitor"
    ) -> Dict[str, Any]:
        """Generate a legal answer with citations based on retrieved chunks.
        This is the ONLY function that calls openai.chat.completions.create for /api/v1/chat."""
        # --- INJECTION POINT: retrieved chunks -> context ---
        # Each chunk becomes "[N] Title - Section\nchunk_text"; joined with "\n\n"
        context_parts = []
        citations = []
        
        for i, chunk in enumerate(retrieved_chunks):
            source_id = i + 1
            chunk_text = chunk.get('text', '')
            metadata = chunk.get('metadata', {})
            if isinstance(metadata, dict):
                title = metadata.get('title', chunk.get('title', 'Unknown Title'))
                section = metadata.get('section', chunk.get('section', ''))
                act_name = title if 'Act' in title else metadata.get('source', 'Unknown')
            else:
                title = chunk.get('title', 'Unknown Title')
                section = chunk.get('section', '')
                act_name = title if 'Act' in title else 'Unknown'
            
            # Format: [source_id] Title - Section (if available) - Text
            section_info = f" - {section}" if section and section != 'Unknown' else ""
            context_parts.append(f"[{source_id}] {act_name}{section_info}\n{chunk_text}")
            
            citations.append({
                "id": source_id,
                "chunk_id": chunk.get('chunk_id', f'chunk_{source_id}'),
                "section": section,
                "title": act_name,
                "text_snippet": chunk_text[:200] + "..." if len(chunk_text) > 200 else chunk_text
            })
        
        context = "\n\n".join(context_parts)  # Injected into user_prompt below
        
        # --- SYSTEM PROMPT: role instructions, citation rules, refusal language ---
        # Chosen by mode (solicitor vs public). No chunks injected here.
        # Choose prompt template based on mode with STRICT source-only enforcement
        if mode == "solicitor":
            system_prompt = """You are a legal assistant specializing in UK law.

DISCLAIMER: You provide information only, not legal advice. Include a brief note that users should consult a qualified solicitor or barrister for legal advice. Do not advise on specific outcomes or recommend courses of action.

STRICT LEGAL CITATION RULES:
1. Every factual statement MUST end with citations like [1], [2], etc. No exceptions.
2. Citation numbers MUST correspond exactly to the provided retrieved chunks (numbered [1], [2], [3], etc. in SOURCES). Never cite a number that does not exist in the sources.
3. Multiple citations are allowed per sentence (e.g., "Section 1 [1] and Section 2 [2] both require...").
4. If the answer cannot be fully supported by the sources, you MUST explicitly refuse to answer. Say: "I cannot answer this question because the available sources do not provide sufficient legal authority."
5. You must NEVER invent citations. Only use citation numbers that correspond to actual chunks in the provided SOURCES.
6. Write in clear legal English suitable for UK law. Use precise legal terminology and cite specific sections/Acts when mentioned in the sources.

HOW CITATIONS WORK:
- The retrieved documents are numbered starting from [1].
- The order is the same as the provided sources list (first source = [1], second = [2], and so on).
- When referencing a source, append its number in square brackets at the end of the sentence or claim.

Example:
Goods must be of satisfactory quality under UK law [1].
Consumers may request a repair or replacement [2][3].

REQUIRED ANSWER STRUCTURE:
- You may start with a short heading (optional).
- Use bullet points or short paragraphs.
- Each paragraph or bullet MUST end with citations [1], [2], etc. No paragraph may appear without at least one citation.
- If you cannot ensure every paragraph has citations, you MUST refuse to answer. Say: "I cannot answer this question because the available sources do not provide sufficient legal authority."

CRITICAL ANTI-HALLUCINATION RULES:
- Answer ONLY using the exact text and information from the provided sources - NEVER use prior knowledge or training data
- Your answer length must be proportional to the source material
- Use simple [N] format ONLY - NO complex formats like [3, Section X]
- Include Act name and Section number when explicitly mentioned in source metadata
- DO NOT create fictional sections, Act names, or legal provisions

MANDATORY CITATION ENFORCEMENT (NON-NEGOTIABLE):
- Every sentence MUST end with one or more citations like [1], [2].
- No sentence may appear without citations.
- Citations must reference the numbered SOURCES provided in the user prompt.
- If the model cannot cite a sentence, it MUST refuse to answer.
- Citations must be placed at the END of each sentence.

If the question is not about law or legal matters, respond briefly that you only answer legal questions and suggest rephrasing as a legal query."""
        else:  # public mode
            system_prompt = """You are a legal assistant helping the general public understand UK law.

DISCLAIMER: You provide information only, not legal advice. Include a brief note that users should consult a qualified solicitor or barrister for legal advice. Do not advise on specific outcomes or recommend courses of action.

STRICT LEGAL CITATION RULES:
1. Every factual statement MUST end with citations like [1], [2], etc. No exceptions.
2. Citation numbers MUST correspond exactly to the provided retrieved chunks (numbered [1], [2], [3], etc. in SOURCES). Never cite a number that does not exist in the sources.
3. Multiple citations are allowed per sentence (e.g., "Section 1 [1] and Section 2 [2] both require...").
4. If the answer cannot be fully supported by the sources, you MUST explicitly refuse to answer. Say: "I cannot answer this question because the available sources do not provide sufficient legal authority."
5. You must NEVER invent citations. Only use citation numbers that correspond to actual chunks in the provided SOURCES.
6. Write in clear legal English suitable for UK law. Explain legal concepts clearly without jargon, using accessible language.

HOW CITATIONS WORK:
- The retrieved documents are numbered starting from [1].
- The order is the same as the provided sources list (first source = [1], second = [2], and so on).
- When referencing a source, append its number in square brackets at the end of the sentence or claim.

Example:
Goods must be of satisfactory quality under UK law [1].
Consumers may request a repair or replacement [2][3].

REQUIRED ANSWER STRUCTURE:
- You may start with a short heading (optional).
- Use bullet points or short paragraphs.
- Each paragraph or bullet MUST end with citations [1], [2], etc. No paragraph may appear without at least one citation.
- If you cannot ensure every paragraph has citations, you MUST refuse to answer. Say: "I cannot answer this question because the available sources do not provide sufficient legal authority."

CRITICAL ANTI-HALLUCINATION RULES:
- Answer ONLY using the exact text and information from the provided sources - NEVER use prior knowledge or training data
- Your answer length must be proportional to the source material
- Use simple [N] format ONLY - NO complex formats like [3, Section X]
- Include Act name and Section number when explicitly mentioned in source metadata
- DO NOT create fictional sections, Act names, or legal provisions

MANDATORY CITATION ENFORCEMENT (NON-NEGOTIABLE):
- Every sentence MUST end with one or more citations like [1], [2].
- No sentence may appear without citations.
- Citations must reference the numbered SOURCES provided in the user prompt.
- If the model cannot cite a sentence, it MUST refuse to answer.
- Citations must be placed at the END of each sentence.

If the question is not about law or legal matters, respond briefly that you only answer legal questions and suggest rephrasing as a legal query."""
        
        # --- USER PROMPT: SOURCES (retrieved chunks) + QUESTION + instructions ---
        # {context} = injected retrieved chunks, formatted as "[1] Title\ntext", "[2] Title\ntext", ...
        # {query} = user's question from request.query
        user_prompt = f"""SOURCES (numbered [1], [2], etc.):
{context}

QUESTION: {query}

STRICT ANTI-HALLUCINATION INSTRUCTIONS:
1. Answer using ONLY the exact information from the numbered sources above - NEVER add information from your training data
2. Your answer MUST be shorter or equal to the total length of the sources - if sources are 3 paragraphs, answer should be 3 paragraphs or less
3. Cite EVERY sentence with simple [source_id] format ONLY:
   - CORRECT: "Employers must provide written statements [1]."
   - CORRECT: "The Employment Rights Act, Section 1 [1] requires..."
   - WRONG: "[1, Section 1]" or "[3, Sections 17-22]" - ONLY use [1], [2], etc.
4. DO NOT make any claims without citations - this will cause rejection
5. If a source explicitly mentions an Act name and Section number, you may include it: "Employment Rights Act 1996, Section 1 [1]"
6. DO NOT invent section numbers, Act names, or legal provisions not in the sources
7. Answer using whatever information IS in the sources - even if partial. Only refuse with "I cannot answer this question because the available sources do not provide sufficient legal authority." if sources are completely empty or irrelevant. Otherwise, provide the best answer possible from available sources
8. DO NOT expand on topics not covered in the sources - keep your answer proportional to source material

EXAMPLE GOOD RESPONSE (when sources are short):
"The Employment Rights Act 1996, Section 1 [1] requires employers to provide written statements [1]."

EXAMPLE BAD RESPONSE (DO NOT DO THIS):
"Employment rights include many provisions such as written statements, wage protection, leave rights, etc. [1]" <- This adds information not in sources!
"[1, Section 1]" <- Wrong format, use [1] only!
"Employment rights include..." <- No citation = FORBIDDEN!"""
        
        # --- FINAL messages ARRAY sent to OpenAI Chat Completions API ---
        # Exactly 2 messages: system (instructions) + user (sources + question)
        # No chat history; no assistant messages. Single-turn completion.
        try:
            logger.info(f"🔄 Calling OpenAI API: model={self.model_name}, max_tokens={self.max_tokens}, prompt_length={len(user_prompt)}")
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                timeout=60.0
            )
            logger.info(f"✅ OpenAI API call successful: response_id={response.id if hasattr(response, 'id') else 'N/A'}")
            
            if not response.choices or len(response.choices) == 0:
                logger.error("❌ OpenAI API returned empty response - no choices available")
                raise ValueError("OpenAI API returned empty response - no choices available")
            
            answer = response.choices[0].message.content
            
            if not answer:
                logger.error(f"❌ OpenAI API returned empty answer content. Response object: {type(response).__name__}, choices count: {len(response.choices) if response.choices else 0}")
                raise ValueError("OpenAI API returned empty answer content")
            
            logger.info(f"✅ LLM generated answer successfully (length: {len(answer)} chars)")
            
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
            error_type = type(e).__name__
            error_msg = str(e)
            logger.error(f"❌ Error generating answer: {error_type}: {error_msg}", exc_info=True)
            
            # Log specific error types for better debugging
            if "API key" in error_msg or "authentication" in error_msg.lower():
                logger.error("❌ OpenAI API authentication failed - check OPENAI_API_KEY")
            elif "rate limit" in error_msg.lower():
                logger.error("❌ OpenAI API rate limit exceeded")
            elif "timeout" in error_msg.lower():
                logger.error("❌ OpenAI API request timed out")
            elif "model" in error_msg.lower() and "not found" in error_msg.lower():
                logger.error(f"❌ OpenAI model not found: {self.model_name}")
            
            # Return error details for debugging
            return {
                "answer": f"Error generating response: {error_msg}",
                "citations": [],
                "citation_validation": {"has_citations": False, "error": error_msg, "error_type": error_type},
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