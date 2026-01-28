# Legal Chatbot - Document Analyzer Tool for LangChain Agent
# MVP: Executive summary, key clauses extraction, heuristic risk flags.
# NOT legal advice; for informational use only.

import json
import re
import logging
from typing import Optional, List, Dict, Any

try:
    from langchain_core.tools import BaseTool
except ImportError:
    from langchain.tools import BaseTool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Max document length to send to LLM (chars). Keeps context within model limits.
MAX_DOCUMENT_LENGTH = 30000

# Disclaimer included in every analysis output.
DISCLAIMER = (
    "This analysis is for informational purposes only and does not constitute "
    "legal advice. Risk flags are heuristic-based; consult a qualified solicitor "
    "for legal interpretation."
)


class DocumentAnalyzerInput(BaseModel):
    """Input for document analyzer tool."""

    document_text: str = Field(
        description="Full text content of the document to analyze (contract, agreement, or legal text)."
    )
    analysis_type: Optional[str] = Field(
        default="full",
        description="Type of analysis: 'full' (summary + clauses + risk flags) or 'summary' only.",
    )


def _empty_result(error_message: str) -> Dict[str, Any]:
    """Return a structured empty result with disclaimer and optional error."""
    return {
        "executive_summary": [],
        "key_clauses": [],
        "risk_flags": [],
        "disclaimer": DISCLAIMER,
        "error": error_message,
    }


def _build_analysis_prompt() -> str:
    """Build the user prompt that asks the LLM for structured JSON output."""
    return """Analyze the document below and respond with a single valid JSON object only. No markdown, no code fences, no citations, no explanation outside the JSON.

Use this exact structure (no extra keys):
{
  "executive_summary": ["bullet 1", "bullet 2", "..."],
  "key_clauses": [{"section_title": "Title", "excerpt": "Short excerpt"}],
  "risk_flags": [{"flag": "short label", "reason": "brief heuristic reason"}]
}

Rules:
- executive_summary: 5 to 8 bullet points (short sentences) summarizing the document's purpose, parties, main obligations, and critical terms.
- key_clauses: Extract 5 to 12 important clauses or sections; each has "section_title" and "excerpt" (1–2 sentences from the document).
- risk_flags: List 0 to 6 potential risk areas (e.g. liability, termination, indemnity, confidentiality) based only on the document text. Use "flag" for a short label and "reason" for why it was flagged. Heuristic only—not legal advice.
- Output ONLY the JSON object. No other text."""


def _extract_json_from_response(raw: str) -> Optional[Dict[str, Any]]:
    """
    Extract a JSON object from LLM response. Handles markdown code blocks and stray text.
    Returns None if no valid JSON is found.
    """
    if not raw or not raw.strip():
        return None
    text = raw.strip()
    # Remove markdown code block if present
    if "```" in text:
        match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
        if match:
            text = match.group(1).strip()
    # Remove citation-style markers that might appear inside strings (e.g. [1])
    # Only strip trailing/leading brackets that look like citations; be conservative
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to find JSON object boundaries
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                return json.loads(text[start:end])
            except json.JSONDecodeError:
                pass
    return None


def _normalize_result(parsed: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure result has required keys and disclaimer; truncate lists if needed."""
    summary = parsed.get("executive_summary")
    clauses = parsed.get("key_clauses")
    flags = parsed.get("risk_flags")
    if not isinstance(summary, list):
        summary = []
    if not isinstance(clauses, list):
        clauses = []
    if not isinstance(flags, list):
        flags = []
    # Enforce 5–8 bullets for summary
    summary = summary[:8] if len(summary) > 8 else (summary if len(summary) >= 5 else summary)
    clauses = clauses[:15]
    flags = flags[:8]
    # Normalize clause/flag items to dicts with expected keys
    key_clauses = []
    for c in clauses:
        if isinstance(c, dict):
            key_clauses.append({
                "section_title": str(c.get("section_title", ""))[:200],
                "excerpt": str(c.get("excerpt", ""))[:500],
            })
    risk_flags = []
    for f in flags:
        if isinstance(f, dict):
            risk_flags.append({
                "flag": str(f.get("flag", ""))[:100],
                "reason": str(f.get("reason", ""))[:300],
            })
    return {
        "executive_summary": [str(s)[:500] for s in summary],
        "key_clauses": key_clauses,
        "risk_flags": risk_flags,
        "disclaimer": DISCLAIMER,
    }


class DocumentAnalyzerTool(BaseTool):
    """
    Analyzes legal document text and returns structured analysis: executive summary,
    key clauses/sections, and heuristic risk flags. Stateless; uses existing LLM service.
    Output is informational only and does not constitute legal advice.
    """

    name: str = "analyze_document"
    description: str = """Analyze a legal document (contract, agreement, or legal text) to extract:
1) Executive summary (5–8 bullet points),
2) Key clauses or sections with short excerpts,
3) Potential risk flags (heuristic-based; not legal advice).

Use when the user provides or pastes document text to analyze. Input: document_text (required).
Output is JSON. This tool does not give legal advice."""

    args_schema: type[BaseModel] = DocumentAnalyzerInput

    # Excluded from serialization; set in __init__
    llm_service: Any = Field(default=None, exclude=True)

    def __init__(self, llm_service=None, **kwargs):
        """Initialize with existing LLM service (do not create new OpenAI clients)."""
        super().__init__(**kwargs)
        object.__setattr__(self, "llm_service", llm_service)

    def _run(
        self,
        document_text: str,
        analysis_type: str = "full",
    ) -> str:
        """
        Run document analysis. Returns JSON string with executive_summary,
        key_clauses, risk_flags, and disclaimer. Graceful fallback if input is empty.
        """
        # Graceful fallback: empty document
        if not document_text or not document_text.strip():
            logger.info("Document analyzer: empty document text, returning empty result")
            return json.dumps(_empty_result("Document text is empty."), indent=2)

        if not self.llm_service:
            return json.dumps(
                _empty_result("Document analyzer is not available: LLM service not initialized."),
                indent=2,
            )

        # Truncate to stay within context
        text = document_text.strip()
        if len(text) > MAX_DOCUMENT_LENGTH:
            text = text[:MAX_DOCUMENT_LENGTH] + "\n[... document truncated ...]"
            logger.info(f"Document truncated to {MAX_DOCUMENT_LENGTH} chars for analysis")

        # Single chunk for LLM (same interface as generate_legal_answer)
        chunk = {
            "text": text,
            "metadata": {"title": "User document", "source": "document_analyzer"},
        }
        query = _build_analysis_prompt()

        try:
            result = self.llm_service.generate_legal_answer(
                query=query,
                retrieved_chunks=[chunk],
                mode="solicitor",
            )
            answer = result.get("answer") or ""
        except Exception as e:
            logger.error(f"Document analyzer LLM call failed: {e}", exc_info=True)
            return json.dumps(
                _empty_result(f"Analysis failed: {str(e)}"),
                indent=2,
            )

        parsed = _extract_json_from_response(answer)
        if not parsed:
            # Fallback: return structured result with raw answer in error
            return json.dumps(
                {
                    **_empty_result("Could not parse model output as JSON."),
                    "raw_response_preview": (answer[:1000] + "..." if len(answer) > 1000 else answer),
                },
                indent=2,
            )

        normalized = _normalize_result(parsed)
        return json.dumps(normalized, indent=2)

    async def _arun(
        self,
        document_text: str,
        analysis_type: str = "full",
    ) -> str:
        """Async execution (delegates to sync _run; stateless)."""
        return self._run(document_text=document_text, analysis_type=analysis_type)
