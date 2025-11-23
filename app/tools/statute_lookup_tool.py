# Legal Chatbot - Statute Lookup Tool for LangChain Agent
# Allows agent to look up specific UK statutes by name

from typing import Optional, Any
try:
    from langchain_core.tools import BaseTool
except ImportError:
    from langchain.tools import BaseTool
from pydantic import BaseModel, Field
import logging

logger = logging.getLogger(__name__)


class StatuteLookupInput(BaseModel):
    """Input for statute lookup tool"""
    statute_name: str = Field(description="Name of the UK statute to look up (e.g., 'Sale of Goods Act 1979', 'Consumer Rights Act 2015')")


class StatuteLookupTool(BaseTool):
    """
    Tool for looking up specific UK statutes by name.
    
    This tool searches for a specific statute by name and returns relevant sections.
    Use this when you need information about a specific piece of legislation.
    """
    
    name: str = "get_specific_statute"
    description: str = """Look up a specific UK statute by name.
    
Use this tool when the user asks about a specific statute or Act, such as:
- "Sale of Goods Act 1979"
- "Consumer Rights Act 2015"
- "Data Protection Act 2018"
- "Employment Rights Act 1996"

The tool will search for and return relevant sections from the specified statute.
"""
    
    args_schema: type[BaseModel] = StatuteLookupInput
    
    rag_service: Any = Field(default=None, exclude=True)  # Exclude from Pydantic validation
    
    def __init__(self, rag_service, **kwargs):
        """Initialize statute lookup tool with RAG service"""
        super().__init__(**kwargs)
        object.__setattr__(self, 'rag_service', rag_service)  # Bypass Pydantic validation
    
    def _run(self, statute_name: str) -> str:
        """Execute the statute lookup tool"""
        try:
            from retrieval.metadata_filter import MetadataFilter
            
            # Build search query for the statute
            query = f"{statute_name} statute sections provisions"
            
            # Filter by document_type if possible
            metadata_filter = MetadataFilter()
            metadata_filter.add_equals_filter("document_type", "statute")
            
            # Perform search
            results = self.rag_service.search(
                query=query,
                top_k=10,
                use_hybrid=True,
                metadata_filter=metadata_filter,
                include_explanation=True
            )
            
            if not results:
                return f"Could not find information about '{statute_name}'. Please check the spelling or try a different search."
            
            # Filter results to those matching the statute name (in title or text)
            statute_results = []
            statute_name_lower = statute_name.lower()
            for result in results:
                title = result.get("metadata", {}).get("title", "").lower()
                text = result.get("text", "").lower()
                
                # Check if statute name appears in title or text
                if statute_name_lower in title or statute_name_lower in text[:500]:
                    statute_results.append(result)
            
            if not statute_results:
                return f"Found documents but none specifically match '{statute_name}'. Found {len(results)} related documents."
            
            # Format results
            formatted_results = []
            for i, result in enumerate(statute_results[:5], 1):  # Top 5 results
                title = result.get("metadata", {}).get("title", "Unknown Title")
                section = result.get("section", "Unknown Section")
                text = result.get("text", "")[:600]
                score = result.get("similarity_score", 0.0)
                
                formatted_results.append(
                    f"[{i}] {title} - {section}\n"
                    f"Relevance: {score:.3f}\n"
                    f"Content: {text}...\n"
                    f"Chunk ID: {result.get('chunk_id', 'unknown')}"
                )
            
            return f"Found {len(statute_results)} sections from '{statute_name}':\n\n" + "\n\n".join(formatted_results)
            
        except Exception as e:
            logger.error(f"Statute lookup tool error: {e}", exc_info=True)
            return f"Error looking up statute: {str(e)}"
    
    async def _arun(self, statute_name: str) -> str:
        """Async execution (delegates to sync for now)"""
        return self._run(statute_name)

