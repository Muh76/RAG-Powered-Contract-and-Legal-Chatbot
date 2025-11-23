# Legal Chatbot - Legal Search Tool for LangChain Agent
# Wraps existing hybrid search functionality

from typing import Optional, List, Dict, Any
try:
    from langchain_core.tools import BaseTool
except ImportError:
    from langchain.tools import BaseTool
from pydantic import BaseModel, Field
import logging

logger = logging.getLogger(__name__)


class LegalSearchInput(BaseModel):
    """Input for legal search tool"""
    query: str = Field(description="Search query for legal documents, statutes, or contracts")
    jurisdiction: Optional[str] = Field(
        default=None,
        description="Filter by jurisdiction (e.g., 'UK', 'US'). If not specified, searches all jurisdictions."
    )
    document_type: Optional[str] = Field(
        default=None,
        description="Filter by document type (e.g., 'statute', 'contract', 'case_law'). If not specified, searches all types."
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of results to return (default: 5, max: 20)"
    )


class LegalSearchTool(BaseTool):
    """
    Tool for searching UK legal documents using hybrid search (BM25 + Semantic).
    
    This tool uses the existing Phase 2 hybrid retrieval system to search for relevant
    legal documents, statutes, and contracts. It supports metadata filtering and
    returns results with citations and relevance scores.
    """
    
    name: str = "search_legal_documents"
    description: str = """Search UK legal documents, statutes, and contracts using advanced hybrid search (BM25 + semantic).
    
Use this tool when you need to:
- Find relevant legal documents or statutes
- Search for specific legal concepts or terms
- Retrieve information about UK law
- Find contracts or case law

The tool returns relevant legal chunks with citations, metadata, and similarity scores.
It supports filtering by jurisdiction and document type.
"""
    
    args_schema: type[BaseModel] = LegalSearchInput
    rag_service: Any = Field(default=None, exclude=True)  # Exclude from Pydantic validation
    
    def __init__(self, rag_service, **kwargs):
        """Initialize legal search tool with RAG service"""
        super().__init__(**kwargs)
        object.__setattr__(self, 'rag_service', rag_service)  # Bypass Pydantic validation
    
    def _run(
        self,
        query: str,
        jurisdiction: Optional[str] = None,
        document_type: Optional[str] = None,
        top_k: int = 5
    ) -> str:
        """Execute the legal search tool"""
        try:
            from retrieval.metadata_filter import MetadataFilter
            
            # Build metadata filter if needed
            metadata_filter = None
            if jurisdiction or document_type:
                metadata_filter = MetadataFilter()
                if jurisdiction:
                    metadata_filter.add_equals_filter("jurisdiction", jurisdiction)
                if document_type:
                    metadata_filter.add_equals_filter("document_type", document_type)
            
            # Perform hybrid search
            results = self.rag_service.search(
                query=query,
                top_k=top_k,
                use_hybrid=True,
                include_explanation=True,
                highlight_sources=True,
                metadata_filter=metadata_filter
            )
            
            if not results:
                return f"No relevant legal documents found for query: '{query}'"
            
            # Format results for LLM
            formatted_results = []
            for i, result in enumerate(results, 1):
                chunk_text = result.get("text", "")[:500]  # Limit text length
                section = result.get("section", "Unknown")
                title = result.get("metadata", {}).get("title", "Unknown Title")
                score = result.get("similarity_score", 0.0)
                
                # Get highlighted text if available
                highlighted = result.get("highlighted_text", chunk_text)
                
                formatted_results.append(
                    f"[{i}] {title} - {section}\n"
                    f"Relevance Score: {score:.3f}\n"
                    f"Content: {highlighted}\n"
                    f"Chunk ID: {result.get('chunk_id', 'unknown')}"
                )
            
            return f"Found {len(results)} relevant legal documents:\n\n" + "\n\n".join(formatted_results)
            
        except Exception as e:
            logger.error(f"Legal search tool error: {e}", exc_info=True)
            return f"Error searching legal documents: {str(e)}"
    
    async def _arun(
        self,
        query: str,
        jurisdiction: Optional[str] = None,
        document_type: Optional[str] = None,
        top_k: int = 5
    ) -> str:
        """Async execution (delegates to sync for now)"""
        return self._run(query, jurisdiction, document_type, top_k)

