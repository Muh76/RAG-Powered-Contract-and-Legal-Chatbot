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
        default=8,  # Increased from 5 for better coverage
        ge=1,
        le=20,
        description="Number of results to return (default: 8, max: 20)"
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
- Answer questions about specific sections of statutes (e.g., "What does Section 7 of the Sale of Goods Act say?")

IMPORTANT: For queries about specific sections of statutes, use this tool with the full query including the section number.
Example: For "What does Section 7 of the Sale of Goods Act 1979 say?", use query="Section 7 Sale of Goods Act 1979 perishing goods".

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
            # Note: Be flexible with filters - many queries work better without strict filtering
            metadata_filter = None
            
            # Only apply filters if explicitly requested and likely to match
            # Many queries work better with broader search
            if jurisdiction and document_type:
                # Both filters - be careful, might be too restrictive
                metadata_filter = MetadataFilter()
                if jurisdiction:
                    # Use case-insensitive matching
                    metadata_filter.add_equals_filter("jurisdiction", jurisdiction.upper())
                if document_type:
                    # Map common document types to actual metadata values
                    # Many chunks use "Legislation" instead of "statute"
                    doc_type_map = {
                        "statute": "Legislation",
                        "legislation": "Legislation",
                        "act": "Legislation"
                    }
                    mapped_type = doc_type_map.get(document_type.lower(), document_type)
                    metadata_filter.add_equals_filter("document_type", mapped_type)
            elif jurisdiction:
                # Only jurisdiction filter - less restrictive
                metadata_filter = MetadataFilter()
                metadata_filter.add_equals_filter("jurisdiction", jurisdiction.upper())
            # If only document_type, skip filter (too restrictive, often fails)
            
            # Perform hybrid search
            # Try with filter first if we have one, but fallback to no filter if no results
            results = []
            if metadata_filter:
                results = self.rag_service.search(
                    query=query,
                    top_k=top_k,
                    use_hybrid=True,
                    include_explanation=True,
                    highlight_sources=True,
                    metadata_filter=metadata_filter
                )
                logger.debug(f"Search with metadata filter returned {len(results)} results")
            
            # If no results (with or without filter), try without any filter
            if not results:
                if metadata_filter:
                    logger.info(f"No results with metadata filter, trying without filter...")
                results = self.rag_service.search(
                    query=query,
                    top_k=top_k,
                    use_hybrid=True,
                    include_explanation=True,
                    highlight_sources=True,
                    metadata_filter=None
                )
                logger.debug(f"Search without filter returned {len(results)} results")
            
            if not results:
                return f"No relevant legal documents found for query: '{query}'"
            
            # Format results for LLM with better structure
            formatted_results = []
            for i, result in enumerate(results, 1):
                text = result.get("text", "")
                section = result.get("section", "Unknown")
                title = result.get("metadata", {}).get("title", "Unknown Title")
                source = result.get("metadata", {}).get("source", "Unknown Source")
                score = result.get("similarity_score", 0.0)
                
                # Get highlighted text if available, otherwise use full text
                highlighted = result.get("highlighted_text", text)
                # Use full text if highlighted is shorter (highlighted might be truncated)
                full_text = result.get("text", "")
                
                # Prefer full text if it's longer and more complete, otherwise use highlighted
                # Show more content (up to 2500 chars) for better context - this is the ACTUAL section content
                if len(full_text) > len(highlighted) and len(full_text) > 500:
                    content_preview = full_text[:2500] if len(full_text) > 2500 else full_text
                else:
                    content_preview = highlighted[:2500] if len(highlighted) > 2500 else highlighted
                
                # Format to make it clear this is the complete section text from the knowledge base
                formatted_results.append(
                    f"[{i}] {title} - {section}\n"
                    f"Source: {source}\n"
                    f"Relevance Score: {score:.3f}\n"
                    f"COMPLETE SECTION TEXT FROM KNOWLEDGE BASE:\n{content_preview}\n"
                    f"(This is the actual legal text from the statute - use it directly in your answer)\n"
                    f"Chunk ID: {result.get('chunk_id', 'unknown')}"
                )
            
            return f"SUCCESSFULLY RETRIEVED {len(results)} relevant legal documents from the knowledge base. The content below is the ACTUAL TEXT from the statutes - use it directly to answer the user's question:\n\n" + "\n\n".join(formatted_results)
            
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

