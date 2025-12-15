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
    description: str = """Look up a specific UK statute by name. ALWAYS include section information if the user's query mentions a specific section.
    
Use this tool when the user asks about a specific statute or Act, such as:
- "Sale of Goods Act 1979" (for general statute lookup)
- "Sale of Goods Act 1979 Section 7" (if user asks about Section 7)
- "Consumer Rights Act 2015"
- "Data Protection Act 2018 Section 3" (if user asks about Section 3)

IMPORTANT: If the user's query mentions a specific section (e.g., "Section 7", "s.7", "section 7"), 
include that section information in the statute_name parameter, for example: "Sale of Goods Act 1979 Section 7".

The tool will search for and return relevant sections from the specified statute, prioritizing the specified section if provided.
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
            
            # Build better search query - use the statute name directly
            # If the statute_name contains section info, include it in the query
            query = statute_name  # Use the exact statute name for better matching
            
            # Check if query mentions a specific section (e.g., "Section 7" or "s.7")
            import re
            section_match = re.search(r'section\s+(\d+)|s\.?\s*(\d+)', statute_name.lower())
            target_section = None
            if section_match:
                target_section = section_match.group(1) or section_match.group(2)
                # Enhance query to prioritize the specific section
                query = f"{statute_name} Section {target_section}"
            
            # Filter by document_type - map "statute" to "Legislation" (actual metadata value)
            metadata_filter = MetadataFilter()
            metadata_filter.add_equals_filter("document_type", "Legislation")  # Use "Legislation" not "statute"
            
            # Perform search with filter first
            results = self.rag_service.search(
                query=query,
                top_k=15,  # Increased from 10 to get more results
                use_hybrid=True,
                metadata_filter=metadata_filter,
                include_explanation=True,
                highlight_sources=True
            )
            
            # If no results with filter, try without filter (fallback)
            if not results:
                logger.info(f"No results with Legislation filter for '{statute_name}', trying without filter...")
                results = self.rag_service.search(
                    query=query,
                    top_k=15,
                    use_hybrid=True,
                    metadata_filter=None,  # No filter - broader search
                    include_explanation=True,
                    highlight_sources=True
            )
            
            if not results:
                return f"Could not find information about '{statute_name}' in the knowledge base. Please check the spelling or try a different search."
            
            # Filter results to those matching the statute name (in title or text)
            # Prioritize results that match the target section if specified
            statute_results = []
            statute_name_lower = statute_name.lower()
            # Extract key words (remove common words like "Act", "of", "the")
            key_words = [w for w in statute_name_lower.split() if w not in ["act", "of", "the", "a", "an", "section"] and len(w) > 2]
            
            for result in results:
                title = result.get("metadata", {}).get("title", "").lower()
                text = result.get("text", "").lower()
                source = result.get("metadata", {}).get("source", "").lower()
                section = result.get("section", "").lower()
                
                # Check if statute name or key words appear in title, source, or text
                matches_title = statute_name_lower in title or any(word in title for word in key_words)
                matches_source = statute_name_lower in source or any(word in source for word in key_words)
                matches_text = statute_name_lower in text[:500] or sum(1 for word in key_words if word in text[:500]) >= len(key_words) * 0.5
                
                # If target section specified, prioritize exact section matches
                matches_section = False
                if target_section:
                    matches_section = f"section {target_section}" in section or f"s.{target_section}" in section or f"section {target_section}" in text[:200]
                
                if matches_title or matches_source or matches_text:
                    # Add priority score for section matching
                    result["_priority"] = 2 if matches_section else (1 if matches_title else 0)
                    statute_results.append(result)
            
            if not statute_results:
                # If no exact matches, return top results anyway (might still be relevant)
                logger.warning(f"No exact matches for '{statute_name}', returning top results")
                statute_results = results[:5]  # Return top 5 results even if not exact match
            
            # Format results with better information - prioritize exact matches
            formatted_results = []
            # Sort by priority (section match > title match > score)
            sorted_results = sorted(
                statute_results,
                key=lambda x: (
                    x.get("_priority", 0),  # Section match priority first (if target_section specified)
                    1 if statute_name.lower() in x.get("metadata", {}).get("title", "").lower() else 0,  # Exact title match second
                    -x.get("similarity_score", 0.0)  # Then by score (descending)
                ),
                reverse=True
            )
            
            for i, result in enumerate(sorted_results[:8], 1):  # Top 8 results
                title = result.get("metadata", {}).get("title", "Unknown Title")
                section = result.get("section", "Unknown Section")
                source = result.get("metadata", {}).get("source", "Unknown Source")
                text = result.get("text", "")
                score = result.get("similarity_score", 0.0)
                
                # Get highlighted text if available, otherwise use full text
                highlighted = result.get("highlighted_text", text)
                
                # Show more content (up to 1000 chars) for better context
                content_preview = highlighted[:1000] if len(highlighted) > 1000 else highlighted
                
                formatted_results.append(
                    f"[{i}] {title} - {section}\n"
                    f"Source: {source}\n"
                    f"Relevance Score: {score:.3f}\n"
                    f"Full Content: {content_preview}\n"
                    f"Chunk ID: {result.get('chunk_id', 'unknown')}"
                )
            
            return f"SUCCESSFULLY RETRIEVED {len(statute_results)} relevant sections from the knowledge base for '{statute_name}':\n\n" + "\n\n".join(formatted_results)
            
        except Exception as e:
            logger.error(f"Statute lookup tool error: {e}", exc_info=True)
            return f"Error looking up statute: {str(e)}"
    
    async def _arun(self, statute_name: str) -> str:
        """Async execution (delegates to sync for now)"""
        return self._run(statute_name)

