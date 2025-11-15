# Legal Chatbot - Metadata Filter
# Phase 2: Module 3 - Metadata Filtering System

from typing import List, Dict, Any, Optional, Set, Union
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class FilterOperator(Enum):
    """Filter operators for metadata filtering"""
    EQUALS = "eq"
    NOT_EQUALS = "ne"
    IN = "in"
    NOT_IN = "nin"
    CONTAINS = "contains"
    STARTS_WITH = "startswith"
    ENDS_WITH = "endswith"
    GREATER_THAN = "gt"
    LESS_THAN = "lt"
    GREATER_THAN_OR_EQUAL = "gte"
    LESS_THAN_OR_EQUAL = "lte"


class MetadataFilter:
    """
    Metadata filter for filtering chunks based on structured metadata.
    
    Supports filtering on various metadata fields such as:
    - jurisdiction (e.g., "UK", "US")
    - document_type (e.g., "statute", "contract", "case_law")
    - source (e.g., "CUAD", "legislation.gov.uk")
    - section (e.g., "Section 12", "Chapter 3")
    - title, date, legal_domain, etc.
    """
    
    def __init__(self):
        """Initialize metadata filter."""
        self.filters: List[Dict[str, Any]] = []
    
    def add_filter(
        self,
        field: str,
        value: Union[str, List[str], int, float],
        operator: Union[FilterOperator, str] = FilterOperator.EQUALS
    ) -> 'MetadataFilter':
        """
        Add a filter condition.
        
        Args:
            field: Metadata field name (e.g., "jurisdiction", "document_type")
            value: Filter value(s)
            operator: Filter operator (default: EQUALS)
            
        Returns:
            self (for method chaining)
        """
        if isinstance(operator, str):
            try:
                operator = FilterOperator(operator)
            except ValueError:
                raise ValueError(f"Invalid operator: {operator}")
        
        self.filters.append({
            "field": field,
            "value": value,
            "operator": operator
        })
        
        logger.debug(f"Added filter: {field} {operator.value} {value}")
        return self
    
    def add_equals_filter(self, field: str, value: str) -> 'MetadataFilter':
        """Add an equals filter (convenience method)."""
        return self.add_filter(field, value, FilterOperator.EQUALS)
    
    def add_in_filter(self, field: str, values: List[str]) -> 'MetadataFilter':
        """Add an 'in' filter (field value in list) (convenience method)."""
        return self.add_filter(field, values, FilterOperator.IN)
    
    def add_not_in_filter(self, field: str, values: List[str]) -> 'MetadataFilter':
        """Add a 'not in' filter (field value not in list) (convenience method)."""
        return self.add_filter(field, values, FilterOperator.NOT_IN)
    
    def add_contains_filter(self, field: str, value: str) -> 'MetadataFilter':
        """Add a contains filter (convenience method)."""
        return self.add_filter(field, value, FilterOperator.CONTAINS)
    
    def clear(self) -> 'MetadataFilter':
        """Clear all filters."""
        self.filters.clear()
        return self
    
    def filter_chunks(
        self,
        chunks: List[Dict[str, Any]],
        metadata_key: str = "metadata"
    ) -> List[Dict[str, Any]]:
        """
        Filter chunks based on metadata filters.
        
        Args:
            chunks: List of chunks (each chunk is a dict with metadata)
            metadata_key: Key in chunk dict where metadata is stored (default: "metadata")
            
        Returns:
            Filtered list of chunks
        """
        if not self.filters:
            return chunks
        
        filtered_chunks = []
        
        for chunk in chunks:
            # Extract metadata
            metadata = chunk.get(metadata_key, {})
            if not isinstance(metadata, dict):
                continue
            
            # Check if chunk matches all filters (AND logic)
            matches_all = True
            for filter_condition in self.filters:
                field = filter_condition["field"]
                value = filter_condition["value"]
                operator = filter_condition["operator"]
                
                # Get field value from metadata
                field_value = metadata.get(field)
                
                # Apply filter
                if not self._apply_filter(field_value, value, operator):
                    matches_all = False
                    break
            
            if matches_all:
                filtered_chunks.append(chunk)
        
        logger.debug(f"Filtered {len(chunks)} chunks -> {len(filtered_chunks)} chunks")
        return filtered_chunks
    
    def filter_indices(
        self,
        chunks: List[Dict[str, Any]],
        metadata_key: str = "metadata"
    ) -> List[int]:
        """
        Filter chunks and return matching indices.
        
        Args:
            chunks: List of chunks
            metadata_key: Key in chunk dict where metadata is stored
            
        Returns:
            List of indices of chunks that match all filters
        """
        if not self.filters:
            return list(range(len(chunks)))
        
        matching_indices = []
        
        for idx, chunk in enumerate(chunks):
            metadata = chunk.get(metadata_key, {})
            if not isinstance(metadata, dict):
                continue
            
            matches_all = True
            for filter_condition in self.filters:
                field = filter_condition["field"]
                value = filter_condition["value"]
                operator = filter_condition["operator"]
                
                field_value = metadata.get(field)
                
                if not self._apply_filter(field_value, value, operator):
                    matches_all = False
                    break
            
            if matches_all:
                matching_indices.append(idx)
        
        return matching_indices
    
    def _apply_filter(
        self,
        field_value: Any,
        filter_value: Any,
        operator: FilterOperator
    ) -> bool:
        """
        Apply a single filter condition.
        
        Args:
            field_value: Value from metadata field
            filter_value: Filter value to compare against
            operator: Filter operator
            
        Returns:
            True if field_value matches filter condition, False otherwise
        """
        # Handle None/missing field values
        if field_value is None:
            return False
        
        # Apply operator
        if operator == FilterOperator.EQUALS:
            return str(field_value).lower() == str(filter_value).lower()
        
        elif operator == FilterOperator.NOT_EQUALS:
            return str(field_value).lower() != str(filter_value).lower()
        
        elif operator == FilterOperator.IN:
            if not isinstance(filter_value, list):
                filter_value = [filter_value]
            field_value_str = str(field_value).lower()
            return any(str(v).lower() == field_value_str for v in filter_value)
        
        elif operator == FilterOperator.NOT_IN:
            if not isinstance(filter_value, list):
                filter_value = [filter_value]
            field_value_str = str(field_value).lower()
            return not any(str(v).lower() == field_value_str for v in filter_value)
        
        elif operator == FilterOperator.CONTAINS:
            return str(filter_value).lower() in str(field_value).lower()
        
        elif operator == FilterOperator.STARTS_WITH:
            return str(field_value).lower().startswith(str(filter_value).lower())
        
        elif operator == FilterOperator.ENDS_WITH:
            return str(field_value).lower().endswith(str(filter_value).lower())
        
        elif operator == FilterOperator.GREATER_THAN:
            try:
                return float(field_value) > float(filter_value)
            except (ValueError, TypeError):
                return False
        
        elif operator == FilterOperator.LESS_THAN:
            try:
                return float(field_value) < float(filter_value)
            except (ValueError, TypeError):
                return False
        
        elif operator == FilterOperator.GREATER_THAN_OR_EQUAL:
            try:
                return float(field_value) >= float(filter_value)
            except (ValueError, TypeError):
                return False
        
        elif operator == FilterOperator.LESS_THAN_OR_EQUAL:
            try:
                return float(field_value) <= float(filter_value)
            except (ValueError, TypeError):
                return False
        
        else:
            logger.warning(f"Unknown operator: {operator}")
            return False
    
    def is_empty(self) -> bool:
        """Check if filter has no conditions."""
        return len(self.filters) == 0
    
    def __len__(self) -> int:
        """Return number of filter conditions."""
        return len(self.filters)
    
    def __repr__(self) -> str:
        """String representation of filter."""
        if not self.filters:
            return "MetadataFilter(empty)"
        return f"MetadataFilter({len(self.filters)} conditions)"

