# Legal Chatbot - Document Chunking

import re
from typing import List, Dict, Any
from dataclasses import dataclass
import logging
from ingestion.loaders.document_loaders import DocumentChunk, DocumentMetadata

logger = logging.getLogger(__name__)


@dataclass
class ChunkingConfig:
    """Configuration for document chunking"""
    chunk_size: int = 1000  # Target chunk size in characters
    overlap_size: int = 200  # Overlap between chunks
    min_chunk_size: int = 100  # Minimum chunk size
    max_chunk_size: int = 1500  # Maximum chunk size
    preserve_sentences: bool = True  # Try to preserve sentence boundaries


class DocumentChunker:
    """Document chunking utility"""
    
    def __init__(self, config: ChunkingConfig = None):
        self.config = config or ChunkingConfig()
    
    def chunk_document(self, chunk: DocumentChunk) -> List[DocumentChunk]:
        """Split a document chunk into smaller chunks"""
        if len(chunk.text) <= self.config.chunk_size:
            return [chunk]
        
        chunks = []
        text = chunk.text
        start = 0
        chunk_index = 0
        
        while start < len(text):
            # Determine end position
            end = min(start + self.config.chunk_size, len(text))
            
            # Try to preserve sentence boundaries
            if self.config.preserve_sentences and end < len(text):
                # Look for sentence endings within the last 200 characters
                sentence_end = self._find_sentence_end(text, start, end)
                if sentence_end > start + self.config.min_chunk_size:
                    end = sentence_end
            
            # Extract chunk text
            chunk_text = text[start:end].strip()
            
            # Skip if too small
            if len(chunk_text) < self.config.min_chunk_size:
                break
            
            # Create new chunk
            new_chunk = DocumentChunk(
                chunk_id=f"{chunk.chunk_id}_part_{chunk_index}",
                text=chunk_text,
                metadata=chunk.metadata,
                chunk_index=chunk_index,
                start_char=start,
                end_char=end
            )
            chunks.append(new_chunk)
            
            # Move start position with overlap
            start = end - self.config.overlap_size
            chunk_index += 1
            
            # Prevent infinite loop
            if start >= end:
                break
        
        return chunks
    
    def _find_sentence_end(self, text: str, start: int, end: int) -> int:
        """Find the end of a sentence within the given range"""
        # Look for sentence endings in the last 200 characters
        search_start = max(start, end - 200)
        search_text = text[search_start:end]
        
        # Common sentence endings
        sentence_endings = ['. ', '! ', '? ', '.\n', '!\n', '?\n']
        
        for ending in sentence_endings:
            pos = search_text.rfind(ending)
            if pos != -1:
                return search_start + pos + len(ending)
        
        return end
    
    def chunk_by_sections(self, chunk: DocumentChunk) -> List[DocumentChunk]:
        """Chunk document by sections (for legal documents)"""
        text = chunk.text
        chunks = []
        
        # Common legal section patterns
        section_patterns = [
            r'\n\s*Section\s+\d+[\.\s]',  # Section 1.
            r'\n\s*Article\s+\d+[\.\s]',  # Article 1.
            r'\n\s*Clause\s+\d+[\.\s]',   # Clause 1.
            r'\n\s*Subsection\s+\(\d+\)',  # Subsection (1)
            r'\n\s*\(\d+\)\s+',           # (1)
        ]
        
        # Find section boundaries
        boundaries = [0]
        for pattern in section_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                boundaries.append(match.start())
        
        boundaries.append(len(text))
        boundaries = sorted(set(boundaries))
        
        # Create chunks for each section
        for i in range(len(boundaries) - 1):
            start = boundaries[i]
            end = boundaries[i + 1]
            section_text = text[start:end].strip()
            
            if len(section_text) > self.config.min_chunk_size:
                # Extract section title if possible
                section_title = self._extract_section_title(section_text)
                
                new_chunk = DocumentChunk(
                    chunk_id=f"{chunk.chunk_id}_section_{i}",
                    text=section_text,
                    metadata=DocumentMetadata(
                        title=f"{chunk.metadata.title} - {section_title}",
                        source=chunk.metadata.source,
                        jurisdiction=chunk.metadata.jurisdiction,
                        document_type=chunk.metadata.document_type,
                        url=chunk.metadata.url,
                        section=section_title,
                        date=chunk.metadata.date,
                        file_path=chunk.metadata.file_path
                    ),
                    chunk_index=i,
                    start_char=start,
                    end_char=end
                )
                chunks.append(new_chunk)
        
        return chunks if chunks else [chunk]
    
    def _extract_section_title(self, text: str) -> str:
        """Extract section title from text"""
        lines = text.split('\n')
        for line in lines[:3]:  # Check first 3 lines
            line = line.strip()
            if line and len(line) < 100:  # Reasonable title length
                return line
        return "Section"
    
    def chunk_by_paragraphs(self, chunk: DocumentChunk) -> List[DocumentChunk]:
        """Chunk document by paragraphs"""
        paragraphs = chunk.text.split('\n\n')
        chunks = []
        
        for i, paragraph in enumerate(paragraphs):
            paragraph = paragraph.strip()
            if len(paragraph) > self.config.min_chunk_size:
                new_chunk = DocumentChunk(
                    chunk_id=f"{chunk.chunk_id}_para_{i}",
                    text=paragraph,
                    metadata=chunk.metadata,
                    chunk_index=i,
                    start_char=0,
                    end_char=len(paragraph)
                )
                chunks.append(new_chunk)
        
        return chunks if chunks else [chunk]


class ChunkingStrategy:
    """Chunking strategy selector"""
    
    def __init__(self):
        self.chunker = DocumentChunker()
    
    def chunk_document(self, chunk: DocumentChunk, strategy: str = "adaptive") -> List[DocumentChunk]:
        """Chunk document using specified strategy"""
        
        if strategy == "fixed":
            return self.chunker.chunk_document(chunk)
        elif strategy == "sections":
            return self.chunker.chunk_by_sections(chunk)
        elif strategy == "paragraphs":
            return self.chunker.chunk_by_paragraphs(chunk)
        elif strategy == "adaptive":
            # Choose strategy based on document type
            if chunk.metadata.document_type == "Legislation":
                return self.chunker.chunk_by_sections(chunk)
            elif len(chunk.text) > 2000:
                return self.chunker.chunk_document(chunk)
            else:
                return [chunk]
        else:
            raise ValueError(f"Unknown chunking strategy: {strategy}")


# Test the chunking
if __name__ == "__main__":
    from ingestion.loaders.document_loaders import DocumentChunk, DocumentMetadata
    
    # Create test document
    test_text = """
    Sale of Goods Act 1979
    
    Section 12 - Implied condition as to title
    
    In a contract of sale, unless the circumstances of the contract are such as to show a different intention, 
    there is an implied condition on the part of the seller that in the case of a sale he has a right to sell 
    the goods, and in the case of an agreement to sell he will have a right to sell the goods at the time 
    when the property is to pass.
    
    Section 13 - Sale by description
    
    Where there is a contract for the sale of goods by description, there is an implied condition that the 
    goods will correspond with the description; and, if the sale is by sample as well as by description, 
    it is not sufficient that the bulk of the goods corresponds with the sample if the goods do not also 
    correspond with the description.
    
    Section 14 - Implied terms about quality or fitness
    
    Except as provided by this section and section 15 below and subject to the provisions of any other 
    enactment, there is no implied condition or warranty about the quality or fitness for any particular 
    purpose of goods supplied under a contract of sale.
    """
    
    test_chunk = DocumentChunk(
        chunk_id="test_doc",
        text=test_text,
        metadata=DocumentMetadata(
            title="Sale of Goods Act 1979",
            source="Test",
            jurisdiction="UK",
            document_type="Legislation"
        ),
        chunk_index=0,
        start_char=0,
        end_char=len(test_text)
    )
    
    # Test different chunking strategies
    strategies = ["fixed", "sections", "paragraphs", "adaptive"]
    
    for strategy in strategies:
        print(f"\n=== {strategy.upper()} CHUNKING ===")
        chunker = ChunkingStrategy()
        chunks = chunker.chunk_document(test_chunk, strategy)
        
        print(f"Created {len(chunks)} chunks:")
        for i, chunk in enumerate(chunks):
            print(f"  {i+1}. {chunk.chunk_id} ({len(chunk.text)} chars)")
            print(f"     Preview: {chunk.text[:80]}...")
            if chunk.metadata.section:
                print(f"     Section: {chunk.metadata.section}")
            print()


