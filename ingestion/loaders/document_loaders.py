# Legal Chatbot - Document Loaders

import os
import json
import requests
from typing import List, Dict, Any, Optional
from pathlib import Path
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class DocumentMetadata:
    """Document metadata structure"""
    title: str
    source: str
    jurisdiction: str
    document_type: str
    url: Optional[str] = None
    section: Optional[str] = None
    date: Optional[str] = None
    file_path: Optional[str] = None


@dataclass
class DocumentChunk:
    """Document chunk structure"""
    chunk_id: str
    text: str
    metadata: DocumentMetadata
    chunk_index: int
    start_char: int
    end_char: int


class BaseLoader:
    """Base class for document loaders"""
    
    def load_documents(self, source: str) -> List[DocumentChunk]:
        """Load documents from source and return chunks"""
        raise NotImplementedError


class PDFLoader(BaseLoader):
    """PDF document loader"""
    
    def __init__(self):
        try:
            import PyPDF2
            self.PyPDF2 = PyPDF2
        except ImportError:
            logger.error("PyPDF2 not installed. Install with: pip install PyPDF2")
            raise
    
    def load_documents(self, file_path: str) -> List[DocumentChunk]:
        """Load PDF documents and return chunks"""
        chunks = []
        
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = self.PyPDF2.PdfReader(file)
                
                for page_num, page in enumerate(pdf_reader.pages):
                    text = page.extract_text()
                    if text.strip():
                        chunk = DocumentChunk(
                            chunk_id=f"pdf_{Path(file_path).stem}_page_{page_num}",
                            text=text.strip(),
                            metadata=DocumentMetadata(
                                title=Path(file_path).stem,
                                source="PDF",
                                jurisdiction="UK",
                                document_type="PDF",
                                file_path=file_path
                            ),
                            chunk_index=page_num,
                            start_char=0,
                            end_char=len(text)
                        )
                        chunks.append(chunk)
                        
        except Exception as e:
            logger.error(f"Error loading PDF {file_path}: {e}")
            
        return chunks


class TextLoader(BaseLoader):
    """Plain text document loader"""
    
    def load_documents(self, file_path: str) -> List[DocumentChunk]:
        """Load text documents and return chunks"""
        chunks = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
                
                if text.strip():
                    chunk = DocumentChunk(
                        chunk_id=f"txt_{Path(file_path).stem}",
                        text=text.strip(),
                        metadata=DocumentMetadata(
                            title=Path(file_path).stem,
                            source="TEXT",
                            jurisdiction="UK",
                            document_type="TEXT",
                            file_path=file_path
                        ),
                        chunk_index=0,
                        start_char=0,
                        end_char=len(text)
                    )
                    chunks.append(chunk)
                    
        except Exception as e:
            logger.error(f"Error loading text file {file_path}: {e}")
            
        return chunks


class UKLegislationLoader(BaseLoader):
    """UK Legislation API loader"""
    
    def __init__(self):
        self.base_url = "https://www.legislation.gov.uk/api/v1"
    
    def load_documents(self, act_id: str) -> List[DocumentChunk]:
        """Load UK legislation from API"""
        chunks = []
        
        try:
            # Get act information
            response = requests.get(f"{self.base_url}/{act_id}")
            response.raise_for_status()
            
            act_data = response.json()
            
            # Extract text content
            if 'text' in act_data:
                text = act_data['text']
                
                chunk = DocumentChunk(
                    chunk_id=f"uk_leg_{act_id}",
                    text=text,
                    metadata=DocumentMetadata(
                        title=act_data.get('title', act_id),
                        source="UK Legislation API",
                        jurisdiction="UK",
                        document_type="Legislation",
                        url=f"https://www.legislation.gov.uk/{act_id}",
                        date=act_data.get('date')
                    ),
                    chunk_index=0,
                    start_char=0,
                    end_char=len(text)
                )
                chunks.append(chunk)
                
        except Exception as e:
            logger.error(f"Error loading UK legislation {act_id}: {e}")
            
        return chunks


class JSONLoader(BaseLoader):
    """JSON document loader for UK legislation and structured legal documents"""
    
    def load_documents(self, file_path: str) -> List[DocumentChunk]:
        """Load JSON documents and return chunks"""
        chunks = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
                
                # Handle UK legislation JSON format
                if 'content' in data:
                    title = data.get('title', Path(file_path).stem)
                    url = data.get('url')
                    content = data.get('content', '')
                    
                    # Extract Act name from title (e.g., "Employment Rights Act 1996")
                    act_name = title
                    
                    # Parse content to extract sections properly
                    # UK legislation format: "Section X. Text..." or "X. Text..." or "Part X Section Y"
                    import re
                    
                    # Pattern to find section markers: "1. ", "Section 1", "Part I", etc.
                    section_pattern = r'(?:Section|S\.|S)\s*(\d+[A-Za-z]?)[\.:]\s*|^(\d+[A-Za-z]?)[\.:]\s+|^Part\s+([IVX]+|[\d]+)\s+|^PART\s+([IVX]+|[\d]+)\s+'
                    
                    # Split content by section markers
                    section_matches = list(re.finditer(section_pattern, content, re.MULTILINE))
                    
                    if section_matches:
                        # Create chunks for each section
                        for i, match in enumerate(section_matches):
                            start_pos = match.start()
                            end_pos = section_matches[i + 1].start() if i + 1 < len(section_matches) else len(content)
                            
                            section_text = content[start_pos:end_pos].strip()
                            if not section_text or len(section_text) < 50:  # Skip very short sections
                                continue
                            
                            # Extract section number from match
                            section_num = match.group(1) or match.group(2) or match.group(3) or match.group(4)
                            if not section_num:
                                continue
                            
                            # Skip "Section 0" - doesn't exist
                            if section_num == "0" or section_num == "0A":
                                continue
                            
                            # Extract section title (first line or first sentence)
                            section_title = section_text.split('\n')[0].strip()[:100]
                            if not section_title:
                                section_title = section_text.split('.')[0].strip()[:100]
                            
                            # Create chunk with proper section metadata
                            chunk = DocumentChunk(
                                chunk_id=f"json_{Path(file_path).stem}_s{section_num}",
                                text=section_text,
                                metadata=DocumentMetadata(
                                    title=f"{act_name} - Section {section_num}",
                                    source=act_name,  # Use Act name as source
                                    jurisdiction="UK",
                                    document_type="Legislation",
                                    url=url,
                                    section=f"Section {section_num}"  # Proper section identifier
                                ),
                                chunk_index=int(re.findall(r'\d+', section_num)[0]) if re.findall(r'\d+', section_num) else i,
                                start_char=start_pos,
                                end_char=end_pos
                            )
                            chunks.append(chunk)
                    else:
                        # If no sections found, chunk by paragraphs (fallback)
                        paragraphs = content.split('\n\n')
                        for idx, para in enumerate(paragraphs):
                            para = para.strip()
                            if para and len(para) > 100:
                                chunk = DocumentChunk(
                                    chunk_id=f"json_{Path(file_path).stem}_para_{idx}",
                                    text=para,
                                    metadata=DocumentMetadata(
                                        title=act_name,
                                        source=act_name,
                                        jurisdiction="UK",
                                        document_type="Legislation",
                                        url=url
                                    ),
                                    chunk_index=idx,
                                    start_char=0,
                                    end_char=len(para)
                                )
                                chunks.append(chunk)
                    
                    # If still no chunks, create one for entire content
                    if not chunks and content.strip():
                        chunk = DocumentChunk(
                            chunk_id=f"json_{Path(file_path).stem}",
                            text=content.strip(),
                            metadata=DocumentMetadata(
                                title=act_name,
                                source=act_name,
                                jurisdiction="UK",
                                document_type="Legislation",
                                url=url
                            ),
                            chunk_index=0,
                            start_char=0,
                            end_char=len(content)
                        )
                        chunks.append(chunk)
                else:
                    # Generic JSON - convert to text
                    text = json.dumps(data, indent=2)
                    if text.strip():
                        chunk = DocumentChunk(
                            chunk_id=f"json_{Path(file_path).stem}",
                            text=text.strip(),
                            metadata=DocumentMetadata(
                                title=Path(file_path).stem,
                                source="JSON",
                                jurisdiction="UK",
                                document_type="JSON",
                                file_path=file_path
                            ),
                            chunk_index=0,
                            start_char=0,
                            end_char=len(text)
                        )
                        chunks.append(chunk)
                        
        except Exception as e:
            logger.error(f"Error loading JSON file {file_path}: {e}")
            
        return chunks


class DocumentLoaderFactory:
    """Factory for creating document loaders"""
    
    @staticmethod
    def get_loader(file_path: str) -> BaseLoader:
        """Get appropriate loader based on file extension"""
        extension = Path(file_path).suffix.lower()
        
        if extension == '.pdf':
            return PDFLoader()
        elif extension in ['.txt', '.md']:
            return TextLoader()
        elif extension == '.json':
            return JSONLoader()
        else:
            raise ValueError(f"Unsupported file type: {extension}")


# Test the loaders
if __name__ == "__main__":
    # Test with a sample text file
    test_file = "data/raw/test_document.txt"
    
    # Create test directory and file
    os.makedirs("data/raw", exist_ok=True)
    
    with open(test_file, "w") as f:
        f.write("""
        Sale of Goods Act 1979 - Section 12
        
        In a contract of sale, unless the circumstances of the contract are such as to show a different intention, 
        there is an implied condition on the part of the seller that in the case of a sale he has a right to sell 
        the goods, and in the case of an agreement to sell he will have a right to sell the goods at the time 
        when the property is to pass.
        
        This section establishes the fundamental principle that a seller must have the legal right to sell 
        the goods they are offering for sale.
        """)
    
    # Test the loader
    loader = DocumentLoaderFactory.get_loader(test_file)
    chunks = loader.load_documents(test_file)
    
    print(f"Loaded {len(chunks)} chunks:")
    for chunk in chunks:
        print(f"- Chunk ID: {chunk.chunk_id}")
        print(f"- Title: {chunk.metadata.title}")
        print(f"- Text preview: {chunk.text[:100]}...")
        print()


