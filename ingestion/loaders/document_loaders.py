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


