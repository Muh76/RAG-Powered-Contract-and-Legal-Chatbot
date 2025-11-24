#!/usr/bin/env python3
# Legal Chatbot - Data Ingestion Script
# Run this to build FAISS index from legal documents

import os
import sys
import gc
import faiss
import pickle
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ingestion.loaders.document_loaders import DocumentLoaderFactory
from ingestion.chunkers.document_chunker import ChunkingStrategy, ChunkingConfig
from retrieval.embeddings.openai_embedding_generator import OpenAIEmbeddingGenerator, OpenAIEmbeddingConfig
from app.core.config import settings

def ingest_data():
    """Ingest documents and create FAISS index"""
    print("🚀 Starting data ingestion...")
    print("=" * 60)
    
    # PORTFOLIO MODE: Limit to subset for faster indexing and demo purposes
    MAX_CHUNKS_TO_INDEX = 5000  # 5K chunks - very fast indexing for portfolio demo (~5-10 min indexing)
    print(f"📊 PORTFOLIO MODE: Limiting to {MAX_CHUNKS_TO_INDEX:,} chunks for faster indexing")
    print(f"   ✅ Perfect for demo - can expand later if needed")
    print("")
    
    embedding_gen = None  # Initialize to None for cleanup
    
    try:
        # 1. Load documents
        data_dir = project_root / "data" / "raw"
        
        if not data_dir.exists():
            print(f"❌ Data directory not found: {data_dir}")
            print(f"   Creating directory and sample file...")
            data_dir.mkdir(parents=True, exist_ok=True)
            
            # Create sample file
            sample_file = data_dir / "uk_legal_sample.txt"
            with open(sample_file, "w") as f:
                f.write("""
Sale of Goods Act 1979

Section 12 - Implied condition as to title

In a contract of sale, unless the circumstances of the contract are such as to show a different intention, 
there is an implied condition on the part of the seller that in the case of a sale he has a right to sell 
the goods, and in the case of an agreement to sell he will have a right to sell the goods at the time 
when the property is to pass.

Section 13 - Sale by description

Where there is a contract for the sale of goods by description, there is an implied condition that the 
goods will correspond with the description.

Section 14 - Implied terms about quality or fitness

Except as provided by this section and section 15 below, there is no implied condition or warranty about 
the quality or fitness for any particular purpose of goods supplied under a contract of sale.

Employment Rights Act 1996

Section 1 - Statement of initial employment particulars

An employer shall give to an employee a written statement of particulars of employment.

Section 2 - Statement of initial employment particulars

The statement required by section 1 shall contain particulars of the names of the employer and employee.

Data Protection Act 2018

Section 1 - The data protection principles

Personal data shall be processed lawfully, fairly and in a transparent manner.

Section 2 - The data protection principles

Personal data shall be collected for specified, explicit and legitimate purposes.
""")
            print(f"✅ Created sample file: {sample_file}")
        
        documents = []
        
        # Load .txt files from data/raw
        for file_path in data_dir.glob("*.txt"):
            print(f"📄 Loading: {file_path.name}")
            try:
                loader = DocumentLoaderFactory.get_loader(str(file_path))
                chunks = loader.load_documents(str(file_path))
                documents.extend(chunks)
            except Exception as e:
                print(f"⚠️ Error loading {file_path.name}: {e}")
        
        # Load JSON files from data/uk_legislation
        uk_legislation_dir = project_root / "data" / "uk_legislation"
        if uk_legislation_dir.exists():
            print(f"\n📚 Loading UK Legislation files from {uk_legislation_dir}")
            for file_path in uk_legislation_dir.glob("*.json"):
                print(f"📄 Loading: {file_path.name}")
                try:
                    loader = DocumentLoaderFactory.get_loader(str(file_path))
                    chunks = loader.load_documents(str(file_path))
                    documents.extend(chunks)
                    print(f"   ✅ Loaded {len(chunks)} chunks from {file_path.name}")
                except Exception as e:
                    print(f"⚠️ Error loading {file_path.name}: {e}")
        else:
            print(f"⚠️ UK legislation directory not found: {uk_legislation_dir}")
        
        # Load CUAD dataset (parquet files)
        cuad_dir = project_root / "data" / "cuad" / "data"
        if cuad_dir.exists():
            print(f"\n📚 Loading CUAD dataset from {cuad_dir}")
            parquet_files = list(cuad_dir.glob("*.parquet"))
            
            if parquet_files:
                print(f"   Found {len(parquet_files)} parquet files")
                for file_path in sorted(parquet_files):
                    print(f"📄 Loading: {file_path.name}")
                    try:
                        loader = DocumentLoaderFactory.get_loader(str(file_path))
                        chunks = loader.load_documents(str(file_path))
                        documents.extend(chunks)
                        print(f"   ✅ Loaded {len(chunks)} chunks from {file_path.name}")
                    except Exception as e:
                        print(f"   ⚠️ Error loading {file_path.name}: {e}")
                        import traceback
                        traceback.print_exc()
            else:
                print(f"   ⚠️ No parquet files found in {cuad_dir}")
        else:
            print(f"\n⚠️ CUAD directory not found: {cuad_dir}")
        
        print(f"✅ Loaded {len(documents)} documents")
        
        if len(documents) == 0:
            print("❌ No documents found! Add .txt files to data/raw/")
            return
        
        # 2. Chunk documents
        chunker = ChunkingStrategy()
        config = ChunkingConfig(
            chunk_size=600,
            overlap_size=100,
            preserve_sentences=True
        )
        chunker.chunker.config = config
        
        all_chunks = []
        for doc in documents:
            chunks = chunker.chunk_document(doc, "sections")
            all_chunks.extend(chunks)
        
        total_chunks_created = len(all_chunks)
        print(f"✅ Created {total_chunks_created:,} chunks")
        
        # PORTFOLIO MODE: Limit chunks to subset for faster indexing
        if total_chunks_created > MAX_CHUNKS_TO_INDEX:
            print(f"\n📊 Applying portfolio mode limit: {MAX_CHUNKS_TO_INDEX:,} chunks (from {total_chunks_created:,} total)")
            
            # Strategy: Prioritize UK legislation (important, small) + CUAD sample
            # Separate chunks by source type
            uk_legislation_chunks = []
            cuad_chunks = []
            other_chunks = []
            
            for chunk in all_chunks:
                source = getattr(chunk.metadata, 'source', '').lower()
                doc_type = getattr(chunk.metadata, 'document_type', '').lower()
                
                if 'legislation' in doc_type or 'act' in source or 'regulation' in source:
                    uk_legislation_chunks.append(chunk)
                elif 'cuad' in chunk.chunk_id.lower() or 'contract' in source.lower():
                    cuad_chunks.append(chunk)
                else:
                    other_chunks.append(chunk)
            
            # Keep ALL UK legislation (small, important)
            selected_chunks = uk_legislation_chunks.copy()
            remaining_slots = MAX_CHUNKS_TO_INDEX - len(selected_chunks)
            
            print(f"   ✅ Keeping ALL {len(uk_legislation_chunks)} UK legislation chunks")
            
            # Fill remaining slots with CUAD chunks
            if cuad_chunks and remaining_slots > 0:
                cuad_to_add = min(remaining_slots, len(cuad_chunks))
                selected_chunks.extend(cuad_chunks[:cuad_to_add])
                remaining_slots -= cuad_to_add
                print(f"   ✅ Adding {cuad_to_add:,} CUAD chunks (sample from {len(cuad_chunks):,} total)")
            
            # Add other chunks if space remains
            if other_chunks and remaining_slots > 0:
                other_to_add = min(remaining_slots, len(other_chunks))
                selected_chunks.extend(other_chunks[:other_to_add])
                print(f"   ✅ Adding {other_to_add:,} other chunks")
            
            all_chunks = selected_chunks
            print(f"   📊 Final selection: {len(all_chunks):,} chunks (reduced from {total_chunks_created:,})")
            print(f"   ⚡ This will significantly speed up indexing!")
        else:
            print(f"   ℹ️  Total chunks ({total_chunks_created:,}) is within limit - no reduction needed")
        
        print("")
        
        # 3. Generate embeddings using OpenAI API (NO PyTorch - eliminates segfaults!)
        print("🧠 Generating embeddings using OpenAI API...")
        print("   ✅ Using OpenAI embeddings (NO PyTorch - eliminates segfaults!)")
        
        # Get OpenAI API key
        api_key = os.getenv("OPENAI_API_KEY") or settings.OPENAI_API_KEY
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY not found! "
                "Set it as environment variable: export OPENAI_API_KEY='your-key'"
            )
        
        # Configure OpenAI embeddings with balanced rate limiting
        # With subset approach (20K chunks), we can use moderate settings
        embedding_config = OpenAIEmbeddingConfig(
            api_key=api_key,
            model=settings.OPENAI_EMBEDDING_MODEL if hasattr(settings, 'OPENAI_EMBEDDING_MODEL') else "text-embedding-3-small",
            dimension=None,  # Use model default (1536 for text-embedding-3-small)
            batch_size=50,  # Moderate batch size
            max_retries=5,  # Retries for network/SSL errors
            timeout=60,  # Longer timeout
            requests_per_minute=30  # Moderate rate: 30 requests/min = 2s delay between batches (faster than before)
        )
        
        try:
            embedding_gen = OpenAIEmbeddingGenerator(embedding_config)
            chunk_texts = [chunk.text for chunk in all_chunks]
            
            total_chunks = len(chunk_texts)
            estimated_batches = (total_chunks + embedding_config.batch_size - 1) // embedding_config.batch_size
            estimated_minutes = (estimated_batches * 3) // 60  # 3s delay per batch
            
            print(f"   Generating embeddings for {total_chunks:,} chunks...")
            print(f"   📊 Processing in batches of {embedding_config.batch_size} ({estimated_batches:,} batches)")
            print(f"   ⏳ Estimated time: {estimated_minutes}-{estimated_minutes + 10} minutes (much faster with subset!)")
            print(f"   💡 The script will automatically handle rate limits")
            print("")
            
            # Generate embeddings with progress tracking
            embeddings = embedding_gen.generate_embeddings_batch(chunk_texts)
            
            if embeddings and len(embeddings) > 0:
                embedding_dim = len(embeddings[0])
                print(f"✅ Generated {len(embeddings)} embeddings using OpenAI API")
                print(f"   Embedding dimension: {embedding_dim}")
                print(f"   ✅✅✅ NO PYTORCH - NO SEGFAULTS!")
            else:
                raise ValueError("Embedding generation returned empty results")
                
        except Exception as e:
            print(f"❌ OpenAI embedding generation failed: {e}")
            print("   Check your OPENAI_API_KEY and API access")
            raise
        
        # 4. Create FAISS index
        dimension = len(embeddings[0])
        index = faiss.IndexFlatIP(dimension)
        
        # Normalize embeddings
        embeddings_np = np.array(embeddings)
        norms = np.linalg.norm(embeddings_np, axis=1, keepdims=True)
        normalized_embeddings = embeddings_np / norms
        
        index.add(normalized_embeddings.astype('float32'))
        
        # 5. Save index and metadata
        output_dir = project_root / "data"
        output_dir.mkdir(exist_ok=True)
        
        faiss_path = output_dir / "faiss_index.bin"
        metadata_path = output_dir / "chunk_metadata.pkl"
        
        faiss.write_index(index, str(faiss_path))
        
        chunk_metadata = [
            {
                "chunk_id": chunk.chunk_id,
                "text": chunk.text,
                "metadata": {
                    "title": chunk.metadata.title,
                    "source": chunk.metadata.source,
                    "jurisdiction": chunk.metadata.jurisdiction,
                    "document_type": chunk.metadata.document_type,
                    "section": getattr(chunk.metadata, 'section', None)
                },
                "chunk_index": chunk.chunk_index
            }
            for chunk in all_chunks
        ]
        
        with open(metadata_path, "wb") as f:
            pickle.dump(chunk_metadata, f)
        
        print(f"\n✅ Data ingestion complete!")
        print(f"📁 Files created:")
        print(f"   - FAISS index: {faiss_path}")
        print(f"   - Metadata: {metadata_path}")
        print(f"   - Total chunks: {len(chunk_metadata)}")
        print(f"   - Embedding dimension: {dimension}")
        
    finally:
        # CRITICAL FIX: Skip cleanup to prevent PyTorch segfault
        # The data is already saved, so cleanup is not critical
        # PyTorch cleanup causes segfaults, so we just exit
        # The OS will clean up memory when the process exits
        pass
        # Note: gc.collect() and del operations cause segfaults with PyTorch
        # This is a known issue - the data is already saved, so it's safe to skip cleanup

if __name__ == "__main__":
    ingest_data()