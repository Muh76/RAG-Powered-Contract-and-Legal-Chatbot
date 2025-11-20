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
from retrieval.embeddings.embedding_generator import EmbeddingGenerator, EmbeddingConfig

def ingest_data():
    """Ingest documents and create FAISS index"""
    print("🚀 Starting data ingestion...")
    print("=" * 60)
    
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
        
        print(f"✅ Created {len(all_chunks)} chunks")
        
        # 3. Generate embeddings
        print("🧠 Generating embeddings...")
        embedding_config = EmbeddingConfig(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            dimension=384,
            batch_size=32
        )
        
        try:
            embedding_gen = EmbeddingGenerator(embedding_config)
            chunk_texts = [chunk.text for chunk in all_chunks]
            embeddings = embedding_gen.generate_embeddings_batch(chunk_texts)
            print(f"✅ Generated {len(embeddings)} embeddings using sentence-transformers")
        except Exception as e:
            print(f"⚠️ Sentence-transformers failed: {e}")
            print("🔄 Falling back to TF-IDF...")
            from sklearn.feature_extraction.text import TfidfVectorizer
            vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words='english',
                ngram_range=(1, 2)
            )
            chunk_texts = [chunk.text for chunk in all_chunks]
            tfidf_matrix = vectorizer.fit_transform(chunk_texts)
            embeddings = tfidf_matrix.toarray().tolist()
            embedding_gen = vectorizer  # Store for cleanup
            print(f"✅ Generated {len(embeddings)} embeddings using TF-IDF")
        
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