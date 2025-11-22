#!/usr/bin/env python3
# Legal Chatbot - Data Ingestion Script (Fixed Version)
# Run this to build FAISS index from legal documents
# This version handles PyTorch segfaults better

import os
import sys
import faiss
import pickle
import numpy as np
from pathlib import Path
import signal

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def signal_handler(sig, frame):
    """Handle signals gracefully"""
    print("\n⚠️ Interrupted! Saving progress...")
    sys.exit(1)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def ingest_data():
    """Ingest documents and create FAISS index"""
    print("🚀 Starting data ingestion (Fixed Version)...")
    print("=" * 60)
    
    try:
        # 1. Load documents
        data_dir = project_root / "data" / "raw"
        
        if not data_dir.exists():
            print(f"❌ Data directory not found: {data_dir}")
            return
        
        from ingestion.loaders.document_loaders import DocumentLoaderFactory
        from ingestion.chunkers.document_chunker import ChunkingStrategy, ChunkingConfig
        
        documents = []
        for file_path in sorted(data_dir.glob("*.txt")):
            print(f"📄 Loading: {file_path.name}")
            try:
                loader = DocumentLoaderFactory.get_loader(str(file_path))
                chunks = loader.load_documents(str(file_path))
                documents.extend(chunks)
                print(f"   ✅ Loaded {len(chunks)} document(s)")
            except Exception as e:
                print(f"   ⚠️ Error loading {file_path.name}: {e}")
                continue
        
        print(f"\n✅ Loaded {len(documents)} total documents")
        
        if len(documents) == 0:
            print("❌ No documents found! Add .txt files to data/raw/")
            return
        
        # 2. Chunk documents
        print("\n📝 Chunking documents...")
        chunker = ChunkingStrategy()
        config = ChunkingConfig(
            chunk_size=600,
            overlap_size=100,
            preserve_sentences=True
        )
        chunker.chunker.config = config
        
        all_chunks = []
        for doc in documents:
            try:
                chunks = chunker.chunk_document(doc, "sections")
                all_chunks.extend(chunks)
            except Exception as e:
                print(f"⚠️ Error chunking document: {e}")
                # Add as single chunk if chunking fails
                all_chunks.append(doc)
        
        print(f"✅ Created {len(all_chunks)} chunks")
        
        if len(all_chunks) == 0:
            print("❌ No chunks created!")
            return
        
        # 3. Generate embeddings with better error handling
        print("\n🧠 Generating embeddings...")
        chunk_texts = [chunk.text for chunk in all_chunks]
        
        embeddings = None
        embedding_dim = None
        
        # Try sentence-transformers first
        try:
            from retrieval.embeddings.embedding_generator import EmbeddingGenerator, EmbeddingConfig
            
            embedding_config = EmbeddingConfig(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                dimension=384,
                batch_size=16  # Smaller batch to avoid memory issues
            )
            
            print("   Attempting to use sentence-transformers...")
            embedding_gen = EmbeddingGenerator(embedding_config)
            
            if embedding_gen.model is None:
                raise RuntimeError("Model is None")
            
            # Generate embeddings in smaller batches to avoid crashes
            embeddings_list = []
            batch_size = 8  # Very small batches
            for i in range(0, len(chunk_texts), batch_size):
                batch = chunk_texts[i:i+batch_size]
                print(f"   Processing batch {i//batch_size + 1}/{(len(chunk_texts)-1)//batch_size + 1}...")
                try:
                    batch_embeddings = embedding_gen.generate_embeddings_batch(batch)
                    embeddings_list.extend(batch_embeddings)
                except Exception as e:
                    print(f"   ⚠️ Error in batch {i//batch_size + 1}: {e}")
                    # Fall back to TF-IDF for this batch
                    raise
            
            embeddings = embeddings_list
            embedding_dim = len(embeddings[0]) if embeddings else None
            print(f"✅ Generated {len(embeddings)} embeddings using sentence-transformers (dim={embedding_dim})")
            
        except Exception as e:
            print(f"⚠️ Sentence-transformers failed: {e}")
            print("🔄 Falling back to TF-IDF...")
            
            try:
                from sklearn.feature_extraction.text import TfidfVectorizer
                vectorizer = TfidfVectorizer(
                    max_features=1000,
                    stop_words='english',
                    ngram_range=(1, 2)
                )
                tfidf_matrix = vectorizer.fit_transform(chunk_texts)
                embeddings = tfidf_matrix.toarray().tolist()
                embedding_dim = len(embeddings[0]) if embeddings else None
                print(f"✅ Generated {len(embeddings)} embeddings using TF-IDF (dim={embedding_dim})")
            except Exception as e2:
                print(f"❌ TF-IDF also failed: {e2}")
                return
        
        if embeddings is None or len(embeddings) == 0:
            print("❌ No embeddings generated!")
            return
        
        # 4. Create FAISS index
        print("\n💾 Creating FAISS index...")
        dimension = len(embeddings[0])
        index = faiss.IndexFlatIP(dimension)
        
        # Normalize embeddings
        embeddings_np = np.array(embeddings, dtype=np.float32)
        norms = np.linalg.norm(embeddings_np, axis=1, keepdims=True)
        # Avoid division by zero
        norms = np.where(norms == 0, 1.0, norms)
        normalized_embeddings = embeddings_np / norms
        
        index.add(normalized_embeddings)
        print(f"✅ FAISS index created with {index.ntotal} vectors (dim={dimension})")
        
        # 5. Save index and metadata
        print("\n💾 Saving index and metadata...")
        output_dir = project_root / "data"
        output_dir.mkdir(exist_ok=True)
        
        faiss_path = output_dir / "faiss_index.bin"
        metadata_path = output_dir / "chunk_metadata.pkl"
        
        # Backup old files
        if faiss_path.exists():
            backup_path = output_dir / "faiss_index.bin.backup"
            print(f"   Backing up old index to {backup_path}")
            faiss_path.rename(backup_path)
        
        if metadata_path.exists():
            backup_path = output_dir / "chunk_metadata.pkl.backup"
            print(f"   Backing up old metadata to {backup_path}")
            metadata_path.rename(backup_path)
        
        faiss.write_index(index, str(faiss_path))
        
        chunk_metadata = []
        for chunk in all_chunks:
            try:
                chunk_metadata.append({
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
                })
            except Exception as e:
                print(f"⚠️ Error processing chunk metadata: {e}")
                continue
        
        with open(metadata_path, "wb") as f:
            pickle.dump(chunk_metadata, f)
        
        print(f"\n✅ Data ingestion complete!")
        print(f"📁 Files created:")
        print(f"   - FAISS index: {faiss_path} ({faiss_path.stat().st_size / 1024:.1f} KB)")
        print(f"   - Metadata: {metadata_path} ({metadata_path.stat().st_size / 1024:.1f} KB)")
        print(f"   - Total chunks: {len(chunk_metadata)}")
        print(f"   - Embedding dimension: {dimension}")
        
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error during ingestion: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    ingest_data()






