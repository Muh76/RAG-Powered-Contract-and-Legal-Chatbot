#!/usr/bin/env python3
# Legal Chatbot - Data Ingestion Script
# Run this to build or update FAISS index from legal documents.
# Supports incremental ingestion: existing index is preserved; only new chunks are embedded and added.

import os
import sys
import logging
import faiss
import pickle
import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ingestion.loaders.document_loaders import DocumentLoaderFactory
from ingestion.chunkers.document_chunker import ChunkingStrategy, ChunkingConfig
from retrieval.embeddings.openai_embedding_generator import OpenAIEmbeddingGenerator, OpenAIEmbeddingConfig
from app.core.config import settings, _validate_embedding_config

# Logging: clear ingestion steps (console + optional file)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("ingest_data")

# Output paths (must match RAG service lookup order where applicable)
DATA_DIR = project_root / "data"
INDICES_DIR = project_root / "data" / "indices"
FAISS_BIN_PATH = DATA_DIR / "faiss_index.bin"
METADATA_PKL_PATH = DATA_DIR / "chunk_metadata.pkl"
FAISS_COMBINED_PKL_PATH = INDICES_DIR / "faiss_index.pkl"


def _log_step(step: str, detail: str) -> None:
    """Log an ingestion step clearly."""
    logger.info("[INGEST] %s — %s", step, detail)
    print(f"   {step}: {detail}")


def load_existing_index() -> Tuple[Optional[faiss.Index], List[dict]]:
    """
    Load existing FAISS index and chunk metadata if present.
    Does not delete or modify any files. Returns (index, metadata_list) or (None, []).
    Idempotent: safe to call when no index exists.
    """
    existing_metadata: List[dict] = []
    existing_index: Optional[faiss.Index] = None

    # Prefer combined pkl (same format we write for RAG)
    if FAISS_COMBINED_PKL_PATH.exists():
        try:
            with open(FAISS_COMBINED_PKL_PATH, "rb") as f:
                data = pickle.load(f)
            existing_index = data.get("faiss_index")
            existing_metadata = data.get("chunk_metadata") or []
            if existing_index is not None:
                _log_step("Load existing index", f"Loaded from {FAISS_COMBINED_PKL_PATH}: {existing_index.ntotal} vectors, {len(existing_metadata)} chunks")
                return existing_index, existing_metadata
        except Exception as e:
            logger.warning("Could not load combined pkl %s: %s", FAISS_COMBINED_PKL_PATH, e)

    # Fallback: separate .bin + .pkl
    if FAISS_BIN_PATH.exists() and METADATA_PKL_PATH.exists():
        try:
            existing_index = faiss.read_index(str(FAISS_BIN_PATH))
            with open(METADATA_PKL_PATH, "rb") as f:
                existing_metadata = pickle.load(f)
            if existing_metadata is None:
                existing_metadata = []
            _log_step("Load existing index", f"Loaded from {FAISS_BIN_PATH}: {existing_index.ntotal} vectors, {len(existing_metadata)} chunks")
            return existing_index, existing_metadata
        except Exception as e:
            logger.warning("Could not load existing index from %s / %s: %s", FAISS_BIN_PATH, METADATA_PKL_PATH, e)

    _log_step("Load existing index", "No existing index found; will create new index.")
    return None, []


def save_index_and_metadata(
    index: faiss.Index,
    chunk_metadata: List[dict],
) -> None:
    """Save FAISS index and metadata to disk. Overwrites only these files; does not delete other corpora."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    INDICES_DIR.mkdir(parents=True, exist_ok=True)

    # Save separate files (backward compatible)
    faiss.write_index(index, str(FAISS_BIN_PATH))
    _log_step("Save", f"Wrote FAISS index to {FAISS_BIN_PATH} ({index.ntotal} vectors)")

    with open(METADATA_PKL_PATH, "wb") as f:
        pickle.dump(chunk_metadata, f)
    _log_step("Save", f"Wrote chunk metadata to {METADATA_PKL_PATH} ({len(chunk_metadata)} chunks)")

    # Save combined pkl for RAG service (preferred path)
    with open(FAISS_COMBINED_PKL_PATH, "wb") as f:
        pickle.dump({"faiss_index": index, "chunk_metadata": chunk_metadata}, f)
    _log_step("Save", f"Wrote combined index to {FAISS_COMBINED_PKL_PATH}")


def chunk_to_metadata_item(chunk) -> dict:
    """Build a single chunk metadata dict for storage."""
    return {
        "chunk_id": chunk.chunk_id,
        "text": chunk.text,
        "metadata": {
            "title": chunk.metadata.title,
            "source": chunk.metadata.source,
            "jurisdiction": chunk.metadata.jurisdiction,
            "document_type": chunk.metadata.document_type,
            "section": getattr(chunk.metadata, "section", None),
        },
        "chunk_index": chunk.chunk_index,
    }


def ingest_data() -> None:
    """
    Ingest documents and create or update FAISS index.
    - Existing embeddings are never deleted; new chunks are appended.
    - Deduplication by chunk_id makes re-runs idempotent (no duplicate chunks).
    - Logs each ingestion step clearly.
    """
    logger.info("Starting data ingestion (incremental; existing corpora preserved).")
    _validate_embedding_config()
    print("=" * 60)
    print("Data ingestion — incremental mode")
    print("=" * 60)

    MAX_CHUNKS_TO_INDEX = 5000
    _log_step("Config", f"Portfolio limit: {MAX_CHUNKS_TO_INDEX:,} chunks (UK legislation prioritised)")

    existing_index, existing_metadata = load_existing_index()
    existing_chunk_ids = {m.get("chunk_id") for m in existing_metadata if m.get("chunk_id")}
    _log_step("Deduplication", f"Existing chunk IDs in index: {len(existing_chunk_ids):,}")

    embedding_gen = None
    try:
        # 1. Load documents
        _log_step("Load documents", "Scanning data/raw, data/uk_legislation, data/cuad/data")
        data_dir = project_root / "data" / "raw"
        if not data_dir.exists():
            data_dir.mkdir(parents=True, exist_ok=True)
            sample_file = data_dir / "uk_legal_sample.txt"
            if not sample_file.exists():
                with open(sample_file, "w") as f:
                    f.write("Sale of Goods Act 1979\nSection 12 - Implied condition as to title.\n")
                _log_step("Load documents", f"Created sample file: {sample_file}")

        documents: List = []
        for file_path in data_dir.glob("*.txt"):
            try:
                loader = DocumentLoaderFactory.get_loader(str(file_path))
                chunks = loader.load_documents(str(file_path))
                documents.extend(chunks)
                logger.info("Loaded %s: %d chunks", file_path.name, len(chunks))
            except Exception as e:
                logger.warning("Error loading %s: %s", file_path.name, e)

        uk_legislation_dir = project_root / "data" / "uk_legislation"
        if uk_legislation_dir.exists():
            for file_path in uk_legislation_dir.glob("*.json"):
                try:
                    loader = DocumentLoaderFactory.get_loader(str(file_path))
                    chunks = loader.load_documents(str(file_path))
                    documents.extend(chunks)
                    logger.info("Loaded %s: %d chunks", file_path.name, len(chunks))
                except Exception as e:
                    logger.warning("Error loading %s: %s", file_path.name, e)

        cuad_dir = project_root / "data" / "cuad" / "data"
        if cuad_dir.exists():
            for file_path in sorted(cuad_dir.glob("*.parquet")):
                try:
                    loader = DocumentLoaderFactory.get_loader(str(file_path))
                    chunks = loader.load_documents(str(file_path))
                    documents.extend(chunks)
                    logger.info("Loaded %s: %d chunks", file_path.name, len(chunks))
                except Exception as e:
                    logger.warning("Error loading %s: %s", file_path.name, e)

        _log_step("Load documents", f"Total documents loaded: {len(documents):,}")
        if len(documents) == 0:
            _log_step("Load documents", "No documents found. Add files to data/raw or data/uk_legislation.")
            return

        # 2. Chunk
        chunker = ChunkingStrategy()
        chunker.chunker.config = ChunkingConfig(
            chunk_size=600,
            overlap_size=100,
            preserve_sentences=True,
        )
        all_chunks: List = []
        for doc in documents:
            chunks = chunker.chunk_document(doc, "sections")
            all_chunks.extend(chunks)
        _log_step("Chunk", f"Created {len(all_chunks):,} chunks from {len(documents):,} documents")

        # Portfolio limit: prioritise UK legislation
        if len(all_chunks) > MAX_CHUNKS_TO_INDEX:
            uk_legislation_chunks = []
            cuad_chunks = []
            other_chunks = []
            for chunk in all_chunks:
                source = getattr(chunk.metadata, "source", "").lower()
                doc_type = getattr(chunk.metadata, "document_type", "").lower()
                if "legislation" in doc_type or "act" in source or "regulation" in source:
                    uk_legislation_chunks.append(chunk)
                elif "cuad" in chunk.chunk_id.lower() or "contract" in source.lower():
                    cuad_chunks.append(chunk)
                else:
                    other_chunks.append(chunk)
            selected = uk_legislation_chunks.copy()
            remaining = MAX_CHUNKS_TO_INDEX - len(selected)
            if cuad_chunks and remaining > 0:
                selected.extend(cuad_chunks[: min(remaining, len(cuad_chunks))])
                remaining = MAX_CHUNKS_TO_INDEX - len(selected)
            if other_chunks and remaining > 0:
                selected.extend(other_chunks[: min(remaining, len(other_chunks))])
            all_chunks = selected
            _log_step("Chunk", f"Applied portfolio limit: {len(all_chunks):,} chunks selected")

        # 3. Deduplicate: only index chunks not already in existing index (idempotency)
        new_chunks = [c for c in all_chunks if c.chunk_id not in existing_chunk_ids]
        _log_step("Deduplication", f"New chunks to index: {len(new_chunks):,} (skipping {len(all_chunks) - len(new_chunks):,} already in index)")

        if len(new_chunks) == 0:
            _log_step("Idempotency", "No new chunks to index. Exiting without overwriting existing index.")
            print("Ingestion finished (no changes).")
            return

        # 4. Embed only new chunks
        api_key = os.getenv("OPENAI_API_KEY") or settings.OPENAI_API_KEY
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set. Set it in environment or .env.")
        embedding_config = OpenAIEmbeddingConfig(
            api_key=api_key,
            model=settings.OPENAI_EMBEDDING_MODEL,
            dimension=settings.EMBEDDING_DIMENSION,
            batch_size=50,
            max_retries=5,
            timeout=60,
            requests_per_minute=30,
        )
        embedding_gen = OpenAIEmbeddingGenerator(embedding_config)
        chunk_texts = [c.text for c in new_chunks]
        _log_step("Embed", f"Generating embeddings for {len(chunk_texts):,} new chunks (OpenAI)")
        embeddings = embedding_gen.generate_embeddings_batch(chunk_texts)
        if not embeddings or len(embeddings) != len(new_chunks):
            raise ValueError(f"Embedding count mismatch: got {len(embeddings) if embeddings else 0}, expected {len(new_chunks)}")
        dimension = len(embeddings[0])
        _log_step("Embed", f"Generated {len(embeddings)} embeddings (dim={dimension})")

        # 5. Merge with existing index or create new one
        if existing_index is not None:
            if existing_index.d != dimension:
                raise ValueError(
                    f"Dimension mismatch: existing index has d={existing_index.d}, new embeddings have dim={dimension}. "
                    "Use the same embedding model or re-run full ingestion."
                )
            embeddings_np = np.array(embeddings, dtype=np.float32)
            norms = np.linalg.norm(embeddings_np, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            normalized = embeddings_np / norms
            existing_index.add(normalized)
            index = existing_index
            new_metadata = [chunk_to_metadata_item(c) for c in new_chunks]
            combined_metadata = existing_metadata + new_metadata
            _log_step("Merge", f"Appended {len(new_chunks):,} vectors to existing index; total vectors: {index.ntotal}, total metadata: {len(combined_metadata):,}")
        else:
            index = faiss.IndexFlatIP(dimension)
            embeddings_np = np.array(embeddings, dtype=np.float32)
            norms = np.linalg.norm(embeddings_np, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            normalized = embeddings_np / norms
            index.add(normalized)
            combined_metadata = [chunk_to_metadata_item(c) for c in new_chunks]
            _log_step("Merge", f"Created new index with {index.ntotal} vectors and {len(combined_metadata):,} metadata entries")

        # 6. Save (overwrites only index/metadata files; does not delete other data)
        save_index_and_metadata(index, combined_metadata)
        logger.info("Ingestion complete. Total chunks in index: %s", len(combined_metadata))
        print("Ingestion complete.")

    except Exception as e:
        logger.exception("Ingestion failed: %s", e)
        raise
    finally:
        pass  # No aggressive cleanup (avoids segfaults with some embedders)


if __name__ == "__main__":
    ingest_data()
