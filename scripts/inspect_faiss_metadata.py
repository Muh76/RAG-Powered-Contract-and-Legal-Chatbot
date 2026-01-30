#!/usr/bin/env python3
"""
Inspect FAISS metadata (chunk_metadata) the same way RAG service loads it.
Prints total chunks and first 10 chunk texts (truncated to 200 chars).
Does NOT modify any code or data.
"""
import pickle
from pathlib import Path

# Same paths as app/services/rag_service.py and app/api/routes/chat.py
project_root = Path(__file__).resolve().parent.parent
possible_pkl = [
    project_root / "data" / "indices" / "faiss_index.pkl",
    project_root / "data" / "indices" / "chunk_metadata.pkl",
    project_root / "data" / "chunk_metadata.pkl",
    project_root / "notebooks" / "phase1" / "data" / "chunk_metadata.pkl",
    Path("data/indices/faiss_index.pkl"),
    Path("data/indices/chunk_metadata.pkl"),
    Path("data/chunk_metadata.pkl"),
]

def load_chunk_metadata():
    """Load chunk_metadata from combined faiss_index.pkl or standalone chunk_metadata.pkl."""
    for path in possible_pkl:
        p = path if path.is_absolute() else (project_root / path)
        if not p.exists():
            continue
        try:
            with open(p, "rb") as f:
                data = pickle.load(f)
            if isinstance(data, dict):
                chunks = data.get("chunk_metadata", [])
                if chunks is not None:
                    return chunks, str(p)
            else:
                # Standalone chunk_metadata.pkl is a list
                if isinstance(data, list):
                    return data, str(p)
        except Exception as e:
            print(f"   Skip {p}: {e}")
            continue
    return None, None

def main():
    print("FAISS metadata inspection (same load logic as RAG)")
    print("=" * 60)
    chunks, source = load_chunk_metadata()
    if chunks is None:
        print("No FAISS metadata file found. Tried:")
        for p in possible_pkl:
            q = p if p.is_absolute() else (project_root / p)
            print(f"   {q}")
        return
    print(f"Source: {source}")
    print(f"Total number of chunks: {len(chunks)}")
    print()
    print("First 10 chunk texts (truncated to 200 chars each):")
    print("-" * 60)
    for i, chunk in enumerate(chunks[:10]):
        if isinstance(chunk, dict):
            text = chunk.get("text", chunk.get("content", str(chunk)[:200]))
        else:
            text = str(chunk)
        text = (text or "")[:200]
        if len((chunk.get("text") or chunk.get("content") or "")) > 200:
            text += "..."
        print(f"  [{i+1}] {text}")
    print("-" * 60)
    # Quick scan for Sale of Goods Act 1979
    sale_of_goods_count = 0
    for chunk in chunks:
        t = (chunk.get("text") or chunk.get("metadata", {}).get("title") or "") if isinstance(chunk, dict) else str(chunk)
        if "Sale of Goods" in t or "sale of goods" in t or "1979" in t:
            sale_of_goods_count += 1
    print(f"Chunks mentioning 'Sale of Goods' or '1979': {sale_of_goods_count}")

if __name__ == "__main__":
    main()
