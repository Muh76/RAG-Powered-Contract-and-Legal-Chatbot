#!/usr/bin/env python3
"""
Standalone test for RAGService search (no FastAPI, no auth).
- Instantiates RAGService
- Runs search for "Sale of Goods Act 1979 faulty goods"
- Prints number of results and top 3 result texts (truncated).

Does NOT modify production code.
"""
import sys
from pathlib import Path

# Ensure project root is on path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from app.services.rag_service import RAGService

QUERY = "Sale of Goods Act 1979 faulty goods"
TOP_N = 3
TRUNCATE = 200


def main():
    print("RAGService standalone search test")
    print("=" * 60)
    print(f"Query: {QUERY!r}")
    print()
    rag = RAGService(use_hybrid=True)
    results = rag.search(query=QUERY, top_k=10)
    print()
    print(f"Number of results: {len(results)}")
    print()
    print(f"Top {TOP_N} result texts (truncated to {TRUNCATE} chars):")
    print("-" * 60)
    for i, r in enumerate(results[:TOP_N], 1):
        text = (r.get("text") or "").strip()
        if len(text) > TRUNCATE:
            text = text[:TRUNCATE] + "..."
        print(f"  [{i}] {text}")
    print("-" * 60)


if __name__ == "__main__":
    main()
