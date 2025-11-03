#!/usr/bin/env python3
"""
Build BM25 index from existing ChromaDB collection for hybrid search
This only needs to be run once, or when the collection is updated
"""

import sys
from pathlib import Path
import pickle
from collections import defaultdict

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import chromadb
from chromadb.config import Settings
from rank_bm25 import BM25Okapi


def build_bm25_index(embeddings_dir: str = "/fastpool/rag_embeddings"):
    """Build BM25 index from ChromaDB collection"""

    print("=" * 80)
    print("BUILDING BM25 INDEX FOR HYBRID SEARCH")
    print("=" * 80)
    print()

    embeddings_path = Path(embeddings_dir)

    # Load ChromaDB collection
    print(f"Loading ChromaDB collection from: {embeddings_path}")
    client = chromadb.PersistentClient(
        path=str(embeddings_path),
        settings=Settings(anonymized_telemetry=False)
    )

    try:
        collection = client.get_collection(name="research_papers")
        print(f"✓ Loaded collection with {collection.count()} documents")
    except Exception as e:
        print(f"✗ Error: Could not load collection: {e}")
        return False

    # Get all documents from collection
    print("\nFetching all documents from collection...")
    # ChromaDB has a limit, so fetch in batches
    batch_size = 5000
    offset = 0
    all_docs = []
    all_metadata = []

    while True:
        results = collection.get(
            limit=batch_size,
            offset=offset,
            include=["documents", "metadatas"]
        )

        if not results['documents']:
            break

        all_docs.extend(results['documents'])
        all_metadata.extend(results['metadatas'])
        offset += batch_size

        print(f"  Fetched {len(all_docs)} documents so far...")

    print(f"✓ Total documents fetched: {len(all_docs)}")

    # Group documents by filename (same paper, different chunks)
    print("\nGrouping documents by paper...")
    papers = defaultdict(list)
    for doc, meta in zip(all_docs, all_metadata):
        filename = meta['filename']
        papers[filename].append(doc)

    print(f"✓ Found {len(papers)} unique papers")

    # Create corpus: concatenate all chunks for each paper
    print("\nBuilding BM25 corpus...")
    corpus = []
    doc_map = []  # Maps index position to filename

    for filename, chunks in papers.items():
        # Concatenate all chunks for this paper
        full_text = " ".join(chunks)

        # Tokenize for BM25 (simple lowercase + split)
        tokenized = full_text.lower().split()

        corpus.append(tokenized)
        doc_map.append(filename)

    print(f"✓ Created corpus with {len(corpus)} documents")

    # Build BM25 index
    print("\nBuilding BM25 index...")
    bm25 = BM25Okapi(corpus)
    print("✓ BM25 index built successfully")

    # Save index and document mapping
    output_path = embeddings_path / "bm25_index.pkl"
    print(f"\nSaving BM25 index to: {output_path}")

    with open(output_path, 'wb') as f:
        pickle.dump({
            'index': bm25,
            'doc_map': doc_map
        }, f)

    print(f"✓ Saved BM25 index ({output_path.stat().st_size / (1024**2):.1f} MB)")

    # Test the index
    print("\n" + "=" * 80)
    print("TESTING BM25 INDEX")
    print("=" * 80)

    test_query = "Clostridioides difficile levofloxacin"
    print(f"\nTest query: '{test_query}'")

    query_tokens = test_query.lower().split()
    scores = bm25.get_scores(query_tokens)

    # Get top 5 results
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:5]

    print("\nTop 5 BM25 results:")
    for rank, idx in enumerate(top_indices, 1):
        print(f"  {rank}. {doc_map[idx][:60]}... (score: {scores[idx]:.2f})")

    print("\n" + "=" * 80)
    print("BM25 INDEX BUILD COMPLETE")
    print("=" * 80)
    print("\nThe hybrid_search() method is now available in CitationAssistant!")
    print()

    return True


if __name__ == "__main__":
    success = build_bm25_index()
    sys.exit(0 if success else 1)
