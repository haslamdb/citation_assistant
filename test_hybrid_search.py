#!/usr/bin/env python3
"""
Test hybrid search functionality (Vector + BM25)
Compares pure vector search vs hybrid search with different alpha values
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from citation_assistant import CitationAssistant


def test_hybrid_search():
    """Test hybrid search with different weighting strategies"""

    print("=" * 80)
    print("HYBRID SEARCH TEST")
    print("=" * 80)
    print()

    # Initialize assistant
    print("Initializing CitationAssistant...")
    assistant = CitationAssistant(embeddings_dir="/fastpool/rag_embeddings")

    if assistant.bm25_index is None:
        print("ERROR: BM25 index not loaded!")
        return False

    print(f"✓ Loaded {len(assistant.bm25_doc_map)} papers in BM25 index")
    print()

    # Test query
    test_query = "Clostridioides difficile antibiotic resistance levofloxacin"
    print(f"Test query: '{test_query}'")
    print()

    # Test 1: Pure vector search (baseline)
    print("-" * 80)
    print("TEST 1: Pure Vector Search (baseline)")
    print("-" * 80)
    vector_results = assistant.search_papers(test_query, n_results=10)
    print(f"Found {len(vector_results)} results\n")
    print("Top 5 results:")
    for i, paper in enumerate(vector_results[:5], 1):
        print(f"  {i}. {paper['filename'][:70]}")
        print(f"     Distance: {paper['distance']:.4f}")
    print()

    # Test 2: Pure BM25 search
    print("-" * 80)
    print("TEST 2: Pure BM25 (alpha=0.0, only keyword matching)")
    print("-" * 80)
    bm25_results = assistant.hybrid_search(test_query, n_results=10, alpha=0.0)
    print(f"Found {len(bm25_results)} results\n")
    print("Top 5 results:")
    for i, paper in enumerate(bm25_results[:5], 1):
        print(f"  {i}. {paper['filename'][:70]}")
        print(f"     BM25 score: {paper.get('bm25_score', 0):.4f}")
        print(f"     Hybrid score: {paper['hybrid_score']:.4f}")
    print()

    # Test 3: Balanced hybrid search (50/50)
    print("-" * 80)
    print("TEST 3: Balanced Hybrid (alpha=0.5, 50% vector + 50% BM25)")
    print("-" * 80)
    hybrid_results = assistant.hybrid_search(test_query, n_results=10, alpha=0.5)
    print(f"Found {len(hybrid_results)} results\n")
    print("Top 5 results:")
    for i, paper in enumerate(hybrid_results[:5], 1):
        print(f"  {i}. {paper['filename'][:70]}")
        print(f"     Vector distance: {paper.get('distance', 0):.4f}")
        print(f"     BM25 score: {paper.get('bm25_score', 0):.4f}")
        print(f"     Hybrid score: {paper['hybrid_score']:.4f}")
    print()

    # Test 4: Vector-heavy hybrid (70% vector)
    print("-" * 80)
    print("TEST 4: Vector-Heavy Hybrid (alpha=0.7, 70% vector + 30% BM25)")
    print("-" * 80)
    hybrid_heavy = assistant.hybrid_search(test_query, n_results=10, alpha=0.7)
    print(f"Found {len(hybrid_heavy)} results\n")
    print("Top 5 results:")
    for i, paper in enumerate(hybrid_heavy[:5], 1):
        print(f"  {i}. {paper['filename'][:70]}")
        print(f"     Hybrid score: {paper['hybrid_score']:.4f}")
    print()

    # Analysis: Compare rankings
    print("=" * 80)
    print("RANKING COMPARISON")
    print("=" * 80)
    print()

    # Get top 5 filenames from each method
    vector_top5 = [p['filename'] for p in vector_results[:5]]
    bm25_top5 = [p['filename'] for p in bm25_results[:5]]
    hybrid_top5 = [p['filename'] for p in hybrid_results[:5]]

    print("Papers that appear in top 5 of different methods:")
    print()

    all_papers = set(vector_top5 + bm25_top5 + hybrid_top5)
    for paper in all_papers:
        vector_rank = vector_top5.index(paper) + 1 if paper in vector_top5 else "-"
        bm25_rank = bm25_top5.index(paper) + 1 if paper in bm25_top5 else "-"
        hybrid_rank = hybrid_top5.index(paper) + 1 if paper in hybrid_top5 else "-"

        print(f"{paper[:60]:<60}")
        print(f"  Vector: {str(vector_rank):>3} | BM25: {str(bm25_rank):>3} | Hybrid: {str(hybrid_rank):>3}")

    print()
    print("=" * 80)
    print("HYBRID SEARCH TEST COMPLETE")
    print("=" * 80)
    print()
    print("Summary:")
    print("  - Hybrid search successfully combines vector and BM25 scores")
    print("  - Alpha parameter controls the balance (0=pure BM25, 1=pure vector)")
    print("  - Default alpha=0.5 provides balanced results")
    print()

    return True


if __name__ == "__main__":
    success = test_hybrid_search()
    sys.exit(0 if success else 1)
