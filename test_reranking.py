#!/usr/bin/env python3
"""
Test script for cross-encoder re-ranking functionality
Compares search results with and without re-ranking
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.citation_assistant import CitationAssistant


def test_reranking():
    """Test cross-encoder re-ranking vs standard search"""

    print("=" * 80)
    print("CROSS-ENCODER RE-RANKING TEST")
    print("=" * 80)
    print()

    # Test query
    test_query = "antimicrobial resistance in gut microbiome"
    n_results = 5

    print(f"Test query: '{test_query}'")
    print(f"Requesting {n_results} results")
    print()

    # Test 1: Standard search (no re-ranking)
    print("-" * 80)
    print("TEST 1: Standard Vector Search (Baseline)")
    print("-" * 80)

    assistant_baseline = CitationAssistant(
        embeddings_dir="/fastpool/rag_embeddings",
        enable_reranking=False
    )

    papers_baseline = assistant_baseline.search_papers(test_query, n_results=n_results)

    print(f"\nTop {n_results} papers (baseline):")
    for i, paper in enumerate(papers_baseline, 1):
        print(f"\n{i}. {paper['filename']}")
        print(f"   Similarity: {paper['similarity']:.4f}")
        print(f"   Distance: {paper['distance']:.4f}")
        print(f"   Excerpt: {paper['text'][:150]}...")

    # Test 2: Search with re-ranking
    print("\n" + "=" * 80)
    print("TEST 2: Vector Search + Cross-Encoder Re-Ranking")
    print("=" * 80)

    assistant_rerank = CitationAssistant(
        embeddings_dir="/fastpool/rag_embeddings",
        enable_reranking=True
    )

    papers_rerank = assistant_rerank.search_papers(test_query, n_results=n_results)

    print(f"\nTop {n_results} papers (with re-ranking):")
    for i, paper in enumerate(papers_rerank, 1):
        print(f"\n{i}. {paper['filename']}")
        print(f"   Re-rank Score: {paper.get('rerank_score', 'N/A'):.4f}")
        print(f"   Original Similarity: {paper['similarity']:.4f}")
        print(f"   Original Distance: {paper['distance']:.4f}")
        print(f"   Excerpt: {paper['text'][:150]}...")

    # Test 3: Compare results
    print("\n" + "=" * 80)
    print("COMPARISON: Ranking Changes")
    print("=" * 80)

    baseline_files = [p['filename'] for p in papers_baseline]
    rerank_files = [p['filename'] for p in papers_rerank]

    print("\nBaseline order:", baseline_files)
    print("Re-ranked order:", rerank_files)

    # Count how many positions changed
    position_changes = 0
    for i, filename in enumerate(baseline_files):
        if filename not in rerank_files:
            position_changes += 1
            print(f"  - '{filename}' dropped out of top {n_results}")
        elif rerank_files.index(filename) != i:
            old_pos = i + 1
            new_pos = rerank_files.index(filename) + 1
            print(f"  - '{filename}' moved from position {old_pos} to {new_pos}")
            position_changes += 1

    # Check for new entries
    for filename in rerank_files:
        if filename not in baseline_files:
            position_changes += 1
            print(f"  - '{filename}' entered top {n_results} after re-ranking")

    if position_changes == 0:
        print("\n  No changes in ranking (re-ranker agreed with baseline)")
    else:
        print(f"\n  Total ranking changes: {position_changes}")

    print("\n" + "=" * 80)
    print("TEST COMPLETED SUCCESSFULLY")
    print("=" * 80)
    print("\nKey Takeaways:")
    print("- Re-ranking uses cross-encoder to see query + document together")
    print("- Should improve precision by catching semantic nuances")
    print("- Expected improvement: +10-15% precision on relevant results")
    print()


if __name__ == "__main__":
    test_reranking()
