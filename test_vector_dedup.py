#!/usr/bin/env python3
"""
Test script to verify vector-based duplicate detection
Shows distance comparisons for papers that might be duplicates
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.citation_assistant import CitationAssistant


def test_vector_deduplication():
    """Test vector-based duplicate detection with detailed output"""

    print("=" * 80)
    print("VECTOR-BASED DUPLICATE DETECTION TEST")
    print("=" * 80)
    print()

    test_query = "antimicrobial resistance in gut microbiome"
    n_results = 10

    print(f"Test query: '{test_query}'")
    print(f"Requesting {n_results} results")
    print()

    # Test with vector-based deduplication
    assistant = CitationAssistant(
        embeddings_dir="/fastpool/rag_embeddings",
        enable_reranking=False
    )

    papers = assistant.search_papers(test_query, n_results=n_results)

    print(f"\nTop {n_results} papers (with vector deduplication):")
    print("-" * 80)

    # Show papers with their distances
    for i, paper in enumerate(papers, 1):
        print(f"\n{i}. {paper['filename']}")
        print(f"   Distance: {paper['distance']:.4f}")
        print(f"   Similarity: {paper['similarity']:.4f}")

    # Check for potential duplicates by comparing distances
    print("\n" + "=" * 80)
    print("DISTANCE ANALYSIS (looking for potential duplicates)")
    print("=" * 80)

    threshold = 0.1
    print(f"\nUsing threshold: {threshold} (papers within this distance are considered duplicates)")
    print()

    potential_duplicates = []
    for i in range(len(papers)):
        for j in range(i + 1, len(papers)):
            dist_diff = abs(papers[i]['distance'] - papers[j]['distance'])
            if dist_diff < threshold * 2:  # Show pairs that are close
                potential_duplicates.append((i, j, dist_diff))
                status = "DUPLICATE (filtered)" if dist_diff < threshold else "Close (kept)"
                print(f"Papers {i+1} and {j+1}:")
                print(f"  '{papers[i]['filename'][:50]}...'")
                print(f"  '{papers[j]['filename'][:50]}...'")
                print(f"  Distance difference: {dist_diff:.4f} - {status}")
                print()

    if not potential_duplicates:
        print("✓ No potential duplicates found - deduplication working well!")

    # Show statistics
    print("=" * 80)
    print("STATISTICS")
    print("=" * 80)
    print(f"Total unique papers returned: {len(papers)}")
    print(f"Distance range: {papers[-1]['distance']:.4f} - {papers[0]['distance']:.4f}")
    print(f"Similarity range: {papers[0]['similarity']:.4f} - {papers[-1]['similarity']:.4f}")

    # Calculate average distance between consecutive papers
    if len(papers) > 1:
        dist_diffs = [papers[i+1]['distance'] - papers[i]['distance']
                      for i in range(len(papers) - 1)]
        avg_diff = sum(dist_diffs) / len(dist_diffs)
        print(f"Average distance between consecutive results: {avg_diff:.4f}")
        print(f"Duplicate threshold: {threshold:.4f}")
        print()
        if avg_diff > threshold:
            print("✓ Good diversity: papers are well-separated")
        else:
            print("⚠ Low diversity: consider increasing n_results or adjusting threshold")

    print()


if __name__ == "__main__":
    test_vector_deduplication()
