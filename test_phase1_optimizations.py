#!/usr/bin/env python3
"""
Test Phase 1 Optimizations for Citation Assistant

This script compares search results between:
1. OLD settings: fetch_multiplier=10, max_fetch=500, keyword_boost=0.1^n
2. NEW settings: fetch_multiplier=50, max_fetch=2000, keyword_boost=0.7^n
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from citation_assistant import CitationAssistant


def test_search_comparison(query: str, keywords: str = ""):
    """Compare old vs new search parameters"""
    print("\n" + "="*80)
    print(f"Testing query: '{query}'")
    if keywords:
        print(f"Keywords: '{keywords}'")
    print("="*80)

    # Load assistant with OLD parameters (for comparison)
    print("\n[OLD PARAMETERS]")
    print("  fetch_multiplier=10, max_fetch=500, keyword_boost=0.1^n")
    assistant_old = CitationAssistant(
        embeddings_dir="/fastpool/rag_embeddings",
        default_fetch_multiplier=10,
        default_max_fetch=500,
        default_keyword_boost=0.1  # Very aggressive (old behavior)
    )

    old_results = assistant_old.search_papers(
        query,
        n_results=10,
        boost_keywords=keywords
    )

    print(f"\nFound {len(old_results)} unique papers")
    print("\nTop 5 results:")
    for i, paper in enumerate(old_results[:5], 1):
        print(f"{i}. {paper['filename']}")
        print(f"   Similarity: {paper['similarity']:.3f} | Keywords: {paper['keyword_matches']}")
        print(f"   Excerpt: {paper['text'][:100]}...")
        print()

    # Load assistant with NEW parameters (optimized)
    print("\n[NEW PARAMETERS - OPTIMIZED]")
    print("  fetch_multiplier=50, max_fetch=2000, keyword_boost=0.7^n")
    assistant_new = CitationAssistant(
        embeddings_dir="/fastpool/rag_embeddings",
        default_fetch_multiplier=50,
        default_max_fetch=2000,
        default_keyword_boost=0.7  # Moderate (new behavior)
    )

    new_results = assistant_new.search_papers(
        query,
        n_results=10,
        boost_keywords=keywords
    )

    print(f"\nFound {len(new_results)} unique papers")
    print("\nTop 5 results:")
    for i, paper in enumerate(new_results[:5], 1):
        print(f"{i}. {paper['filename']}")
        print(f"   Similarity: {paper['similarity']:.3f} | Keywords: {paper['keyword_matches']}")
        print(f"   Excerpt: {paper['text'][:100]}...")
        print()

    # Compare results
    print("\n" + "-"*80)
    print("COMPARISON:")
    print("-"*80)

    old_files = {p['filename'] for p in old_results[:10]}
    new_files = {p['filename'] for p in new_results[:10]}

    only_in_old = old_files - new_files
    only_in_new = new_files - old_files
    in_both = old_files & new_files

    print(f"Papers in both top 10: {len(in_both)}")
    print(f"Only in OLD top 10: {len(only_in_old)}")
    print(f"Only in NEW top 10: {len(only_in_new)}")

    if only_in_new:
        print(f"\nNew papers found with optimizations:")
        for filename in list(only_in_new)[:3]:
            print(f"  • {filename}")

    # Check for diversity
    old_top5_files = [p['filename'] for p in old_results[:5]]
    new_top5_files = [p['filename'] for p in new_results[:5]]

    print(f"\nResult diversity (unique papers in top 5):")
    print(f"  OLD: {len(set(old_top5_files))} unique papers")
    print(f"  NEW: {len(set(new_top5_files))} unique papers")


def main():
    print("\n" + "#"*80)
    print("# Phase 1 Optimization Test Suite")
    print("#"*80)

    # Test 1: General biomedical query
    test_search_comparison(
        query="microbiome dysbiosis and inflammatory bowel disease",
        keywords=""
    )

    # Test 2: Specific keyword query
    test_search_comparison(
        query="golgi apparatus trafficking vesicle transport",
        keywords="golgicide, brefeldin"
    )

    # Test 3: Methodological query
    test_search_comparison(
        query="16S rRNA sequencing analysis methods microbiome",
        keywords=""
    )

    print("\n" + "#"*80)
    print("# Test Complete!")
    print("#"*80)
    print("\nExpected improvements with NEW parameters:")
    print("  1. More diverse results (better coverage of collection)")
    print("  2. Better balance between semantic similarity and keyword matches")
    print("  3. More relevant papers that were previously missed")
    print("  4. Less domination by keyword-heavy but contextually wrong papers")


if __name__ == "__main__":
    main()
