#!/usr/bin/env python3
"""
Test script for recency boosting feature
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from citation_assistant import CitationAssistant

def test_recency_boost():
    """Test that recency boosting works correctly"""
    
    print("="*80)
    print("TESTING RECENCY BOOSTING FEATURE")
    print("="*80)
    
    # Initialize the assistant
    assistant = CitationAssistant(
        embeddings_dir="/fastpool/rag_embeddings",
        enable_reranking=False  # Disable re-ranking for clearer recency testing
    )
    
    # Test query
    query = "microbiome analysis"
    
    print(f"\nTest query: '{query}'")
    print("-"*40)
    
    # Test 1: Search WITHOUT recency boost
    print("\n1. Search WITHOUT recency boost:")
    papers_no_boost = assistant.search_papers(
        query,
        n_results=10,
        boost_recency=False,
        boost_haslam=False,  # Disable other boosts for clarity
        boost_keywords=""
    )
    
    print("\nTop 5 results (no recency boost):")
    for i, paper in enumerate(papers_no_boost[:5], 1):
        year = paper.get('publication_year', 'Unknown')
        print(f"  {i}. {paper['filename'][:50]}...")
        print(f"     Year: {year}, Similarity: {paper['similarity']:.3f}")
    
    # Test 2: Search WITH moderate recency boost
    print("\n2. Search WITH moderate recency boost (weight=0.15):")
    papers_moderate = assistant.search_papers(
        query,
        n_results=10,
        boost_recency=True,
        recency_weight=0.15,
        boost_haslam=False,
        boost_keywords=""
    )
    
    print("\nTop 5 results (moderate recency boost):")
    for i, paper in enumerate(papers_moderate[:5], 1):
        year = paper.get('publication_year', 'Unknown')
        years_old = paper.get('years_old', 'N/A')
        boosted = paper.get('recency_boost_applied', False)
        print(f"  {i}. {paper['filename'][:50]}...")
        print(f"     Year: {year}, Years old: {years_old}, Boosted: {boosted}, Similarity: {paper['similarity']:.3f}")
    
    # Test 3: Search WITH aggressive recency boost
    print("\n3. Search WITH aggressive recency boost (weight=0.25):")
    papers_aggressive = assistant.search_papers(
        query,
        n_results=10,
        boost_recency=True,
        recency_weight=0.25,
        boost_haslam=False,
        boost_keywords=""
    )
    
    print("\nTop 5 results (aggressive recency boost):")
    for i, paper in enumerate(papers_aggressive[:5], 1):
        year = paper.get('publication_year', 'Unknown')
        years_old = paper.get('years_old', 'N/A')
        boosted = paper.get('recency_boost_applied', False)
        print(f"  {i}. {paper['filename'][:50]}...")
        print(f"     Year: {year}, Years old: {years_old}, Boosted: {boosted}, Similarity: {paper['similarity']:.3f}")
    
    # Compare year distributions
    print("\n" + "="*80)
    print("YEAR DISTRIBUTION COMPARISON")
    print("="*80)
    
    def get_year_stats(papers):
        years = [p.get('publication_year', 0) for p in papers if p.get('publication_year', 0) > 0]
        if years:
            avg_year = sum(years) / len(years)
            newest = max(years)
            oldest = min(years)
            return avg_year, newest, oldest, len(years)
        return 0, 0, 0, 0
    
    # Stats for no boost
    avg_no, new_no, old_no, count_no = get_year_stats(papers_no_boost[:5])
    print(f"\nNo boost (top 5):")
    print(f"  Average year: {avg_no:.1f}")
    print(f"  Newest: {new_no}, Oldest: {old_no}")
    print(f"  Papers with year: {count_no}/5")
    
    # Stats for moderate boost
    avg_mod, new_mod, old_mod, count_mod = get_year_stats(papers_moderate[:5])
    print(f"\nModerate boost (top 5):")
    print(f"  Average year: {avg_mod:.1f}")
    print(f"  Newest: {new_mod}, Oldest: {old_mod}")
    print(f"  Papers with year: {count_mod}/5")
    
    # Stats for aggressive boost
    avg_agg, new_agg, old_agg, count_agg = get_year_stats(papers_aggressive[:5])
    print(f"\nAggressive boost (top 5):")
    print(f"  Average year: {avg_agg:.1f}")
    print(f"  Newest: {new_agg}, Oldest: {old_agg}")
    print(f"  Papers with year: {count_agg}/5")
    
    # Test hybrid search with recency
    print("\n" + "="*80)
    print("TESTING HYBRID SEARCH WITH RECENCY BOOST")
    print("="*80)
    
    papers_hybrid = assistant.hybrid_search(
        query,
        n_results=10,
        alpha=0.5,
        boost_recency=True,
        recency_weight=0.15
    )
    
    print("\nTop 5 results (hybrid search with recency boost):")
    for i, paper in enumerate(papers_hybrid[:5], 1):
        year = paper.get('publication_year', 'Unknown')
        hybrid_score = paper.get('hybrid_score', 0)
        bm25_score = paper.get('bm25_score', 0)
        print(f"  {i}. {paper['filename'][:50]}...")
        print(f"     Year: {year}, Hybrid: {hybrid_score:.3f}, BM25: {bm25_score:.3f}")
    
    print("\n" + "="*80)
    print("RECENCY BOOST TEST COMPLETE")
    print("="*80)
    print("\nNOTE: Papers need to be re-indexed with the new pdf_indexer.py")
    print("      to have publication_year metadata for this feature to work fully.")
    print("      Run: python src/pdf_indexer.py")

if __name__ == "__main__":
    test_recency_boost()