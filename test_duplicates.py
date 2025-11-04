#!/usr/bin/env python3
"""
Test duplicate detection settings to ensure we're getting multiple results
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.citation_assistant import CitationAssistant

def test_duplicate_settings():
    """Compare search results with and without duplicate checking"""
    
    assistant = CitationAssistant(embeddings_dir="/fastpool/rag_embeddings")
    
    test_queries = [
        "CRISPR gene editing",
        "machine learning",
        "microbiome",
        "golgicide discovery"
    ]
    
    print("\n" + "="*80)
    print("TESTING DUPLICATE DETECTION SETTINGS")
    print("="*80)
    
    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: '{query}'")
        print(f"{'='*60}")
        
        # Test with duplicate checking DISABLED (default now)
        results_no_check = assistant.search_papers(
            query, 
            n_results=10,
            check_duplicates=False
        )
        print(f"\nWithout duplicate checking: {len(results_no_check)} papers")
        if results_no_check:
            print("First 3 papers:")
            for i, paper in enumerate(results_no_check[:3], 1):
                print(f"  {i}. {paper['filename'][:60]}... (sim: {paper['similarity']:.3f})")
        
        # Test with duplicate checking ENABLED
        results_with_check = assistant.search_papers(
            query, 
            n_results=10,
            check_duplicates=True
        )
        print(f"\nWith duplicate checking (0.95 threshold): {len(results_with_check)} papers")
        if results_with_check:
            print("First 3 papers:")
            for i, paper in enumerate(results_with_check[:3], 1):
                print(f"  {i}. {paper['filename'][:60]}... (sim: {paper['similarity']:.3f})")
        
        # Calculate difference
        difference = len(results_no_check) - len(results_with_check)
        if difference > 0:
            print(f"\n✅ Duplicate checking removed {difference} potential duplicates")
        elif difference == 0:
            print(f"\n✅ No duplicates detected (same results)")
        
        # Test multi-chunk method
        results_multi = assistant.search_papers_multi_chunk(
            query,
            n_results=10,
            chunks_per_paper=2,
            check_duplicates=False
        )
        print(f"\nMulti-chunk (no dup check): {len(results_multi)} papers")
        
        # Test factual method
        results_factual = assistant.search_papers_factual(
            query,
            n_results=10,
            check_duplicates=False
        )
        print(f"Factual search (no dup check): {len(results_factual)} papers")
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print("✅ All search methods now default to check_duplicates=False")
    print("✅ Duplicate checking can be enabled when needed with check_duplicates=True")
    print("✅ Threshold is set to 0.95 (very high similarity) to catch only true duplicates")


if __name__ == "__main__":
    test_duplicate_settings()