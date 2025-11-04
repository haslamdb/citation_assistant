#!/usr/bin/env python3
"""
Test the write_document function to diagnose why it's only finding one paper
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.citation_assistant import CitationAssistant

def test_write_document_searches():
    """Test different search methods for the write_document function"""
    
    assistant = CitationAssistant(embeddings_dir="/fastpool/rag_embeddings")
    
    topic = "Describe how golgicide was discovered"
    keywords = "golgicide, high-throughput, screen"
    
    print("\n" + "="*80)
    print(f"TESTING SEARCH METHODS FOR TOPIC: '{topic}'")
    print(f"Keywords: {keywords}")
    print("="*80)
    
    # Test 1: Regular search
    print("\n1. REGULAR SEARCH (search_papers)")
    papers_regular = assistant.search_papers(
        topic,
        n_results=15,
        boost_keywords=keywords,
        check_duplicates=False
    )
    print(f"Found {len(papers_regular)} papers")
    if papers_regular:
        print("Top 5 papers:")
        for i, p in enumerate(papers_regular[:5], 1):
            print(f"  {i}. {p['filename'][:60]}...")
            # Check for golgicide mentions
            if 'golgicide' in p['text'].lower():
                print(f"     ✓ Contains 'golgicide'")
    
    # Test 2: Multi-chunk search
    print("\n2. MULTI-CHUNK SEARCH (search_papers_multi_chunk)")
    papers_multi = assistant.search_papers_multi_chunk(
        topic,
        n_results=15,
        chunks_per_paper=5,
        check_duplicates=False
    )
    print(f"Found {len(papers_multi)} papers")
    if papers_multi:
        print("Top 5 papers:")
        for i, p in enumerate(papers_multi[:5], 1):
            print(f"  {i}. {p['filename'][:60]}... ({p['num_chunks']} chunks)")
            if 'golgicide' in p['text'].lower():
                print(f"     ✓ Contains 'golgicide'")
    
    # Test 3: Factual search
    print("\n3. FACTUAL SEARCH (search_papers_factual)")
    papers_factual = assistant.search_papers_factual(
        topic,
        n_results=15,
        check_duplicates=False
    )
    print(f"Found {len(papers_factual)} papers")
    if papers_factual:
        print("Top 5 papers:")
        for i, p in enumerate(papers_factual[:5], 1):
            print(f"  {i}. {p['filename'][:60]}...")
            if 'golgicide' in p['text'].lower():
                print(f"     ✓ Contains 'golgicide'")
    
    # Test 4: Try with just "golgicide" as query
    print("\n4. SIMPLE QUERY TEST - just 'golgicide'")
    papers_simple = assistant.search_papers_multi_chunk(
        "golgicide",
        n_results=15,
        chunks_per_paper=3,
        check_duplicates=False
    )
    print(f"Found {len(papers_simple)} papers")
    if papers_simple:
        print("Top 5 papers:")
        for i, p in enumerate(papers_simple[:5], 1):
            print(f"  {i}. {p['filename'][:60]}...")
            if 'golgicide' in p['text'].lower():
                count = p['text'].lower().count('golgicide')
                print(f"     ✓ Contains 'golgicide' ({count} times)")
    
    # Test 5: Check if we're hitting max_fetch limits
    print("\n5. FETCH LIMITS TEST")
    print(f"Default fetch multiplier: {assistant.default_fetch_multiplier}")
    print(f"Default max fetch: {assistant.default_max_fetch}")
    print(f"For 15 papers with 5 chunks each:")
    print(f"  - Fetch count would be: {15 * assistant.default_fetch_multiplier * 5} chunks")
    print(f"  - Limited to max: {assistant.default_max_fetch}")
    
    # Test with increased fetch limits
    print("\n6. TEST WITH INCREASED FETCH LIMITS")
    papers_high_fetch = assistant.search_papers_multi_chunk(
        topic,
        n_results=15,
        chunks_per_paper=5,
        check_duplicates=False,
        fetch_multiplier=200,
        max_fetch=10000
    )
    print(f"Found {len(papers_high_fetch)} papers (with higher fetch limits)")
    
    return papers_regular, papers_multi, papers_factual


if __name__ == "__main__":
    papers_regular, papers_multi, papers_factual = test_write_document_searches()
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Regular search found: {len(papers_regular)} papers")
    print(f"Multi-chunk found: {len(papers_multi)} papers")  
    print(f"Factual search found: {len(papers_factual)} papers")
    
    if len(papers_multi) < 10:
        print("\n⚠️ PROBLEM: Multi-chunk search is not returning enough papers!")
        print("This explains why write_document is only using one paper.")