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
        boost_keywords=keywords
    )
    print(f"Found {len(papers_regular)} papers")
    if papers_regular:
        print("Top 5 papers:")
        for i, p in enumerate(papers_regular[:5], 1):
            print(f"  {i}. {p['filename'][:60]}...")
            # Check for golgicide mentions
            if 'golgicide' in p['text'].lower():
                print(f"     ✓ Contains 'golgicide'")

    # Test 2: Search with different keywords
    print("\n2. SEARCH WITH 'golgicide' AS QUERY")
    papers_golgicide = assistant.search_papers(
        "golgicide",
        n_results=15
    )
    print(f"Found {len(papers_golgicide)} papers")
    if papers_golgicide:
        print("Top 5 papers:")
        for i, p in enumerate(papers_golgicide[:5], 1):
            print(f"  {i}. {p['filename'][:60]}...")
            if 'golgicide' in p['text'].lower():
                count = p['text'].lower().count('golgicide')
                print(f"     ✓ Contains 'golgicide' ({count} times)")

    # Test 3: Search with fetch multiplier
    print("\n3. SEARCH WITH INCREASED FETCH LIMITS")
    papers_high_fetch = assistant.search_papers(
        topic,
        n_results=15,
        boost_keywords=keywords,
        fetch_multiplier=200,
        max_fetch=10000
    )
    print(f"Found {len(papers_high_fetch)} papers (with higher fetch limits)")
    if papers_high_fetch:
        print("Top 5 papers:")
        for i, p in enumerate(papers_high_fetch[:5], 1):
            print(f"  {i}. {p['filename'][:60]}...")
            if 'golgicide' in p['text'].lower():
                print(f"     ✓ Contains 'golgicide'")

    # Test 4: Check fetch limits
    print("\n4. FETCH LIMITS INFO")
    print(f"Default fetch multiplier: {assistant.default_fetch_multiplier}")
    print(f"Default max fetch: {assistant.default_max_fetch}")

    return papers_regular, papers_golgicide, papers_high_fetch


if __name__ == "__main__":
    papers_regular, papers_golgicide, papers_high_fetch = test_write_document_searches()

    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Regular search found: {len(papers_regular)} papers")
    print(f"Golgicide search found: {len(papers_golgicide)} papers")
    print(f"High fetch search found: {len(papers_high_fetch)} papers")

    if len(papers_regular) < 10:
        print("\n⚠️ PROBLEM: Search is not returning enough papers!")
        print("This may explain why write_document is only using limited papers.")