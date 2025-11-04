#!/usr/bin/env python3
"""
Test to verify search relevance for specific compounds
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.citation_assistant import CitationAssistant, GlobalConfig

def test_golgicide_search():
    """Test that searching for 'golgicide A' actually finds papers about golgicide A"""
    
    config = GlobalConfig(
        n_papers=10,
        chunks_per_paper=1,
        enable_hybrid=True,
        hybrid_balance=0.5
    )
    
    assistant = CitationAssistant(
        embeddings_dir="/fastpool/rag_embeddings",
        config=config
    )
    
    query = "golgicide A"
    print(f"\nSearching for: '{query}'")
    print("="*60)
    
    results = assistant.search_papers(query)
    
    # Check how many results actually contain "golgicide"
    golgicide_papers = []
    golgi_only_papers = []
    
    for paper in results:
        text_lower = paper['text'].lower()
        filename_lower = paper['filename'].lower()
        
        if 'golgicide' in text_lower or 'golgicide' in filename_lower:
            golgicide_papers.append(paper)
        else:
            golgi_only_papers.append(paper)
    
    print(f"\n📊 Search Results Analysis:")
    print(f"Total papers returned: {len(results)}")
    print(f"Papers mentioning 'golgicide': {len(golgicide_papers)}")
    print(f"Papers only about Golgi: {len(golgi_only_papers)}")
    
    if golgicide_papers:
        print(f"\n✅ Papers with 'golgicide':")
        for p in golgicide_papers[:5]:
            count = p['text'].lower().count('golgicide')
            print(f"  - {p['filename'][:60]}... ({count} mentions)")
    
    if golgi_only_papers:
        print(f"\n❌ Papers WITHOUT 'golgicide' (should rank lower):")
        for p in golgi_only_papers[:5]:
            print(f"  - {p['filename'][:60]}...")
    
    # Test with exact match requirement
    print(f"\n{'='*60}")
    print("Testing with compound-specific search...")
    
    # The enhanced assistant should detect this as a compound query
    query2 = "golgicide A discovery mechanism"
    results2 = assistant.search_papers(query2, boost_keywords="golgicide")
    
    golgicide_papers2 = [p for p in results2 if 'golgicide' in p['text'].lower()]
    
    print(f"Query: '{query2}'")
    print(f"Papers with 'golgicide': {len(golgicide_papers2)}/{len(results2)}")
    
    return len(golgicide_papers) > len(golgi_only_papers)

if __name__ == "__main__":
    success = test_golgicide_search()
    
    print(f"\n{'='*60}")
    if success:
        print("✅ PASS: Search correctly prioritizes exact matches")
    else:
        print("❌ FAIL: Search is NOT finding relevant papers!")
        print("\nThe problem is clear:")
        print("1. Vector similarity between 'golgicide A' and 'Golgi apparatus' is too high")
        print("2. Entity detection and exact match boosting is not working")
        print("3. Papers with exact compound names should get MASSIVE boost")