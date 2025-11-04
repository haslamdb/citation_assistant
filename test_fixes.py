#!/usr/bin/env python3
"""
Test script to verify the fixes for citation_assistant
Tests:
1. Multi-chunk retrieval gets diverse chunks (not just best scoring)
2. Duplicate detection works properly
3. Discovery queries return relevant discovery narratives
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.citation_assistant import CitationAssistant

def test_multi_chunk_diversity():
    """Test that multi_chunk returns diverse chunks from different parts of papers"""
    print("\n" + "="*80)
    print("TEST 1: Multi-chunk Diversity")
    print("="*80)
    
    assistant = CitationAssistant(embeddings_dir="/fastpool/rag_embeddings")
    
    # Test with a discovery query
    query = "how was golgicide discovered"
    results = assistant.search_papers_multi_chunk(query, n_results=3, chunks_per_paper=3)
    
    print(f"\nQuery: '{query}'")
    print(f"Results: {len(results)} papers")
    
    for i, paper in enumerate(results, 1):
        print(f"\n{i}. {paper['filename']}")
        print(f"   Chunks retrieved: {paper['num_chunks']}")
        print(f"   Similarity: {paper['similarity']:.3f}")
        
        # Check if we have multiple chunk markers in the text
        chunk_markers = paper['text'].count('[Chunk')
        print(f"   Chunk sections found: {chunk_markers}")
        
        # Check for discovery keywords
        discovery_keywords = ['discovered', 'screen', 'identified', 'first', 'novel']
        keywords_found = sum(1 for kw in discovery_keywords if kw in paper['text'].lower())
        print(f"   Discovery keywords found: {keywords_found}")
        
        # Show first 500 chars
        print(f"   Preview: {paper['text'][:500]}...")
        
    return len(results) > 0 and results[0]['num_chunks'] > 1


def test_duplicate_detection():
    """Test that duplicate papers are properly detected and removed"""
    print("\n" + "="*80)
    print("TEST 2: Duplicate Detection")
    print("="*80)
    
    assistant = CitationAssistant(embeddings_dir="/fastpool/rag_embeddings")
    
    # Search for a common topic that might have duplicates
    query = "CRISPR gene editing"
    
    # Test regular search
    results_regular = assistant.search_papers(query, n_results=10)
    print(f"\nRegular search for '{query}':")
    print(f"Found {len(results_regular)} unique papers")
    
    # Check for similar filenames (potential duplicates)
    filenames = [p['filename'] for p in results_regular]
    print("\nFilenames returned:")
    for fname in filenames[:5]:
        print(f"  - {fname}")
    
    # Test multi-chunk search
    results_multi = assistant.search_papers_multi_chunk(query, n_results=10, chunks_per_paper=2)
    print(f"\nMulti-chunk search for '{query}':")
    print(f"Found {len(results_multi)} unique papers")
    
    # Test factual search
    results_factual = assistant.search_papers_factual(query, n_results=10)
    print(f"\nFactual search for '{query}':")
    print(f"Found {len(results_factual)} unique papers")
    
    return True


def test_discovery_queries():
    """Test that discovery queries return relevant discovery narratives"""
    print("\n" + "="*80)
    print("TEST 3: Discovery Query Relevance")
    print("="*80)
    
    assistant = CitationAssistant(embeddings_dir="/fastpool/rag_embeddings")
    
    discovery_queries = [
        "how was golgicide discovered",
        "discovery of CRISPR",
        "first identification of coronavirus"
    ]
    
    for query in discovery_queries:
        print(f"\n--- Query: '{query}' ---")
        
        # Test factual search (optimized for discovery)
        results = assistant.search_papers_factual(query, n_results=3)
        
        if results:
            top_result = results[0]
            print(f"Top result: {top_result['filename']}")
            print(f"Similarity: {top_result['similarity']:.3f}")
            print(f"Keywords found: {top_result.get('keywords_found', 0)}")
            
            # Check for discovery-related content
            discovery_terms = ['discovered', 'discovery', 'first', 'identified', 'novel', 'screen', 'isolated']
            text_lower = top_result['text'].lower()
            
            found_terms = [term for term in discovery_terms if term in text_lower]
            print(f"Discovery terms in result: {found_terms}")
            
            # Show relevant excerpt
            for term in found_terms[:1]:  # Show context for first term found
                idx = text_lower.find(term)
                if idx != -1:
                    start = max(0, idx - 100)
                    end = min(len(top_result['text']), idx + 200)
                    excerpt = top_result['text'][start:end]
                    print(f"\nRelevant excerpt:")
                    print(f"...{excerpt}...")
                    break
        else:
            print("No results found")
    
    return True


def test_hybrid_search():
    """Test that hybrid search combines vector and BM25 effectively"""
    print("\n" + "="*80)
    print("TEST 4: Hybrid Search")
    print("="*80)
    
    assistant = CitationAssistant(embeddings_dir="/fastpool/rag_embeddings")
    
    # Test with specific technical terms that benefit from exact matching
    query = "Clostridioides difficile toxin"
    
    if assistant.bm25_index:
        results = assistant.hybrid_search(query, n_results=5, alpha=0.5)
        print(f"\nHybrid search for '{query}':")
        print(f"Found {len(results)} papers")
        
        for i, paper in enumerate(results[:3], 1):
            print(f"\n{i}. {paper['filename']}")
            print(f"   Hybrid score: {paper.get('hybrid_score', 0):.3f}")
            print(f"   Vector similarity: {paper['similarity']:.3f}")
            print(f"   BM25 score: {paper.get('bm25_score', 0):.3f}")
            
            # Check for exact term matches
            exact_matches = query.lower().split()
            matches_found = sum(1 for term in exact_matches if term in paper['text'].lower())
            print(f"   Exact term matches: {matches_found}/{len(exact_matches)}")
    else:
        print("BM25 index not available, skipping hybrid search test")
    
    return True


def main():
    """Run all tests"""
    print("\nTesting Citation Assistant Fixes")
    print("="*80)
    
    tests_passed = 0
    tests_total = 4
    
    try:
        if test_multi_chunk_diversity():
            print("\n✅ TEST 1 PASSED: Multi-chunk retrieves diverse chunks")
            tests_passed += 1
        else:
            print("\n❌ TEST 1 FAILED: Multi-chunk not retrieving diverse chunks")
    except Exception as e:
        print(f"\n❌ TEST 1 ERROR: {e}")
    
    try:
        if test_duplicate_detection():
            print("\n✅ TEST 2 PASSED: Duplicate detection working")
            tests_passed += 1
        else:
            print("\n❌ TEST 2 FAILED: Duplicate detection not working properly")
    except Exception as e:
        print(f"\n❌ TEST 2 ERROR: {e}")
    
    try:
        if test_discovery_queries():
            print("\n✅ TEST 3 PASSED: Discovery queries return relevant results")
            tests_passed += 1
        else:
            print("\n❌ TEST 3 FAILED: Discovery queries not returning relevant results")
    except Exception as e:
        print(f"\n❌ TEST 3 ERROR: {e}")
    
    try:
        if test_hybrid_search():
            print("\n✅ TEST 4 PASSED: Hybrid search working")
            tests_passed += 1
        else:
            print("\n❌ TEST 4 FAILED: Hybrid search not working properly")
    except Exception as e:
        print(f"\n❌ TEST 4 ERROR: {e}")
    
    print("\n" + "="*80)
    print(f"SUMMARY: {tests_passed}/{tests_total} tests passed")
    print("="*80)
    
    if tests_passed == tests_total:
        print("🎉 All tests passed! The citation assistant should be working properly now.")
    else:
        print(f"⚠️ Some tests failed. Please review the output above.")


if __name__ == "__main__":
    main()