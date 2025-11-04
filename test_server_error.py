#!/usr/bin/env python3
"""Test script to reproduce the 500 error"""

import sys
sys.path.insert(0, 'src')

from citation_assistant import CitationAssistant

# Initialize assistant with correct path
print("Initializing CitationAssistant...")
try:
    assistant = CitationAssistant(
        embeddings_dir="/fastpool/rag_embeddings",
        llm_model="gemma2:27b"
    )
    print(f"✓ Assistant initialized with {assistant.collection.count()} documents")
except Exception as e:
    print(f"✗ Failed to initialize: {e}")
    sys.exit(1)

# Test the multi-chunk method
print("\nTesting search_papers_multi_chunk...")
try:
    results = assistant.search_papers_multi_chunk(
        "golgicide",
        n_results=2,
        chunks_per_paper=2
    )
    print(f"✓ Multi-chunk search returned {len(results)} results")
    if results:
        print(f"  First result type: {type(results[0])}")
        print(f"  Keys: {list(results[0].keys())}")
        print(f"  Chunks in first result: {results[0].get('num_chunks', 'N/A')}")
except Exception as e:
    print(f"✗ Multi-chunk search failed: {e}")
    import traceback
    traceback.print_exc()

# Test the factual method
print("\nTesting search_papers_factual...")
try:
    results = assistant.search_papers_factual(
        "how was golgicide discovered",
        n_results=2
    )
    print(f"✓ Factual search returned {len(results)} results")
    if results:
        print(f"  First result type: {type(results[0])}")
        print(f"  Keys: {list(results[0].keys())}")
except Exception as e:
    print(f"✗ Factual search failed: {e}")
    import traceback
    traceback.print_exc()