#!/usr/bin/env python3
"""Test what's actually being retrieved for golgicide discovery query"""

import sys
sys.path.insert(0, 'src')
from citation_assistant import CitationAssistant

# Initialize
assistant = CitationAssistant(
    embeddings_dir="/fastpool/rag_embeddings",
    llm_model="gemma2:27b"
)

query = "how was golgicide discovered"

print("=" * 80)
print("TESTING DIFFERENT SEARCH METHODS FOR GOLGICIDE DISCOVERY")
print("=" * 80)

# Test 1: Default single-chunk search
print("\n1. DEFAULT SEARCH (single chunk per paper):")
print("-" * 40)
results_default = assistant.search_papers(query, n_results=3)
for i, r in enumerate(results_default):
    print(f"\nPaper {i+1}: {r['filename']}")
    print(f"Similarity: {r['similarity']:.3f}")
    # Check if discovery keywords are present
    text_lower = r['text'].lower()
    has_discovery = any(kw in text_lower for kw in ['discovered', 'screen', 'identified', 'found', 'first'])
    print(f"Has discovery keywords: {has_discovery}")
    if 'golgicide' in text_lower:
        print("✓ Contains 'golgicide'")
        # Print context around golgicide mentions
        idx = text_lower.index('golgicide')
        print(f"Context: ...{r['text'][max(0,idx-100):min(len(r['text']),idx+200)]}...")
    else:
        print("✗ No 'golgicide' found in chunk")

# Test 2: Multi-chunk search
print("\n\n2. MULTI-CHUNK SEARCH (2 chunks per paper):")
print("-" * 40)
results_multi = assistant.search_papers_multi_chunk(query, n_results=3, chunks_per_paper=2)
for i, r in enumerate(results_multi):
    print(f"\nPaper {i+1}: {r['filename']}")
    print(f"Similarity: {r['similarity']:.3f}")
    print(f"Number of chunks: {r['num_chunks']}")
    text_lower = r['text'].lower()
    
    # Check for discovery content
    has_discovery = any(kw in text_lower for kw in ['discovered', 'screen', 'identified', 'found', 'first'])
    print(f"Has discovery keywords: {has_discovery}")
    
    # Look for specific discovery phrases
    if 'high-throughput screen' in text_lower:
        print("✓✓ FOUND: 'high-throughput screen'")
    if 'identified a compound' in text_lower:
        print("✓✓ FOUND: 'identified a compound'")
    if 'protected vero cells' in text_lower:
        print("✓✓ FOUND: 'protected Vero cells'")
    if 'shiga toxin' in text_lower:
        print("✓✓ FOUND: 'shiga toxin'")
    
    if 'golgicide' in text_lower:
        print(f"✓ Contains 'golgicide' ({text_lower.count('golgicide')} times)")
        # Find all mentions
        import re
        for match in re.finditer(r'.{0,100}golgicide.{0,100}', text_lower):
            print(f"  Context: ...{match.group()}...")
    else:
        print("✗ No 'golgicide' found")

# Test 3: Factual/Discovery search
print("\n\n3. FACTUAL/DISCOVERY SEARCH (keyword-boosted):")
print("-" * 40)
results_factual = assistant.search_papers_factual(query, n_results=3)
for i, r in enumerate(results_factual):
    print(f"\nPaper {i+1}: {r['filename']}")
    print(f"Similarity: {r['similarity']:.3f}")
    print(f"Keywords found: {r.get('keywords_found', 0)}")
    text_lower = r['text'].lower()
    
    # Check for discovery content
    if 'high-throughput screen' in text_lower:
        print("✓✓ FOUND: 'high-throughput screen'")
    if 'identified a compound' in text_lower:
        print("✓✓ FOUND: 'identified a compound'")
    if 'protected vero cells' in text_lower:
        print("✓✓ FOUND: 'protected Vero cells'")
    
    if 'golgicide' in text_lower:
        print(f"✓ Contains 'golgicide' ({text_lower.count('golgicide')} times)")

# Now check what the top paper actually contains
print("\n" * 2)
print("=" * 80)
print("TOP PAPER ANALYSIS")
print("=" * 80)

# Get the best paper from factual search
if results_factual:
    top_paper = results_factual[0]
    print(f"Top paper: {top_paper['filename']}")
    print(f"Full text length: {len(top_paper['text'])} characters")
    print("\nSearching for key discovery phrases:")
    
    text = top_paper['text']
    text_lower = text.lower()
    
    # Search for the discovery narrative
    discovery_phrases = [
        "high-throughput screen",
        "bacterial toxins",
        "host cells",
        "identified a compound",
        "protected vero cells",
        "shiga toxin",
        "golgicide a",
        "named golgicide"
    ]
    
    for phrase in discovery_phrases:
        if phrase in text_lower:
            print(f"✓ Found: '{phrase}'")
            # Show context
            idx = text_lower.index(phrase)
            context_start = max(0, idx - 200)
            context_end = min(len(text), idx + len(phrase) + 200)
            context = text[context_start:context_end]
            print(f"  Context: ...{context}...")
        else:
            print(f"✗ NOT found: '{phrase}'")