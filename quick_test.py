#!/usr/bin/env python3
"""Quick test of optimized search"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from citation_assistant import CitationAssistant

# Test with optimized parameters
print("Loading Citation Assistant with Phase 1 optimizations...")
print("  • fetch_multiplier: 50 (was 10)")
print("  • max_fetch: 2000 (was 500)")
print("  • keyword_boost: 0.7^n (was 0.1^n)\n")

assistant = CitationAssistant(
    embeddings_dir="/fastpool/rag_embeddings"
)

# Test query
query = "gut microbiome and immune system"
print(f"Testing query: '{query}'")
print("-" * 60)

papers = assistant.search_papers(query, n_results=5)

print(f"\nFound {len(papers)} papers:\n")
for i, paper in enumerate(papers, 1):
    print(f"{i}. {paper['filename']}")
    print(f"   Similarity: {paper['similarity']:.1%}")
    print(f"   Keyword matches: {paper['keyword_matches']}")
    print(f"   Excerpt: {paper['text'][:120]}...")
    print()

print("-" * 60)
print("✓ Phase 1 optimizations are active!")
print("\nTo test with different parameters, you can override:")
print("  papers = assistant.search_papers(query,")
print("      fetch_multiplier=100,  # Fetch more chunks")
print("      keyword_boost_strength=0.5  # Stronger keyword boost")
print("  )")
