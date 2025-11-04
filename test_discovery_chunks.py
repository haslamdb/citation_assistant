#!/usr/bin/env python3
"""Find which chunks contain the golgicide discovery story"""

import sys
sys.path.insert(0, 'src')
from citation_assistant import CitationAssistant

assistant = CitationAssistant(
    embeddings_dir="/fastpool/rag_embeddings",
    llm_model="gemma2:27b"
)

# Get ALL chunks for the main golgicide paper
query = "golgicide discovery high-throughput screen shiga toxin"
query_embedding = assistant.embedding_model.encode([query])[0]

# Fetch many chunks
results = assistant.collection.query(
    query_embeddings=[query_embedding.tolist()],
    n_results=1000,
    include=["documents", "metadatas", "distances"]
)

# Find all chunks from nchembio.144.pdf
target_paper = "nchembio.144.pdf"
paper_chunks = []

for i in range(len(results['ids'][0])):
    if results['metadatas'][0][i]['filename'] == target_paper:
        text = results['documents'][0][i]
        chunk_info = {
            'index': results['metadatas'][0][i].get('chunk_index', i),
            'distance': results['distances'][0][i],
            'text': text,
            'has_discovery': False,
            'discovery_keywords': []
        }
        
        # Check for discovery content
        text_lower = text.lower()
        discovery_terms = {
            'high-throughput': 'high-throughput' in text_lower,
            'screen': 'screen' in text_lower,
            'identified': 'identified' in text_lower,
            'compound': 'compound' in text_lower,
            'protected': 'protected' in text_lower,
            'vero cells': 'vero' in text_lower,
            'shiga toxin': 'shiga' in text_lower,
            'stx': 'stx' in text_lower,
            'luciferase': 'luciferase' in text_lower,
            'bacterial toxin': 'bacterial toxin' in text_lower
        }
        
        chunk_info['discovery_keywords'] = [k for k, v in discovery_terms.items() if v]
        chunk_info['has_discovery'] = len(chunk_info['discovery_keywords']) >= 3
        
        paper_chunks.append(chunk_info)

# Sort by distance
paper_chunks.sort(key=lambda x: x['distance'])

print(f"Found {len(paper_chunks)} chunks from {target_paper}")
print("=" * 80)

# Show top 10 chunks
print("\nTop 10 chunks by distance to query:")
for i, chunk in enumerate(paper_chunks[:10]):
    discovery_marker = "✓✓ DISCOVERY" if chunk['has_discovery'] else ""
    print(f"\nChunk {i+1} (index {chunk['index']}, distance {chunk['distance']:.3f}) {discovery_marker}")
    print(f"Keywords found: {', '.join(chunk['discovery_keywords']) if chunk['discovery_keywords'] else 'none'}")
    
    # Show first 300 chars
    preview = chunk['text'][:300].replace('\n', ' ')
    print(f"Preview: {preview}...")

# Find the chunk with the most discovery keywords
best_discovery_chunk = max(paper_chunks, key=lambda x: len(x['discovery_keywords']))
print("\n" + "=" * 80)
print("CHUNK WITH MOST DISCOVERY KEYWORDS:")
print("=" * 80)
print(f"Chunk index: {best_discovery_chunk['index']}")
print(f"Distance: {best_discovery_chunk['distance']:.3f}")
print(f"Keywords: {', '.join(best_discovery_chunk['discovery_keywords'])}")
print(f"\nFull text:\n{best_discovery_chunk['text']}")

# Check if discovery chunk is in top N
for n in [2, 3, 5, 10]:
    has_discovery = any(c['has_discovery'] for c in paper_chunks[:n])
    print(f"\nTop {n} chunks contain discovery: {'YES' if has_discovery else 'NO'}")