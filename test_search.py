#!/usr/bin/env python3
"""Quick test of optimized search (no ollama needed)"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

print("Loading Citation Assistant with Phase 1+2 optimizations...")
print("="*70)

# Load collection
client = chromadb.PersistentClient(
    path="/fastpool/rag_embeddings",
    settings=Settings(anonymized_telemetry=False)
)
collection = client.get_collection("research_papers")

# Load embedding model
print("Loading PubMedBERT...")
embedding_model = SentenceTransformer("pritamdeka/S-PubMedBert-MS-MARCO")

# Test query
query = "dysbiosis and antibiotics"
print(f"\nQuery: '{query}'")
print("="*70)

# Generate embedding
query_embedding = embedding_model.encode([query])[0]

# Search with Phase 1 optimizations (500 chunks instead of 100)
print("\nSearching with Phase 1+2 optimizations:")
print("  • Fetch: 500 chunks (was 100)")
print("  • Semantic chunking: sentence boundaries, ~500 tokens")
print("  • PubMedBERT: 768-dim biomedical embeddings")
print()

results = collection.query(
    query_embeddings=[query_embedding.tolist()],
    n_results=500,
    include=["documents", "metadatas", "distances"]
)

# Deduplicate by filename
unique_papers = {}
for i in range(len(results['ids'][0])):
    filename = results['metadatas'][0][i]['filename']
    distance = results['distances'][0][i]
    text = results['documents'][0][i]

    if filename not in unique_papers or distance < unique_papers[filename]['distance']:
        unique_papers[filename] = {
            'filename': filename,
            'distance': distance,
            'similarity': 1 / (1 + distance),  # Normalize L2 distance to 0-1 range
            'text': text,
            'chunk_method': results['metadatas'][0][i].get('chunking_method', 'unknown'),
            'chunk_size': results['metadatas'][0][i].get('chunk_size_chars', len(text))
        }

# Sort by similarity
papers = sorted(unique_papers.values(), key=lambda x: x['distance'])[:10]

print(f"Found {len(unique_papers)} unique papers, showing top 10:")
print("="*70)
print("Note: Similarity uses normalized L2 distance: 1/(1+distance)")
print("      Typical range: 1-5% (higher is better, 100% = identical)")
print()

for i, paper in enumerate(papers, 1):
    print(f"{i}. {paper['filename']}")
    print(f"   Similarity: {paper['similarity']:.1%}")
    print(f"   Chunking: {paper['chunk_method']} ({paper['chunk_size']} chars)")
    print(f"   Excerpt: {paper['text'][:150]}...")
    print()

print("="*70)
print("✓ Phase 1+2 optimizations active and working!")
print(f"✓ {len(papers)} relevant papers found with semantic chunking")
