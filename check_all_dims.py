#!/usr/bin/env python3
"""Check if there are any embeddings with wrong dimensions"""

import chromadb
from chromadb.config import Settings

# Load the collection
client = chromadb.PersistentClient(
    path="/fastpool/rag_embeddings",
    settings=Settings(anonymized_telemetry=False)
)
collection = client.get_collection(name="research_papers")

total = collection.count()
print(f"Total documents in collection: {total}")

# Sample a larger number across the collection
sample_size = min(1000, total)
print(f"\nChecking {sample_size} random embeddings...")

result = collection.get(limit=sample_size, include=["embeddings"])
embeddings_list = result.get('embeddings', [])

if embeddings_list is not None and len(embeddings_list) > 0:
    dims = [len(emb) for emb in embeddings_list]
    unique_dims = set(dims)

    print(f"\nFound dimension(s): {unique_dims}")

    if len(unique_dims) == 1:
        dim = list(unique_dims)[0]
        if dim == 768:
            print(f"\n✓ ALL {len(dims)} sampled embeddings are 768-dimensional")
            print("  Consistent with PubMedBERT!")
        elif dim == 384:
            print(f"\n✗ ALL {len(dims)} sampled embeddings are 384-dimensional")
            print("  This is NOT PubMedBERT - likely all-MiniLM-L6-v2")
            print("  You need to DELETE the index and re-index!")
        else:
            print(f"\n? All embeddings have dimension {dim}")
    else:
        print(f"\n⚠ MIXED DIMENSIONS DETECTED!")
        for dim in unique_dims:
            count = dims.count(dim)
            print(f"  {dim}d: {count} embeddings ({count/len(dims)*100:.1f}%)")
        print("\n  Your index contains embeddings from DIFFERENT models!")
        print("  You MUST delete the index and re-index from scratch!")
else:
    print("No embeddings found")
