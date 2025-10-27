#!/usr/bin/env python3
"""Verify PubMedBERT is actually being used"""

import chromadb
from chromadb.config import Settings

# Load the collection
client = chromadb.PersistentClient(
    path="/fastpool/rag_embeddings",
    settings=Settings(anonymized_telemetry=False)
)
collection = client.get_collection(name="research_papers")

print(f"Total documents in collection: {collection.count()}")

# Get multiple samples to verify dimensions
result = collection.get(limit=10, include=["embeddings", "metadatas"])
embeddings_list = result.get('embeddings', [])

if embeddings_list is not None and len(embeddings_list) > 0:
    print(f"\nChecked {len(embeddings_list)} embeddings:")
    dims = [len(emb) for emb in embeddings_list]
    print(f"  All have dimension: {set(dims)}")

    if all(d == 768 for d in dims):
        print("\n✓ CONFIRMED: All embeddings are 768-dimensional")
        print("  This matches PubMedBERT (S-PubMedBert-MS-MARCO)")
        print("\n  However, to be 100% certain it's PubMedBERT and not")
        print("  another 768-dim model (like BERT-base or all-mpnet-base-v2),")
        print("  you should test the search quality with known biomedical queries.")
    elif all(d == 384 for d in dims):
        print("\n✗ WARNING: All embeddings are 384-dimensional")
        print("  This is NOT PubMedBERT! Likely all-MiniLM-L6-v2")
        print("  You need to RE-INDEX with PubMedBERT!")
    else:
        print(f"\n⚠ MIXED DIMENSIONS: {set(dims)}")
        print("  Your index has embeddings from DIFFERENT models!")
        print("  You MUST re-index from scratch!")
else:
    print("No embeddings found")

# Show some sample metadata to understand what's indexed
print("\n" + "="*60)
print("Sample indexed documents:")
for i, meta in enumerate(result.get('metadatas', [])[:3], 1):
    print(f"{i}. {meta.get('filename', 'unknown')}")
