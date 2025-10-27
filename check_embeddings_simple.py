#!/usr/bin/env python3
"""Check the embedding model dimensions in the current index"""

import chromadb
from chromadb.config import Settings

# Load the collection
client = chromadb.PersistentClient(
    path="/fastpool/rag_embeddings",
    settings=Settings(anonymized_telemetry=False)
)
collection = client.get_collection(name="research_papers")

# Get a sample embedding from the database
result = collection.get(limit=1, include=["embeddings"])
embeddings_list = result.get('embeddings', [])
if embeddings_list is not None and len(embeddings_list) > 0:
    db_embedding_dim = len(embeddings_list[0])
    print(f"Database embedding dimension: {db_embedding_dim}")
else:
    print("No embeddings found in database")
    exit(1)

# Compare with known model dimensions
print("\n" + "="*60)
print("Common model dimensions:")
print("  - 384: all-MiniLM-L6-v2, all-MiniLM-L12-v2")
print("  - 768: BERT-base, S-PubMedBert-MS-MARCO, all-mpnet-base-v2")
print("  - 1024: BERT-large")
print("="*60)

if db_embedding_dim == 768:
    print(f"\n✓ Your index uses dimension {db_embedding_dim}")
    print("  This is CONSISTENT with PubMedBERT (S-PubMedBert-MS-MARCO)")
    print("  However, other 768-dim models (like BERT-base) also have this dimension.")
    print("\n  To verify the model is actually PubMedBERT:")
    print("  - Check your commit history for when the index was created")
    print("  - Or re-index with PubMedBERT to be certain")
elif db_embedding_dim == 384:
    print(f"\n✗ Your index uses dimension {db_embedding_dim}")
    print("  This is likely all-MiniLM (NOT PubMedBERT!)")
    print("  You need to re-index with PubMedBERT!")
else:
    print(f"\n? Your index uses dimension {db_embedding_dim}")
    print("  This doesn't match PubMedBERT (should be 768)")
