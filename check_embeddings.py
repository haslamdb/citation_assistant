#!/usr/bin/env python3
"""Check the embedding model dimensions in the current index"""

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

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

# Check PubMedBERT dimensions
print("\nLoading PubMedBERT model...")
pubmedbert = SentenceTransformer("pritamdeka/S-PubMedBert-MS-MARCO")
test_embedding = pubmedbert.encode(["test text"])
pubmedbert_dim = len(test_embedding[0])
print(f"PubMedBERT embedding dimension: {pubmedbert_dim}")

# Compare
if db_embedding_dim == pubmedbert_dim:
    print(f"\n✓ MATCH! Your index is using PubMedBERT (dimension {db_embedding_dim})")
else:
    print(f"\n✗ MISMATCH! Database has dimension {db_embedding_dim}, but PubMedBERT has dimension {pubmedbert_dim}")
    print("\nThis means your index was created with a DIFFERENT model!")
    print("You need to re-index with PubMedBERT to get the full benefits.")

    # Try to identify what model might have been used
    print("\nCommon model dimensions:")
    print("  - 384: all-MiniLM-L6-v2, all-MiniLM-L12-v2")
    print("  - 768: BERT-base, S-PubMedBert-MS-MARCO, sentence-transformers/all-mpnet-base-v2")
    print("  - 1024: BERT-large")
