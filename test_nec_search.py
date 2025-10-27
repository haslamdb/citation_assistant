#!/usr/bin/env python3
"""Test NEC statement citation search"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

print("Testing Citation Search for NEC Statement")
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

# Test statement
statement = """Necrotizing enterocolitis (NEC) is a catastrophic intestinal disease and a leading cause of death in preterm infants, with mortality rates of 20-30%. Survivors face lifelong consequences, including short bowel syndrome and neurodevelopmental impairment. The preterm intestine is developmentally immature, and successful maturation of the epithelial barrier and immune system depends on signals and nutrients provided by commensal bacteria.
Paradoxically, while often life-saving, early antibiotic exposure is one of the strongest risk factors for NEC."""

print(f"\nStatement: {statement[:200]}...")
print("="*70)

# Generate embedding
query_embedding = embedding_model.encode([statement])[0]

# Search with Phase 1 optimizations
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
            'similarity': 1 / (1 + distance),
            'text': text
        }

# Sort by distance
papers = sorted(unique_papers.values(), key=lambda x: x['distance'])[:10]

print(f"\nFound {len(unique_papers)} unique papers, showing top 10:")
print("="*70)
print()

# Key terms to look for
key_terms = ['nec', 'necrotizing', 'enterocolitis', 'preterm', 'infant', 'antibiotic', 'butyrate', 'microbiome', 'gut', 'intestin']

for i, paper in enumerate(papers, 1):
    print(f"{i}. {paper['filename']}")
    print(f"   Distance: {paper['distance']:.2f} | Similarity: {paper['similarity']:.2%}")

    # Check for key terms
    text_lower = paper['text'].lower()
    found_terms = [t for t in key_terms if t in text_lower]
    print(f"   Key terms found: {', '.join(found_terms) if found_terms else 'NONE'}")

    print(f"   Excerpt: {paper['text'][:200]}...")
    print()

print("="*70)
print("Analysis:")
print("- Are the top results relevant to NEC/antibiotics/preterm infants?")
print("- Do they contain appropriate key terms?")
print("- Are important papers missing from top 10?")
