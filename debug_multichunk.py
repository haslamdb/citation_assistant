#!/usr/bin/env python3
"""Debug why multi-chunk isn't finding the right papers"""

import sys
sys.path.insert(0, 'src')
from citation_assistant import CitationAssistant

assistant = CitationAssistant(
    embeddings_dir="/fastpool/rag_embeddings",
    llm_model="gemma2:27b"
)

query = "how was golgicide discovered"

# Get the raw query results
query_embedding = assistant.embedding_model.encode([query])[0]

# Do a large initial fetch
fetch_count = 500
results = assistant.collection.query(
    query_embeddings=[query_embedding.tolist()],
    n_results=fetch_count,
    include=["documents", "metadatas", "distances"]
)

# Count chunks per paper
paper_counts = {}
golgicide_papers = set()

for i in range(len(results['ids'][0])):
    filename = results['metadatas'][0][i]['filename']
    text = results['documents'][0][i]
    distance = results['distances'][0][i]
    
    if filename not in paper_counts:
        paper_counts[filename] = {'count': 0, 'best_distance': float('inf'), 'has_golgicide': False}
    
    paper_counts[filename]['count'] += 1
    paper_counts[filename]['best_distance'] = min(paper_counts[filename]['best_distance'], distance)
    
    if 'golgicide' in text.lower():
        paper_counts[filename]['has_golgicide'] = True
        golgicide_papers.add(filename)

print("=" * 80)
print("PAPER ANALYSIS FROM RAW QUERY RESULTS")
print("=" * 80)

print(f"\nTotal chunks retrieved: {len(results['ids'][0])}")
print(f"Unique papers: {len(paper_counts)}")
print(f"Papers containing 'golgicide': {len(golgicide_papers)}")

print("\nTop 10 papers by best chunk distance:")
sorted_papers = sorted(paper_counts.items(), key=lambda x: x[1]['best_distance'])
for i, (filename, info) in enumerate(sorted_papers[:10]):
    golgicide_marker = "✓ GOLGICIDE" if info['has_golgicide'] else ""
    print(f"{i+1}. {filename[:50]:<50} dist={info['best_distance']:.4f} chunks={info['count']:2d} {golgicide_marker}")

print("\nGolgicide papers and their ranks:")
for paper in golgicide_papers:
    rank = next(i for i, (f, _) in enumerate(sorted_papers) if f == paper) + 1
    print(f"  {paper[:60]:<60} rank={rank:3d} dist={paper_counts[paper]['best_distance']:.4f}")

print("\n" + "=" * 80)
print("WHY MULTI-CHUNK MISSES GOLGICIDE PAPERS")
print("=" * 80)

# Now simulate what multi-chunk does
papers_all_chunks = {}
for i in range(len(results['ids'][0])):
    filename = results['metadatas'][0][i]['filename']
    if filename not in papers_all_chunks:
        papers_all_chunks[filename] = []
    papers_all_chunks[filename].append({
        'distance': results['distances'][0][i],
        'has_golgicide': 'golgicide' in results['documents'][0][i].lower()
    })

# Sort and take best 2 per paper
papers_best_2 = {}
for filename, chunks in papers_all_chunks.items():
    chunks.sort(key=lambda x: x['distance'])
    papers_best_2[filename] = chunks[:2]

# Now flatten and sort by best chunk
flat_papers = []
for filename, chunks in papers_best_2.items():
    best_chunk = chunks[0]
    flat_papers.append({
        'filename': filename,
        'distance': best_chunk['distance'],
        'has_golgicide': any(c['has_golgicide'] for c in chunks)
    })

flat_papers.sort(key=lambda x: x['distance'])

print("\nTop 10 papers after multi-chunk processing:")
for i, paper in enumerate(flat_papers[:10]):
    golgicide_marker = "✓ GOLGICIDE" if paper['has_golgicide'] else ""
    print(f"{i+1}. {paper['filename'][:50]:<50} dist={paper['distance']:.4f} {golgicide_marker}")