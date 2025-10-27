# Citation Assistant

AI-powered citation assistant using your EndNote library with RAG (Retrieval-Augmented Generation).

## Features

- **Incremental Indexing**: Only processes new or modified PDFs from your EndNote library
- **Semantic Search**: Find relevant papers using natural language queries
- **Research Summarization**: Get AI-generated summaries of research on specific topics
- **Smart Citations**: Analyze your manuscript and get citation suggestions for specific claims

## Architecture

- **Embeddings**: sentence-transformers (all-MiniLM-L6-v2) running on GPU
- **Vector DB**: ChromaDB for fast semantic search
- **LLM**: Ollama (gemma2:27b) for summarization and analysis
- **PDF Processing**: PyMuPDF for text extraction

## Directory Structure

```
citation_assistant/
├── src/
│   ├── pdf_indexer.py          # Incremental PDF indexing
│   └── citation_assistant.py   # Citation search and suggestions
├── data/                        # Temporary data
├── configs/                     # Configuration files
└── README.md
```

## Data Locations

- **EndNote Library**: `/home/david/projects/EndNote_Library/`
- **Vector Embeddings**: `/fastpool/rag_embeddings/`
- **Index State**: `/fastpool/rag_embeddings/index_state.json`

## Usage

### 1. Index your EndNote library (first time or after adding papers)

```bash
mamba activate rag
cd /home/david/projects/citation_assistant
python src/pdf_indexer.py
```

This will:
- Scan your EndNote PDF folder
- Index only new or modified PDFs
- Store embeddings in ChromaDB
- Track indexed files for incremental updates

### 2. Search for papers

```python
from src.citation_assistant import CitationAssistant

assistant = CitationAssistant(embeddings_dir="/fastpool/rag_embeddings")

# Search for relevant papers
papers = assistant.search_papers("CRISPR gene editing", n_results=10)

for paper in papers:
    print(f"{paper['filename']}: {paper['similarity']:.2%} match")
```

### 3. Summarize research on a topic

```python
summary = assistant.summarize_research("microbiome dysbiosis in IBD", n_papers=10)
print(summary)
```

### 4. Get citation suggestions for your manuscript

```python
# Read your manuscript
with open("my_manuscript.txt", "r") as f:
    manuscript_text = f.read()

# Get suggestions
suggestions = assistant.suggest_citations_for_manuscript(manuscript_text)

# Display formatted suggestions
print(assistant.format_citation_suggestions(suggestions))
```

## Automatic Updates

Your EndNote library syncs automatically every night at 1 AM via:
`/home/david/scripts/sync/sync_onedrive_endnote.sh`

After sync, you can re-run the indexer to pick up new papers:
```bash
python src/pdf_indexer.py
```

Only new or modified PDFs will be processed, making updates fast!

## Performance

- **Initial indexing**: ~10-20 PDFs/minute (depends on PDF size)
- **Incremental updates**: Only processes changed files
- **Search**: Near-instant (<1 second for most queries)
- **GPU acceleration**: Both embedding generation and LLM use your RTX GPUs

## Tips

1. **Regular re-indexing**: Run `python src/pdf_indexer.py` periodically after adding papers
2. **Citation quality**: More specific queries yield better citations
3. **Manuscript sections**: Process your manuscript in sections for more targeted citations
4. **Review suggestions**: Always review suggested citations - the AI helps find relevant papers but you verify relevance

## Troubleshooting

**"Collection not found" error:**
```bash
# Run the indexer first
python src/pdf_indexer.py
```

**Out of memory:**
```bash
# The embedding model is small and should fit easily
# If issues occur, check GPU memory: nvidia-smi
```

**Slow indexing:**
```bash
# Normal for first run with 6,800 PDFs
# Subsequent runs only process new files
```
