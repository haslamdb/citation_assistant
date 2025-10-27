# Citation Assistant - Quick Start Guide

## System Overview

Your Citation Assistant is a RAG (Retrieval-Augmented Generation) system that:
1. **Indexes** your 6,800+ EndNote PDFs into a searchable database
2. **Searches** semantically (understands meaning, not just keywords)
3. **Summarizes** research findings across multiple papers
4. **Suggests** relevant citations for your manuscripts

## First Time Setup

### Wait for EndNote sync to complete

Check sync progress:
```bash
screen -r endnote_sync
# Or check the log:
tail -f /home/david/logs/endnote_initial_sync.log
```

Once complete (22.5 GB download), proceed to indexing.

### Index your EndNote library

```bash
mamba activate rag
cd /home/david/projects/citation_assistant
python cite.py index
```

**First run will take time** (~30-60 minutes for 6,800 PDFs). Subsequent runs are fast (only new papers).

The indexer will:
- Extract text from each PDF
- Create semantic embeddings using your GPU
- Store in ChromaDB vector database
- Track indexed files (only process changes next time)

## Daily Usage

### 1. Search for papers

```bash
python cite.py search "machine learning for microbiome analysis" -n 10 -v
```

Options:
- `-n NUM`: Number of results (default: 10)
- `-v`: Show text excerpts

### 2. Summarize research

```bash
python cite.py summarize "16S rRNA sequencing methods"
```

Gets AI summary synthesizing findings from your library.

### 3. Get citation suggestions

Create a text file with your manuscript section:
```bash
nano my_intro.txt
# Paste your introduction or methods section
```

Then:
```bash
python cite.py suggest my_intro.txt -o suggested_citations.txt
```

This will:
1. Identify claims needing citations
2. Find relevant papers from your library
3. Provide ranked suggestions with excerpts
4. Save to file if `-o` specified

## Adding New Papers

When you add papers to EndNote and sync:

1. **Sync happens automatically** (nightly at 1 AM)
   - Or manual: `bash /home/david/scripts/sync/sync_onedrive_endnote.sh`

2. **Re-index** to pick up new papers:
   ```bash
   python cite.py index
   ```

Only new/modified PDFs will be processed - takes seconds/minutes, not hours!

## Checking Status

```bash
# Show index statistics
python cite.py index --stats
```

Shows:
- Total indexed files
- Collection size (number of text chunks)
- Storage locations

## Tips for Best Results

### Search queries
- Be specific: "CRISPR-Cas9 off-target effects" > "gene editing"
- Use scientific terms from your field
- Try multiple phrasings if first search doesn't yield good results

### Summarization
- Focus on specific topics rather than broad areas
- Use more papers (`-n 10`) for comprehensive summaries
- Review papers list to verify relevance

### Citation suggestions
- Process manuscript in sections (intro, methods, discussion separately)
- More specific writing → better suggestions
- Always review suggested citations for relevance
- The AI finds *potentially* relevant papers - you verify fit

## Common Workflows

### Writing a manuscript

```bash
# 1. Draft your introduction
nano intro.txt

# 2. Get citation suggestions
python cite.py suggest intro.txt -o intro_citations.txt

# 3. Review suggestions
less intro_citations.txt

# 4. For any unclear topics, get research summary
python cite.py summarize "specific topic mentioned"
```

### Literature review

```bash
# Search for papers on topic
python cite.py search "my research topic" -n 20 -v > papers.txt

# Get AI summary
python cite.py summarize "my research topic" > summary.txt
```

### Keeping current

```bash
# Weekly: re-index after adding papers
python cite.py index

# Monthly: check stats
python cite.py index --stats
```

## Troubleshooting

**"Collection not found":**
```bash
# You haven't indexed yet
python cite.py index
```

**Slow first indexing:**
- Normal! 6,800 PDFs take time
- Leave it running (maybe in screen session)
- Future updates will be fast

**Poor search results:**
- Try different query phrasing
- Be more specific
- Check if you have papers on that topic: `python cite.py search "topic" -v`

**GPU out of memory:**
- Unlikely with your setup (48GB + 24GB GPUs)
- If it happens, restart and try again
- The embedding model is very small

## Next Steps

1. **Wait for EndNote sync to complete**
2. **Run initial indexing**: `python cite.py index`
3. **Try a search**: `python cite.py search "your research area"`
4. **Test citation suggestions** on a short manuscript excerpt

Enjoy your AI-powered citation assistant!
