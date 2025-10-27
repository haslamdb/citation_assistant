#!/usr/bin/env python3
"""
Citation Assistant Web API Server
FastAPI-based web service for remote access to citation assistant
"""

import sys
from pathlib import Path
from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from pdf_indexer import PDFIndexer
from citation_assistant import CitationAssistant

# Configuration
ENDNOTE_PDF_DIR = "/home/david/projects/EndNote_Library/PDF"
EMBEDDINGS_DIR = "/fastpool/rag_embeddings"

# Initialize FastAPI app
app = FastAPI(
    title="Citation Assistant API",
    description="AI-powered citation suggestions using your EndNote library",
    version="1.0.0"
)

# Add CORS middleware for remote access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to specific domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Thread pool for CPU-intensive tasks
executor = ThreadPoolExecutor(max_workers=2)

# Global instances (loaded on startup)
indexer = None
assistant = None


# Pydantic models for API
class SearchQuery(BaseModel):
    query: str
    n_results: int = 10


class SummarizeQuery(BaseModel):
    query: str
    n_papers: int = 5


class CitationSuggestion(BaseModel):
    statement: str
    suggested_papers: List[dict]
    confidence: float


class IndexStats(BaseModel):
    total_indexed_files: int
    collection_size: int
    embeddings_dir: str
    collection_name: str


@app.on_event("startup")
async def startup_event():
    """Initialize citation assistant on server startup"""
    global indexer, assistant

    print("Initializing Citation Assistant...")

    # Check if collection exists
    try:
        assistant = CitationAssistant(embeddings_dir=EMBEDDINGS_DIR)
        print(f"✓ Loaded collection with {assistant.collection.count()} documents")
    except Exception as e:
        print(f"⚠ Collection not found. Please run indexer first.")
        print(f"  Error: {e}")
        assistant = None

    # Initialize indexer (doesn't require existing collection)
    indexer = PDFIndexer(
        endnote_pdf_dir=ENDNOTE_PDF_DIR,
        embeddings_dir=EMBEDDINGS_DIR
    )
    print("✓ Indexer initialized")

    print("Server ready!")


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve web interface"""
    html_file = Path(__file__).parent / "web" / "index.html"
    if html_file.exists():
        return html_file.read_text()
    else:
        return """
        <html>
            <head><title>Citation Assistant</title></head>
            <body>
                <h1>Citation Assistant API</h1>
                <p>API is running. Visit <a href="/docs">/docs</a> for API documentation.</p>
                <p>Web interface not found. Please create web/index.html</p>
            </body>
        </html>
        """


@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "indexer_ready": indexer is not None,
        "assistant_ready": assistant is not None,
        "collection_size": assistant.collection.count() if assistant else 0
    }


@app.get("/api/stats", response_model=IndexStats)
async def get_stats():
    """Get indexing statistics"""
    if not indexer:
        raise HTTPException(status_code=503, detail="Indexer not initialized")

    stats = indexer.get_stats()
    return IndexStats(**stats)


@app.post("/api/index")
async def run_indexing():
    """Trigger incremental indexing of EndNote library"""
    if not indexer:
        raise HTTPException(status_code=503, detail="Indexer not initialized")

    # Run indexing in background thread
    loop = asyncio.get_event_loop()
    results = await loop.run_in_executor(executor, indexer.index_all_new)

    # Reload assistant if it wasn't initialized
    global assistant
    if not assistant:
        try:
            assistant = CitationAssistant(embeddings_dir=EMBEDDINGS_DIR)
        except:
            pass

    return {
        "status": "completed",
        "results": results
    }


@app.post("/api/search")
async def search_papers(query: SearchQuery):
    """Search for relevant papers"""
    if not assistant:
        raise HTTPException(
            status_code=503,
            detail="Assistant not initialized. Please run indexing first."
        )

    # Run search in background thread
    loop = asyncio.get_event_loop()
    papers = await loop.run_in_executor(
        executor,
        lambda: assistant.search_papers(query.query, n_results=query.n_results)
    )

    return {
        "query": query.query,
        "n_results": len(papers),
        "papers": papers
    }


@app.post("/api/summarize")
async def summarize_research(query: SummarizeQuery):
    """Generate research summary on a topic"""
    if not assistant:
        raise HTTPException(
            status_code=503,
            detail="Assistant not initialized. Please run indexing first."
        )

    # Run summarization in background thread
    loop = asyncio.get_event_loop()
    summary = await loop.run_in_executor(
        executor,
        lambda: assistant.summarize_research(query.query, n_papers=query.n_papers)
    )

    return {
        "query": query.query,
        "summary": summary
    }


@app.post("/api/suggest")
async def suggest_citations(manuscript: UploadFile = File(...), n_suggestions: int = Form(3)):
    """Suggest citations for uploaded manuscript"""
    if not assistant:
        raise HTTPException(
            status_code=503,
            detail="Assistant not initialized. Please run indexing first."
        )

    # Read manuscript
    try:
        manuscript_text = (await manuscript.read()).decode('utf-8')
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading file: {e}")

    # Run citation suggestion in background thread
    loop = asyncio.get_event_loop()
    suggestions = await loop.run_in_executor(
        executor,
        lambda: assistant.suggest_citations_for_manuscript(
            manuscript_text,
            n_suggestions_per_statement=n_suggestions
        )
    )

    return {
        "filename": manuscript.filename,
        "n_suggestions": len(suggestions),
        "suggestions": suggestions
    }


@app.post("/api/suggest/text")
async def suggest_citations_text(
    text: str = Form(...),
    n_suggestions: int = Form(3)
):
    """Suggest citations for text content"""
    if not assistant:
        raise HTTPException(
            status_code=503,
            detail="Assistant not initialized. Please run indexing first."
        )

    # Run citation suggestion in background thread
    loop = asyncio.get_event_loop()
    suggestions = await loop.run_in_executor(
        executor,
        lambda: assistant.suggest_citations_for_manuscript(
            text,
            n_suggestions_per_statement=n_suggestions
        )
    )

    return {
        "n_suggestions": len(suggestions),
        "suggestions": suggestions
    }


if __name__ == "__main__":
    import uvicorn

    # Run server
    print("Starting Citation Assistant Server...")
    print("Access at: http://localhost:8000")
    print("API docs at: http://localhost:8000/docs")

    uvicorn.run(
        app,
        host="0.0.0.0",  # Listen on all interfaces for remote access
        port=8000,
        log_level="info"
    )
