#!/usr/bin/env python3
"""
Citation Assistant Web API Server (SECURE)
FastAPI-based web service with JWT authentication
"""

import sys
import os
from pathlib import Path
from typing import List, Optional
from datetime import timedelta
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends, status
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from pydantic import BaseModel
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from pdf_indexer import PDFIndexer
from citation_assistant import CitationAssistant
from auth import (
    authenticate_user,
    create_access_token,
    get_current_active_user,
    User,
    Token,
    ACCESS_TOKEN_EXPIRE_MINUTES
)

# Configuration
ENDNOTE_PDF_DIR = "/home/david/projects/EndNote_Library/PDF"
EMBEDDINGS_DIR = "/fastpool/rag_embeddings"

# Security: Allowed hosts (add your IP addresses)
# Note: TrustedHostMiddleware doesn't support partial wildcards like 192.168.1.*
# For local network, we'll use a custom approach or just list specific IPs
ALLOWED_HOSTS = [
    "localhost",
    "127.0.0.1",
    "192.168.1.163",  # Your server IP
    "*.local",        # Local network hostnames
]

# Thread pool for CPU-intensive tasks
executor = ThreadPoolExecutor(max_workers=2)

# Global instances (loaded on startup)
indexer = None
assistant = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan: startup and shutdown"""
    # Startup
    global indexer, assistant

    print("=" * 60)
    print("Starting Secure Citation Assistant Server")
    print("=" * 60)

    # Check if collection exists
    try:
        assistant = CitationAssistant(embeddings_dir=EMBEDDINGS_DIR)
        print(f"✓ Loaded collection with {assistant.collection.count()} documents")
    except Exception as e:
        print(f"⚠ Collection not found. Please run indexer first.")
        print(f"  Error: {e}")
        assistant = None

    # Initialize indexer with Phase 2 semantic chunking
    indexer = PDFIndexer(
        endnote_pdf_dir=ENDNOTE_PDF_DIR,
        embeddings_dir=EMBEDDINGS_DIR,
        use_semantic_chunking=True,      # Phase 2 optimization
        target_chunk_tokens=512,         # Use full PubMedBERT capacity
        overlap_sentences=2              # Semantic overlap
    )
    print("✓ Indexer initialized")
    print(f"  • Semantic chunking: {indexer.use_semantic_chunking}")
    print(f"  • Target chunk size: {indexer.target_chunk_tokens} tokens")
    print(f"  • Sentence overlap: {indexer.overlap_sentences}")

    print("\n⚠ SECURITY ENABLED ⚠")
    print("  - JWT Authentication required")
    print("  - Trusted host middleware active")
    print("  - Password hashing (bcrypt)")
    print("\nServer ready!")
    print("=" * 60)

    yield  # Server runs

    # Shutdown
    executor.shutdown(wait=True)
    print("\nShutting down Citation Assistant Server...")


# Initialize FastAPI app with lifespan
app = FastAPI(
    title="Citation Assistant API (Secure)",
    description="AI-powered citation suggestions with authentication",
    version="2.0.0",
    lifespan=lifespan
)

# Security middleware - commented out for local network flexibility
# For production with fixed IPs, uncomment and specify exact hosts
# app.add_middleware(
#     TrustedHostMiddleware,
#     allowed_hosts=ALLOWED_HOSTS
# )

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://192.168.1.163:8000", "http://localhost:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic models
class SearchQuery(BaseModel):
    query: str
    n_results: int = 10


class SummarizeQuery(BaseModel):
    query: str
    n_papers: int = 5


class WriteQuery(BaseModel):
    topic: str
    keywords: str = ""       # Optional comma-separated keywords for aggressive boosting
    style: str = "academic"  # "academic" or "grant"
    length: str = "long"     # "short", "medium", or "long"
    n_papers: int = 15


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve secure web interface"""
    html_file = Path(__file__).parent / "web" / "index_secure.html"
    if html_file.exists():
        return FileResponse(html_file)
    else:
        return """
        <html>
            <head><title>Citation Assistant (Secure)</title></head>
            <body>
                <h1>🔒 Citation Assistant API (Secure)</h1>
                <p>Authentication required. Visit <a href="/docs">/docs</a> for API documentation.</p>
                <p>Web interface not found at web/index_secure.html</p>
            </body>
        </html>
        """


@app.post("/api/token", response_model=Token)
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    """
    OAuth2 compatible token login
    Get access token with username and password
    """
    user = authenticate_user(form_data.username, form_data.password)

    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )

    return {"access_token": access_token, "token_type": "bearer"}


@app.get("/api/health")
async def health_check():
    """Public health check endpoint (no auth required)"""
    return {
        "status": "healthy",
        "version": "2.0.0",
        "security": "enabled",
        "indexer_ready": indexer is not None,
        "assistant_ready": assistant is not None,
        "collection_size": assistant.collection.count() if assistant else 0
    }


@app.get("/api/me", response_model=User)
async def read_users_me(current_user: User = Depends(get_current_active_user)):
    """Get current user info"""
    return current_user


@app.get("/api/stats")
async def get_stats(current_user: User = Depends(get_current_active_user)):
    """Get indexing statistics (requires auth)"""
    if not indexer:
        raise HTTPException(status_code=503, detail="Indexer not initialized")

    stats = indexer.get_stats()

    # Add Phase 2 optimization info
    stats["optimizations"] = {
        "phase1_active": True,
        "phase2_semantic_chunking": indexer.use_semantic_chunking,
        "target_chunk_tokens": indexer.target_chunk_tokens,
        "overlap_sentences": indexer.overlap_sentences
    }

    return stats


@app.post("/api/index")
async def run_indexing(current_user: User = Depends(get_current_active_user)):
    """Trigger incremental indexing (requires auth)"""
    if not indexer:
        raise HTTPException(status_code=503, detail="Indexer not initialized")

    loop = asyncio.get_event_loop()
    results = await loop.run_in_executor(executor, indexer.index_all_new)

    # Reload assistant if needed
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
async def search_papers(
    query: SearchQuery,
    current_user: User = Depends(get_current_active_user)
):
    """Search for relevant papers (requires auth)"""
    if not assistant:
        raise HTTPException(
            status_code=503,
            detail="Assistant not initialized. Please run indexing first."
        )

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
async def summarize_research(
    query: SummarizeQuery,
    current_user: User = Depends(get_current_active_user)
):
    """Generate research summary (requires auth)"""
    if not assistant:
        raise HTTPException(
            status_code=503,
            detail="Assistant not initialized. Please run indexing first."
        )

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
async def suggest_citations(
    manuscript: UploadFile = File(...),
    n_suggestions: int = Form(3),
    current_user: User = Depends(get_current_active_user)
):
    """Suggest citations for uploaded manuscript (requires auth)"""
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
    n_suggestions: int = Form(3),
    current_user: User = Depends(get_current_active_user)
):
    """Suggest citations for text content (requires auth)"""
    if not assistant:
        raise HTTPException(
            status_code=503,
            detail="Assistant not initialized. Please run indexing first."
        )

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


@app.post("/api/write")
async def write_document(
    query: WriteQuery,
    current_user: User = Depends(get_current_active_user)
):
    """Write a comprehensive document on a topic (requires auth)"""
    if not assistant:
        raise HTTPException(
            status_code=503,
            detail="Assistant not initialized. Please run indexing first."
        )

    loop = asyncio.get_event_loop()
    document = await loop.run_in_executor(
        executor,
        lambda: assistant.write_document(
            topic=query.topic,
            style=query.style,
            length=query.length,
            n_papers=query.n_papers,
            keywords=query.keywords
        )
    )

    return {
        "topic": query.topic,
        "style": query.style,
        "length": query.length,
        "document": document
    }


if __name__ == "__main__":
    import uvicorn

    print("\nStarting Citation Assistant Server (SECURE)")
    print("Access at: http://192.168.1.163:8000")
    print("API docs at: http://192.168.1.163:8000/docs")
    print("\n⚠ Authentication required for all API endpoints")
    print("Create users with: python manage_users.py\n")

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
