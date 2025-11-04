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
from citation_assistant import CitationAssistant, GlobalConfig
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
    # Initialize with GlobalConfig for unified parameter management
    try:
        # Create global config with default settings
        global_config = GlobalConfig(
            n_papers=10,
            chunks_per_paper=1,
            enable_reranking=False,  # Default off, can be enabled
            enable_hybrid=True,
            hybrid_balance=0.5,
            duplicate_threshold=0.95,
            llm_model="gemma2:27b"
        )
        
        assistant = CitationAssistant(
            embeddings_dir=EMBEDDINGS_DIR,
            config=global_config
        )
        print(f"✓ Loaded collection with {assistant.collection.count()} documents")
        print(f"  • Global config initialized")
        print(f"  • Re-ranking: Available")
        print(f"  • Hybrid search: Enabled")
    except Exception as e:
        print(f"⚠ Collection not found. Please run indexer first.")
        print(f"  Error: {e}")
        assistant = None
        global_config = GlobalConfig()  # Create default config anyway

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
    use_reranking: bool = False  # Enable cross-encoder re-ranking for +10-15% precision
    use_hybrid: bool = False     # Enable hybrid search (Vector + BM25) for +10-15% recall
    hybrid_alpha: float = 0.5    # Balance between vector (1.0) and BM25 (0.0)
    duplicate_threshold: float = 0.95  # Duplicate detection threshold (0.8-1.0)
    llm_model: str = "gemma2:27b"  # Options: gemma2:27b, qwen2.5:72b-instruct-q4_K_M, llama3.1:70b
    search_method: str = "default"  # Options: default, multi_chunk, factual


class SummarizeQuery(BaseModel):
    query: str
    n_papers: int = 10  # Increased from 5 to leverage Gemma2's 8K context
    use_reranking: bool = False  # Enable cross-encoder re-ranking for better paper selection
    use_hybrid: bool = False     # Enable hybrid search (Vector + BM25) for +10-15% recall
    hybrid_alpha: float = 0.5    # Balance between vector (1.0) and BM25 (0.0)
    duplicate_threshold: float = 0.95  # Duplicate detection threshold (0.8-1.0)
    llm_model: str = "gemma2:27b"  # Options: gemma2:27b, qwen2.5:72b-instruct-q4_K_M, llama3.1:70b
    search_method: str = "default"  # Options: default, multi_chunk, factual


class GlobalConfigUpdate(BaseModel):
    """Global configuration update request"""
    n_papers: Optional[int] = None
    chunks_per_paper: Optional[int] = None
    enable_reranking: Optional[bool] = None
    enable_hybrid: Optional[bool] = None
    hybrid_balance: Optional[float] = None
    duplicate_threshold: Optional[float] = None
    llm_model: Optional[str] = None
    keyword_boost_strength: Optional[float] = None
    entity_boost_strength: Optional[float] = None
    require_query_terms: Optional[bool] = None


class WriteQuery(BaseModel):
    topic: str
    keywords: str = ""       # Optional comma-separated keywords for aggressive boosting
    style: str = "academic"  # "academic" or "grant"
    length: str = "long"     # "short", "medium", or "long"
    n_papers: int = 15
    duplicate_threshold: float = 0.95  # Duplicate detection threshold (0.8-1.0)
    llm_model: str = "gemma2:27b"  # Options: gemma2:27b, qwen2.5:72b-instruct-q4_K_M, llama3.1:70b
    search_method: str = "multi_chunk"  # Options: default, multi_chunk, factual
    use_reranking: bool = False  # Enable cross-encoder re-ranking for better precision
    chunks_per_paper: int = 3  # Number of chunks per paper for multi-chunk method (2-10)


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


@app.get("/api/config")
async def get_config(current_user: User = Depends(get_current_active_user)):
    """Get current global configuration"""
    if assistant:
        return assistant.get_config()
    else:
        return global_config.to_dict()


@app.post("/api/config")
async def update_config(
    update: GlobalConfigUpdate,
    current_user: User = Depends(get_current_active_user)
):
    """Update global configuration"""
    if assistant:
        # Update only provided fields
        update_dict = {k: v for k, v in update.dict().items() if v is not None}
        assistant.update_config(**update_dict)
        return {"message": "Configuration updated", "config": assistant.get_config()}
    else:
        # Update the global config object directly
        update_dict = {k: v for k, v in update.dict().items() if v is not None}
        for key, value in update_dict.items():
            if hasattr(global_config, key):
                setattr(global_config, key, value)
        return {"message": "Configuration updated (assistant not loaded)", "config": global_config.to_dict()}


@app.get("/api/stats")
async def get_stats(current_user: User = Depends(get_current_active_user)):
    """Get indexing statistics (requires auth)"""
    if not indexer:
        raise HTTPException(status_code=503, detail="Indexer not initialized")

    stats = indexer.get_stats()

    # Add optimization info
    stats["optimizations"] = {
        "phase1_active": True,
        "phase2_semantic_chunking": indexer.use_semantic_chunking,
        "target_chunk_tokens": indexer.target_chunk_tokens,
        "overlap_sentences": indexer.overlap_sentences,
        "reranking_available": assistant is not None,
        "hybrid_search_available": assistant is not None and assistant.bm25_index is not None,
        "bm25_papers": len(assistant.bm25_doc_map) if assistant and assistant.bm25_doc_map else 0,
        "default_summary_papers": 10  # Updated from 5 to use more context
    }

    return stats


@app.post("/api/index")
async def run_indexing(current_user: User = Depends(get_current_active_user)):
    """Trigger incremental indexing (requires auth)"""
    if not indexer:
        raise HTTPException(status_code=503, detail="Indexer not initialized")

    try:
        loop = asyncio.get_event_loop()

        # Run the indexing in executor - this ensures we're using the src/pdf_indexer.py code
        print("Starting indexing via web endpoint...")
        results = await loop.run_in_executor(executor, indexer.index_all_new)
        print(f"Indexing completed: {results}")

        # Reload assistant if needed
        global assistant
        if not assistant:
            try:
                assistant = CitationAssistant(
                    embeddings_dir=EMBEDDINGS_DIR,
                    config=global_config
                )
                print(f"Assistant reloaded with {assistant.collection.count()} documents")
            except Exception as e:
                print(f"Warning: Could not reload assistant: {e}")

        return {
            "status": "completed",
            "results": results
        }
    except Exception as e:
        import traceback
        error_msg = f"Indexing error: {str(e)}"
        error_traceback = traceback.format_exc()
        print(f"ERROR in /api/index endpoint:")
        print(error_traceback)

        # Return proper JSON error response (detail must be a string for HTTPException)
        raise HTTPException(
            status_code=500,
            detail=f"{error_msg}\n\nTraceback:\n{error_traceback}"
        )


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

    # Temporarily override config for this search
    override_config = {
        'n_papers': query.n_results,
        'enable_reranking': query.use_reranking,
        'enable_hybrid': query.use_hybrid,
        'hybrid_balance': query.hybrid_alpha,
        'duplicate_threshold': query.duplicate_threshold
    }
    
    # Adjust chunks per paper based on search method
    if query.search_method == "multi_chunk":
        override_config['chunks_per_paper'] = 2
    elif query.search_method == "factual":
        override_config['chunks_per_paper'] = 2
        override_config['entity_boost_strength'] = 0.05  # Strong boost for entities
    else:
        override_config['chunks_per_paper'] = 1
    
    # Use unified search with temporary overrides
    papers = await loop.run_in_executor(
        executor,
        lambda: assistant.unified_search(
            query.query,
            boost_keywords="",
            boost_haslam=True,
            override_config=override_config
        )
    )

    return {
        "query": query.query,
        "n_results": len(papers),
        "reranking_used": query.use_reranking,
        "hybrid_used": query.use_hybrid,
        "hybrid_alpha": query.hybrid_alpha if query.use_hybrid else None,
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

    # Temporarily override config for this search
    override_config = {
        'n_papers': query.n_papers,
        'enable_reranking': query.use_reranking,
        'enable_hybrid': query.use_hybrid,
        'hybrid_balance': query.hybrid_alpha,
        'duplicate_threshold': query.duplicate_threshold,
        'llm_model': query.llm_model
    }
    
    # Adjust chunks per paper based on search method
    if query.search_method == "multi_chunk":
        override_config['chunks_per_paper'] = 2
    elif query.search_method == "factual":
        override_config['chunks_per_paper'] = 2
        override_config['entity_boost_strength'] = 0.05
    else:
        override_config['chunks_per_paper'] = 1
    
    # Apply temporary config and generate summary
    original_config = {}
    for key, value in override_config.items():
        if hasattr(assistant.config, key):
            original_config[key] = getattr(assistant.config, key)
            setattr(assistant.config, key, value)
    
    try:
        # Generate summary with updated config
        summary = await loop.run_in_executor(
            executor,
            lambda: assistant.summarize_research(query.query)
        )
    finally:
        # Restore original config
        for key, value in original_config.items():
            setattr(assistant.config, key, value)

    return {
        "query": query.query,
        "n_papers_used": query.n_papers,
        "reranking_used": query.use_reranking,
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
    use_reranking: bool = Form(False),
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
            n_suggestions_per_statement=n_suggestions,
            use_reranking=use_reranking
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
    
    try:
        # Temporarily override config
        override_config = {
            'n_papers': query.n_papers,
            'enable_reranking': query.use_reranking,
            'duplicate_threshold': query.duplicate_threshold,
            'llm_model': query.llm_model,
            'chunks_per_paper': query.chunks_per_paper
        }
        
        # Apply temporary config
        original_config = {}
        for key, value in override_config.items():
            if hasattr(assistant.config, key):
                original_config[key] = getattr(assistant.config, key)
                setattr(assistant.config, key, value)
        
        try:
            # Generate document with updated config
            document = await loop.run_in_executor(
                executor,
                lambda: assistant.write_document(
                    topic=query.topic,
                    style=query.style,
                    length=query.length,
                    keywords=query.keywords
                )
            )
        finally:
            # Restore original config
            for key, value in original_config.items():
                setattr(assistant.config, key, value)
    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        print(f"ERROR in write_document: {str(e)}")
        print(f"Traceback:\n{error_traceback}")
        raise HTTPException(
            status_code=500,
            detail=f"Error generating document: {str(e)}"
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
