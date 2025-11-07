#!/usr/bin/env python3
"""
Automated re-indexing script for running in screen/background
"""

import os
import sys
import shutil
import time
from pathlib import Path
from datetime import datetime

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from pdf_indexer import PDFIndexer

def main():
    # Configuration
    ENDNOTE_PDF_DIR = "/home/david/projects/EndNote_Library/PDF"
    EMBEDDINGS_DIR = "/fastpool/rag_embeddings"
    BACKUP_DIR = "/fastpool/rag_embeddings_backup"
    
    # Settings for automated run
    USE_LLM = True
    LLM_MODEL = "gemma2:27b"  # Using available model
    AUTO_CONFIRM = True  # Skip confirmations
    
    print("="*80)
    print("AUTOMATED PDF RE-INDEXING WITH ENHANCED METADATA")
    print("="*80)
    print(f"\nStarted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nSettings:")
    print(f"  PDF Directory: {ENDNOTE_PDF_DIR}")
    print(f"  Embeddings Directory: {EMBEDDINGS_DIR}")
    print(f"  LLM Model: {LLM_MODEL}")
    print(f"  LLM Metadata: {USE_LLM}")
    
    # Backup existing embeddings
    print("\n" + "-"*40)
    print("CREATING BACKUP")
    print("-"*40)
    
    if Path(EMBEDDINGS_DIR).exists():
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = Path(f"{BACKUP_DIR}_{timestamp}")
        
        print(f"Backing up to: {backup_path}")
        try:
            shutil.copytree(EMBEDDINGS_DIR, backup_path)
            print("✓ Backup created successfully")
        except Exception as e:
            print(f"⚠ Warning: Could not create backup: {e}")
            if not AUTO_CONFIRM:
                sys.exit(1)
    
    # Clear existing embeddings
    print("\n" + "-"*40)
    print("CLEARING EXISTING INDEX")
    print("-"*40)
    
    if Path(EMBEDDINGS_DIR).exists():
        print(f"Removing {EMBEDDINGS_DIR}...")
        shutil.rmtree(EMBEDDINGS_DIR)
        print("✓ Existing index cleared")
    
    # Create fresh indexer
    print("\n" + "-"*40)
    print("INITIALIZING INDEXER")
    print("-"*40)
    
    indexer = PDFIndexer(
        endnote_pdf_dir=ENDNOTE_PDF_DIR,
        embeddings_dir=EMBEDDINGS_DIR,
        use_semantic_chunking=True,
        use_llm_metadata=USE_LLM,
        llm_model=LLM_MODEL
    )
    
    print(f"✓ Indexer initialized with {LLM_MODEL}")
    
    # Start indexing
    print("\n" + "-"*40)
    print("STARTING INDEXING")
    print("-"*40)
    
    start_time = time.time()
    
    # Find all PDFs
    pdf_files = indexer.find_new_or_modified_pdfs()
    total_files = len(pdf_files)
    
    if total_files == 0:
        print("No PDF files found!")
        return
    
    print(f"\nFound {total_files} PDFs to index")
    
    # Estimate time
    if USE_LLM:
        time_per_file = 3  # seconds (approximate with LLM)
        estimated_time = (total_files * time_per_file) / 60
        print(f"Estimated time: {estimated_time:.1f} minutes")
    else:
        time_per_file = 0.5  # seconds (without LLM)
        estimated_time = (total_files * time_per_file) / 60
        print(f"Estimated time: {estimated_time:.1f} minutes")
    
    print("\nIndexing in progress...")
    print("Check progress with: screen -r reindex")
    print("-"*40)
    
    # Index all files
    try:
        stats = indexer.index_all_new()
        
        elapsed = time.time() - start_time
        elapsed_min = elapsed / 60
        
        print("\n" + "="*80)
        print("INDEXING COMPLETE")
        print("="*80)
        print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"\nStatistics:")
        print(f"  Files processed: {stats['new_files']}")
        print(f"  Total chunks: {stats['total_chunks']}")
        print(f"  Failed files: {stats.get('failed_files', 0)}")
        print(f"  Collection size: {stats['collection_size']}")
        print(f"  Time elapsed: {elapsed_min:.1f} minutes")
        
        if stats['new_files'] > 0:
            print(f"  Avg time per file: {elapsed/stats['new_files']:.1f} seconds")
        
        print("\n✓ Re-indexing completed successfully!")
        
        # Quick test
        print("\n" + "-"*40)
        print("TESTING NEW INDEX")
        print("-"*40)
        
        from citation_assistant import CitationAssistant
        
        assistant = CitationAssistant(
            embeddings_dir=EMBEDDINGS_DIR,
            enable_reranking=False
        )
        
        test_query = "microbiome"
        papers = assistant.search_papers(test_query, n_results=1)
        
        if papers:
            paper = papers[0]
            print(f"\n✓ Test successful!")
            print(f"  Sample result: {paper['filename']}")
            if paper.get('publication_year'):
                print(f"  Year: {paper['publication_year']}")
            if paper.get('category'):
                print(f"  Category: {paper['category']}")
        
        print("\n" + "="*80)
        print("✓ All systems operational!")
        
    except KeyboardInterrupt:
        print("\n\n⚠ Indexing interrupted")
        elapsed = time.time() - start_time
        print(f"Time elapsed: {elapsed/60:.1f} minutes")
    except Exception as e:
        print(f"\n\n❌ Error during indexing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()