#!/usr/bin/env python3
"""
Re-index all PDFs with enhanced metadata extraction
Includes: publication year, categories, keywords, study type, etc.
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
    
    print("="*80)
    print("PDF RE-INDEXING WITH ENHANCED METADATA")
    print("="*80)
    print(f"\nThis will re-index all PDFs with:")
    print("  • Publication year extraction")
    print("  • LLM-based metadata (categories, keywords, study type, etc.)")
    print("  • Recency boosting support")
    print("  • Enhanced search filtering")
    print("\nSettings:")
    print(f"  PDF Directory: {ENDNOTE_PDF_DIR}")
    print(f"  Embeddings Directory: {EMBEDDINGS_DIR}")
    print(f"  Backup Directory: {BACKUP_DIR}")
    
    # Prompt for LLM selection
    print("\n" + "-"*40)
    print("SELECT LLM MODEL FOR METADATA EXTRACTION")
    print("-"*40)
    print("1. llama3.2:3b (Fast, good quality)")
    print("2. gemma2:9b (Slower, better quality)")
    print("3. gemma2:27b (Slowest, best quality)")
    print("4. Skip LLM metadata (only extract year)")
    
    choice = input("\nEnter choice (1-4) [default: 1]: ").strip() or "1"
    
    llm_model = "llama3.2:3b"
    use_llm = True
    
    if choice == "2":
        llm_model = "gemma2:9b"
    elif choice == "3":
        llm_model = "gemma2:27b"
    elif choice == "4":
        use_llm = False
        print("\n⚠ LLM metadata extraction disabled")
    
    if use_llm:
        print(f"\n✓ Using {llm_model} for metadata extraction")
        print("  Note: This will be slower but provides richer metadata")
    
    # Backup existing embeddings
    print("\n" + "-"*40)
    print("CREATING BACKUP")
    print("-"*40)
    
    if Path(EMBEDDINGS_DIR).exists():
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = Path(f"{BACKUP_DIR}_{timestamp}")
        
        print(f"Backing up existing embeddings to: {backup_path}")
        try:
            shutil.copytree(EMBEDDINGS_DIR, backup_path)
            print("✓ Backup created successfully")
        except Exception as e:
            print(f"⚠ Warning: Could not create backup: {e}")
            proceed = input("Continue anyway? (y/n): ").strip().lower()
            if proceed != 'y':
                print("Aborted.")
                return
    
    # Clear existing embeddings
    print("\n" + "-"*40)
    print("CLEARING EXISTING INDEX")
    print("-"*40)
    
    proceed = input("This will DELETE the current index. Continue? (y/n): ").strip().lower()
    if proceed != 'y':
        print("Aborted.")
        return
    
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
        use_llm_metadata=use_llm,
        llm_model=llm_model
    )
    
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
    
    if use_llm:
        # Estimate time
        time_per_file = 3  # seconds (approximate with LLM)
        estimated_time = (total_files * time_per_file) / 60
        print(f"Estimated time: {estimated_time:.1f} minutes")
    else:
        time_per_file = 0.5  # seconds (without LLM)
        estimated_time = (total_files * time_per_file) / 60
        print(f"Estimated time: {estimated_time:.1f} minutes")
    
    print("\nPress Ctrl+C to abort at any time")
    print("-"*40)
    
    # Index all files
    try:
        stats = indexer.index_all_new()
        
        elapsed = time.time() - start_time
        elapsed_min = elapsed / 60
        
        print("\n" + "="*80)
        print("INDEXING COMPLETE")
        print("="*80)
        print(f"\nStatistics:")
        print(f"  Files processed: {stats['new_files']}")
        print(f"  Total chunks: {stats['total_chunks']}")
        print(f"  Failed files: {stats.get('failed_files', 0)}")
        print(f"  Collection size: {stats['collection_size']}")
        print(f"  Time elapsed: {elapsed_min:.1f} minutes")
        
        if stats['new_files'] > 0:
            print(f"  Avg time per file: {elapsed/stats['new_files']:.1f} seconds")
        
        print("\n✓ Re-indexing completed successfully!")
        
        # Test the new index
        print("\n" + "-"*40)
        print("TESTING NEW INDEX")
        print("-"*40)
        
        from citation_assistant import CitationAssistant
        
        assistant = CitationAssistant(
            embeddings_dir=EMBEDDINGS_DIR,
            enable_reranking=False
        )
        
        test_query = "microbiome"
        print(f"\nTest query: '{test_query}'")
        papers = assistant.search_papers(test_query, n_results=3)
        
        print("\nTop 3 results:")
        for i, paper in enumerate(papers[:3], 1):
            print(f"\n{i}. {paper['filename']}")
            if paper.get('publication_year'):
                print(f"   Year: {paper['publication_year']}")
            if paper.get('category'):
                print(f"   Category: {paper['category']}")
            if paper.get('llm_keywords'):
                keywords = paper['llm_keywords'][:50] + "..." if len(paper['llm_keywords']) > 50 else paper['llm_keywords']
                print(f"   Keywords: {keywords}")
            if paper.get('study_type'):
                print(f"   Study type: {paper['study_type']}")
            if paper.get('impact'):
                print(f"   Impact: {paper['impact']}")
        
        print("\n" + "="*80)
        print("✓ All systems operational!")
        print("\nYou can now use the enhanced search features:")
        print("  • Recency boosting (boost_recency=True)")
        print("  • Category filtering (filter_category='microbiology')")
        print("  • Study type filtering (filter_study_type='clinical_trial')")
        print("  • Impact filtering (filter_impact='high')")
        print("  • Metadata keyword boosting")
        
    except KeyboardInterrupt:
        print("\n\n⚠ Indexing interrupted by user")
        elapsed = time.time() - start_time
        print(f"Time elapsed: {elapsed/60:.1f} minutes")
        print("\nPartial index may be available.")
        print("Run this script again to continue indexing.")
    except Exception as e:
        print(f"\n\n❌ Error during indexing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()