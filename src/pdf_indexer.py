#!/usr/bin/env python3
"""
PDF Indexer for Citation Assistant
Incrementally indexes PDF files from EndNote library into ChromaDB
"""

import os
import sys
import json
import hashlib
from pathlib import Path
from typing import List, Dict, Tuple

# Configure model locations before importing models
try:
    sys.path.insert(0, str(Path(__file__).parent.parent))
    import model_config  # Sets environment variables
except ImportError:
    pass  # Fall back to default locations

import pymupdf  # PyMuPDF
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from tqdm import tqdm
import re
import ollama


class PDFIndexer:
    """Incrementally index PDFs into ChromaDB with change tracking"""

    def __init__(
        self,
        endnote_pdf_dir: str,
        embeddings_dir: str,
        collection_name: str = "research_papers",
        embedding_model: str = "pritamdeka/S-PubMedBert-MS-MARCO",
        use_semantic_chunking: bool = True,
        target_chunk_tokens: int = 512,
        overlap_sentences: int = 2,
        use_llm_metadata: bool = True,
        llm_model: str = "gemma2:27b"
    ):
        self.endnote_pdf_dir = Path(endnote_pdf_dir)
        self.embeddings_dir = Path(embeddings_dir)
        self.collection_name = collection_name
        self.use_semantic_chunking = use_semantic_chunking
        self.target_chunk_tokens = target_chunk_tokens
        self.overlap_sentences = overlap_sentences
        self.use_llm_metadata = use_llm_metadata
        self.llm_model = llm_model

        # Track indexed files
        self.index_state_file = self.embeddings_dir / "index_state.json"
        self.indexed_files = self._load_index_state()

        # Initialize embedding model
        print(f"Loading embedding model: {embedding_model}")
        self.embedding_model = SentenceTransformer(embedding_model)

        # Initialize ChromaDB
        self.embeddings_dir.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(
            path=str(self.embeddings_dir),
            settings=Settings(anonymized_telemetry=False)
        )
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "EndNote research papers"}
        )

    def _load_index_state(self) -> Dict[str, str]:
        """Load state of previously indexed files"""
        if self.index_state_file.exists():
            with open(self.index_state_file, 'r') as f:
                return json.load(f)
        return {}

    def _save_index_state(self):
        """Save state of indexed files"""
        self.embeddings_dir.mkdir(parents=True, exist_ok=True)
        with open(self.index_state_file, 'w') as f:
            json.dump(self.indexed_files, f, indent=2)

    def _get_file_hash(self, filepath: Path) -> str:
        """Get MD5 hash of file for change detection"""
        hash_md5 = hashlib.md5()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    def _extract_text_from_pdf(self, pdf_path: Path) -> str:
        """Extract text from PDF using PyMuPDF"""
        try:
            doc = pymupdf.open(pdf_path)
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()
            return text.strip()
        except Exception as e:
            print(f"Error extracting text from {pdf_path.name}: {e}")
            return ""
    
    def _extract_publication_year(self, pdf_path: Path, text: str) -> int:
        """Extract publication year from PDF filename or text
        
        Args:
            pdf_path: Path to the PDF file
            text: Extracted text from the PDF
            
        Returns:
            Publication year (1900-2099) or 0 if not found
        """
        import datetime
        current_year = datetime.datetime.now().year
        
        # First, try to extract year from filename
        filename = pdf_path.name
        # Look for 4-digit years in filename (1900-2099)
        year_matches = re.findall(r'\b(19\d{2}|20\d{2})\b', filename)
        for year_str in year_matches:
            year = int(year_str)
            # Validate year is reasonable (not future, not too old)
            if 1900 <= year <= current_year + 1:  # Allow 1 year in future for preprints
                return year
        
        # If not found in filename, try to extract from first 2000 characters of text
        # Look for common publication patterns
        if text:
            text_sample = text[:2000]
            
            # Common patterns: "Published: 2023", "© 2023", "Received: ... Accepted: ... Published: 2023"
            # "Volume 45, 2023", "2023;", "(2023)"
            patterns = [
                r'(?:Published|Publication date|Pub Date|Copyright|©)[:\s]+.*?\b(19\d{2}|20\d{2})\b',
                r'\b(19\d{2}|20\d{2})\b[;,]\s*(?:Volume|\d+\s*\()',  # Year followed by volume
                r'(?:Accepted|Published online)[:\s]+.*?\b(19\d{2}|20\d{2})\b',
                r'\(?\b(19\d{2}|20\d{2})\b\)?(?:\s*;|\s*\))',  # Year in parentheses or followed by semicolon
            ]
            
            for pattern in patterns:
                matches = re.findall(pattern, text_sample, re.IGNORECASE)
                for year_str in matches:
                    year = int(year_str)
                    if 1900 <= year <= current_year + 1:
                        return year
            
            # Last resort: find any 4-digit year in reasonable range
            year_matches = re.findall(r'\b(19\d{2}|20\d{2})\b', text_sample)
            valid_years = []
            for year_str in year_matches:
                year = int(year_str)
                if 1900 <= year <= current_year + 1:
                    valid_years.append(year)
            
            # Return the most recent valid year found (likely the publication year)
            if valid_years:
                return max(valid_years)
        
        return 0  # No year found
    
    def _extract_llm_metadata(self, text: str, pdf_path: Path) -> Dict:
        """Extract rich metadata from PDF text using LLM
        
        Args:
            text: Extracted text from the PDF (first ~3000 chars for efficiency)
            pdf_path: Path to the PDF file
            
        Returns:
            Dictionary with extracted metadata
        """
        if not self.use_llm_metadata:
            return {}
        
        # Use first 3000 characters for metadata extraction (enough for abstract + intro)
        text_sample = text[:3000] if text else ""
        
        if not text_sample:
            return {}
        
        try:
            prompt = f"""Analyze this research paper excerpt and extract the following metadata. Be concise and accurate.

Paper excerpt:
{text_sample}

Please provide the following information in a structured format:

1. CATEGORY: Choose ONE primary category from: 
   [microbiology, immunology, genomics, bioinformatics, clinical_medicine, epidemiology, pharmacology, molecular_biology, cell_biology, biochemistry, neuroscience, public_health, infectious_diseases, cancer_research, computational_biology, other]

2. KEYWORDS: List 5-8 specific keywords or phrases that best describe this paper's content (comma-separated)

3. STUDY_TYPE: Choose ONE from:
   [clinical_trial, cohort_study, case_control, systematic_review, meta_analysis, experimental, computational, observational, case_report, methodology, review, opinion, other]

4. ORGANISM: Main organism(s) studied (e.g., human, mouse, E. coli, SARS-CoV-2, or 'multiple')

5. DISEASE_FOCUS: Primary disease or condition studied (if applicable, otherwise 'none')

6. METHODS: List 2-3 main methodological approaches used (e.g., RNA-seq, CRISPR, flow cytometry, etc.)

7. IMPACT: Rate the potential impact/significance (high, medium, low) based on novelty and importance

Format your response EXACTLY as follows (use these exact labels):
CATEGORY: [your answer]
KEYWORDS: [your answer]
STUDY_TYPE: [your answer]
ORGANISM: [your answer]
DISEASE_FOCUS: [your answer]
METHODS: [your answer]
IMPACT: [your answer]"""

            # Query Ollama for metadata extraction
            response = ollama.chat(
                model=self.llm_model,
                messages=[{'role': 'user', 'content': prompt}],
                options={'temperature': 0.1}  # Low temperature for consistent extraction
            )
            
            # Parse the response
            metadata = {}
            response_text = response['message']['content']
            
            # Extract each field using regex
            patterns = {
                'category': r'CATEGORY:\s*([^\n]+)',
                'keywords': r'KEYWORDS:\s*([^\n]+)',
                'study_type': r'STUDY_TYPE:\s*([^\n]+)',
                'organism': r'ORGANISM:\s*([^\n]+)',
                'disease_focus': r'DISEASE_FOCUS:\s*([^\n]+)',
                'methods': r'METHODS:\s*([^\n]+)',
                'impact': r'IMPACT:\s*([^\n]+)'
            }
            
            for field, pattern in patterns.items():
                match = re.search(pattern, response_text, re.IGNORECASE)
                if match:
                    value = match.group(1).strip()
                    # Clean up the value
                    value = value.strip('[]').strip()
                    
                    # For keywords and methods, split into list
                    if field in ['keywords', 'methods']:
                        # Split by comma and clean each item
                        items = [item.strip() for item in value.split(',')]
                        metadata[field] = ', '.join(items[:8])  # Limit to 8 items
                    else:
                        # Normalize category values
                        if field == 'category':
                            value = value.lower().replace(' ', '_').replace('-', '_')
                        elif field == 'study_type':
                            value = value.lower().replace(' ', '_').replace('-', '_')
                        elif field == 'impact':
                            value = value.lower()
                            if value not in ['high', 'medium', 'low']:
                                value = 'medium'  # Default to medium if unclear
                        metadata[field] = value
            
            return metadata
            
        except Exception as e:
            print(f"  Warning: Could not extract LLM metadata: {e}")
            return {}

    def _chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """Split text into overlapping chunks for better retrieval (DEPRECATED - use _chunk_text_semantic)"""
        if not text:
            return []

        chunks = []
        start = 0
        text_len = len(text)

        while start < text_len:
            end = start + chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start += chunk_size - overlap

        return chunks

    def _simple_sentence_tokenize(self, text: str) -> List[str]:
        """
        Simple sentence tokenizer using regex (no external dependencies)
        Handles common sentence boundaries in scientific text
        """
        # Replace newlines with spaces but preserve paragraph breaks
        text = re.sub(r'\n\s*\n', ' <PARA> ', text)
        text = re.sub(r'\n', ' ', text)

        # Split on sentence boundaries
        # This regex handles common abbreviations and citations
        sentences = re.split(
            r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<![A-Z]\.)(?<=\.|\?|\!)\s+(?=[A-Z])',
            text
        )

        # Clean up and filter
        sentences = [s.strip().replace('<PARA>', '\n\n') for s in sentences if s.strip()]
        return sentences

    def _count_tokens_approx(self, text: str) -> int:
        """
        Approximate token count without loading full tokenizer
        Scientific text averages ~4 chars per token for BERT models
        """
        return len(text) // 4

    def _chunk_text_semantic(
        self,
        text: str,
        target_tokens: int = None,
        overlap_sentences: int = None
    ) -> List[str]:
        """
        Split text into semantic chunks on sentence boundaries

        Args:
            text: Text to chunk
            target_tokens: Target tokens per chunk (default: use instance setting)
            overlap_sentences: Number of sentences to overlap (default: use instance setting)

        Returns:
            List of text chunks with sentence-level boundaries
        """
        if not text:
            return []

        if target_tokens is None:
            target_tokens = self.target_chunk_tokens
        if overlap_sentences is None:
            overlap_sentences = self.overlap_sentences

        # Tokenize into sentences
        sentences = self._simple_sentence_tokenize(text)

        if not sentences:
            return []

        chunks = []
        current_chunk = []
        current_tokens = 0

        for i, sentence in enumerate(sentences):
            sent_tokens = self._count_tokens_approx(sentence)

            # If single sentence exceeds target, include it anyway (don't split mid-sentence)
            if sent_tokens > target_tokens and not current_chunk:
                chunks.append(sentence)
                continue

            # If adding this sentence exceeds target and we have content, save chunk
            if current_tokens + sent_tokens > target_tokens and current_chunk:
                chunks.append(" ".join(current_chunk))

                # Start new chunk with overlap (last N sentences)
                if len(current_chunk) > overlap_sentences:
                    current_chunk = current_chunk[-overlap_sentences:]
                    current_tokens = sum(self._count_tokens_approx(s) for s in current_chunk)
                else:
                    current_chunk = []
                    current_tokens = 0

            current_chunk.append(sentence)
            current_tokens += sent_tokens

        # Add final chunk
        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

    def find_new_or_modified_pdfs(self) -> List[Tuple[Path, str]]:
        """Find PDFs that are new or have been modified"""
        new_or_modified = []

        if not self.endnote_pdf_dir.exists():
            print(f"EndNote PDF directory not found: {self.endnote_pdf_dir}")
            return new_or_modified

        # Recursively find all PDFs (including OneDrive sync conflict files with ..path* suffix)
        pdf_files = list(self.endnote_pdf_dir.rglob("*.pdf"))
        pdf_files.extend(self.endnote_pdf_dir.rglob("*.pdf..path*"))
        print(f"Found {len(pdf_files)} PDF files in EndNote library")

        for pdf_path in pdf_files:
            # Skip if file doesn't actually exist (broken symlink)
            if not pdf_path.exists():
                print(f"  Skipping (not found): {pdf_path.name}")
                continue

            # Get relative path and normalize file key (strip ..path* suffix from OneDrive conflicts)
            relative_path = str(pdf_path.relative_to(self.endnote_pdf_dir))
            # Remove OneDrive sync conflict suffix for consistent indexing
            file_key = relative_path.split('..path')[0] if '..path' in relative_path else relative_path

            try:
                current_hash = self._get_file_hash(pdf_path)
            except Exception as e:
                print(f"  Skipping (error hashing): {pdf_path.name} - {e}")
                continue

            # Check if new or modified
            if file_key not in self.indexed_files or self.indexed_files[file_key] != current_hash:
                new_or_modified.append((pdf_path, file_key))

        return new_or_modified

    def index_pdf(self, pdf_path: Path, file_key: str) -> int:
        """Index a single PDF file, returns number of chunks indexed"""
        print(f"Indexing: {pdf_path.name}")

        try:
            # Extract text
            text = self._extract_text_from_pdf(pdf_path)
            if not text:
                print(f"  Skipping (no text extracted)")
                return 0

            # Extract publication year
            publication_year = self._extract_publication_year(pdf_path, text)
            
            # Extract LLM metadata if enabled
            llm_metadata = {}
            if self.use_llm_metadata:
                print(f"  Extracting metadata with {self.llm_model}...")
                llm_metadata = self._extract_llm_metadata(text, pdf_path)
                if llm_metadata:
                    print(f"    Category: {llm_metadata.get('category', 'unknown')}")
                    print(f"    Keywords: {llm_metadata.get('keywords', 'none')[:50]}...")
            
            # Chunk text (use semantic or character-based)
            if self.use_semantic_chunking:
                chunks = self._chunk_text_semantic(text)
                chunking_method = "semantic"
            else:
                chunks = self._chunk_text(text)
                chunking_method = "character"

            if not chunks:
                print(f"  Skipping (no chunks created)")
                return 0

            # Generate embeddings
            embeddings = self.embedding_model.encode(chunks, show_progress_bar=False)

            # Prepare metadata
            doc_id_base = hashlib.md5(file_key.encode()).hexdigest()[:8]
            ids = [f"{doc_id_base}_chunk_{i}" for i in range(len(chunks))]
            
            # Base metadata for all chunks
            base_metadata = {
                "source": file_key,
                "filename": pdf_path.name,
                "chunking_method": chunking_method,
                "publication_year": publication_year
            }
            
            # Add LLM metadata if available
            if llm_metadata:
                base_metadata.update({
                    "category": llm_metadata.get("category", ""),
                    "keywords": llm_metadata.get("keywords", ""),
                    "study_type": llm_metadata.get("study_type", ""),
                    "organism": llm_metadata.get("organism", ""),
                    "disease_focus": llm_metadata.get("disease_focus", ""),
                    "methods": llm_metadata.get("methods", ""),
                    "impact": llm_metadata.get("impact", "")
                })
            
            # Create metadata for each chunk
            metadatas = []
            for i in range(len(chunks)):
                chunk_metadata = base_metadata.copy()
                chunk_metadata.update({
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "chunk_size_chars": len(chunks[i])
                })
                metadatas.append(chunk_metadata)

            # Add to ChromaDB in batches to avoid size limit
            # ChromaDB has max batch size of ~5461, so we use 5000 to be safe
            BATCH_SIZE = 5000
            embeddings_list = embeddings.tolist()

            for i in range(0, len(chunks), BATCH_SIZE):
                batch_end = min(i + BATCH_SIZE, len(chunks))
                self.collection.add(
                    ids=ids[i:batch_end],
                    embeddings=embeddings_list[i:batch_end],
                    documents=chunks[i:batch_end],
                    metadatas=metadatas[i:batch_end]
                )

            # Update index state
            file_hash = self._get_file_hash(pdf_path)
            self.indexed_files[file_key] = file_hash

            print(f"  Indexed {len(chunks)} chunks")
            return len(chunks)

        except Exception as e:
            print(f"  ERROR indexing {pdf_path.name}: {e}")
            print(f"  Skipping this file and continuing...")
            return 0

    def index_all_new(self) -> Dict[str, int]:
        """Index all new or modified PDFs"""
        new_or_modified = self.find_new_or_modified_pdfs()

        if not new_or_modified:
            print("No new or modified PDFs found. Index is up to date!")
            return {
                "new_files": 0,
                "total_chunks": 0,
                "collection_size": self.collection.count(),
                "failed_files": 0
            }

        print(f"\nFound {len(new_or_modified)} new or modified PDFs to index")

        total_chunks = 0
        failed_files = 0
        # Check if we have a TTY (interactive terminal) for tqdm
        # When running from web server, sys.stderr is not a TTY and causes BrokenPipeError
        has_tty = sys.stderr.isatty() if hasattr(sys.stderr, 'isatty') else False

        if has_tty:
            # Use tqdm for interactive terminal
            for pdf_path, file_key in tqdm(new_or_modified, desc="Indexing PDFs"):
                chunks_indexed = self.index_pdf(pdf_path, file_key)
                if chunks_indexed == 0:
                    failed_files += 1
                total_chunks += chunks_indexed
        else:
            # Print simple progress for non-interactive environments (web server)
            for i, (pdf_path, file_key) in enumerate(new_or_modified):
                print(f"Progress: {i+1}/{len(new_or_modified)} - {pdf_path.name}")
                chunks_indexed = self.index_pdf(pdf_path, file_key)
                if chunks_indexed == 0:
                    failed_files += 1
                total_chunks += chunks_indexed

        # Save state
        self._save_index_state()

        print(f"\n✓ Indexing complete!")
        print(f"  Files processed: {len(new_or_modified)}")
        print(f"  Files succeeded: {len(new_or_modified) - failed_files}")
        print(f"  Files failed/skipped: {failed_files}")
        print(f"  Total chunks indexed: {total_chunks}")
        print(f"  Total documents in collection: {self.collection.count()}")

        return {
            "new_files": len(new_or_modified),
            "total_chunks": total_chunks,
            "collection_size": self.collection.count(),
            "failed_files": failed_files
        }

    def get_stats(self) -> Dict:
        """Get indexing statistics"""
        return {
            "total_indexed_files": len(self.indexed_files),
            "collection_size": self.collection.count(),
            "embeddings_dir": str(self.embeddings_dir),
            "collection_name": self.collection_name
        }


if __name__ == "__main__":
    # Example usage
    indexer = PDFIndexer(
        endnote_pdf_dir="/home/david/projects/EndNote_Library/PDF",
        embeddings_dir="/fastpool/rag_embeddings"
    )

    # Show current stats
    stats = indexer.get_stats()
    print("Current Index Stats:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # Index new/modified PDFs
    print("\nStarting incremental indexing...")
    results = indexer.index_all_new()
