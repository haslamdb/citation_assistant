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
        overlap_sentences: int = 2
    ):
        self.endnote_pdf_dir = Path(endnote_pdf_dir)
        self.embeddings_dir = Path(embeddings_dir)
        self.collection_name = collection_name
        self.use_semantic_chunking = use_semantic_chunking
        self.target_chunk_tokens = target_chunk_tokens
        self.overlap_sentences = overlap_sentences

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

        # Extract text
        text = self._extract_text_from_pdf(pdf_path)
        if not text:
            print(f"  Skipping (no text extracted)")
            return 0

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
        metadatas = [
            {
                "source": file_key,
                "filename": pdf_path.name,
                "chunk_index": i,
                "total_chunks": len(chunks),
                "chunking_method": chunking_method,
                "chunk_size_chars": len(chunks[i])
            }
            for i in range(len(chunks))
        ]

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

    def index_all_new(self) -> Dict[str, int]:
        """Index all new or modified PDFs"""
        new_or_modified = self.find_new_or_modified_pdfs()

        if not new_or_modified:
            print("No new or modified PDFs found. Index is up to date!")
            return {"new_files": 0, "total_chunks": 0}

        print(f"\nFound {len(new_or_modified)} new or modified PDFs to index")

        total_chunks = 0
        for pdf_path, file_key in tqdm(new_or_modified, desc="Indexing PDFs"):
            chunks_indexed = self.index_pdf(pdf_path, file_key)
            total_chunks += chunks_indexed

        # Save state
        self._save_index_state()

        print(f"\n✓ Indexing complete!")
        print(f"  Files processed: {len(new_or_modified)}")
        print(f"  Total chunks indexed: {total_chunks}")
        print(f"  Total documents in collection: {self.collection.count()}")

        return {
            "new_files": len(new_or_modified),
            "total_chunks": total_chunks,
            "collection_size": self.collection.count()
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
