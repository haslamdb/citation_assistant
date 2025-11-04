#!/usr/bin/env python3
"""
Citation Assistant - Enhanced with Unified Configuration
All search methods share global settings with improved relevance ranking
"""

import os
import sys
import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

# Configure model locations before importing models
try:
    sys.path.insert(0, str(Path(__file__).parent.parent))
    import model_config  # Sets environment variables
except ImportError:
    pass  # Fall back to default locations

from sentence_transformers import SentenceTransformer, CrossEncoder
import chromadb
from chromadb.config import Settings
import ollama
import numpy as np
from rank_bm25 import BM25Okapi
import pickle


@dataclass
class GlobalConfig:
    """
    Global configuration shared across all search functions
    Set these once at the top level, applies to all operations
    """
    # Core search parameters
    n_papers: int = 10                      # Number of papers to return
    chunks_per_paper: int = 1               # 1-10 chunks per paper (1=single best, 2+=more context)
    
    # Advanced search options
    enable_reranking: bool = False          # Use cross-encoder re-ranking (10-15% better precision)
    enable_hybrid: bool = True              # Combine vector + BM25 search
    hybrid_balance: float = 0.5             # 0.0=pure BM25, 0.5=balanced, 1.0=pure vector
    
    # LLM configuration
    llm_model: str = "gemma2:27b"          # Model for summarization and writing
    
    # Internal parameters (advanced users can tune these)
    fetch_multiplier: int = 100             # How many chunks to initially fetch (increased for better coverage)
    max_fetch: int = 10000                  # Maximum chunks to fetch (increased)
    duplicate_threshold: float = 0.95       # Similarity threshold for duplicate detection (0.8-1.0)
    keyword_boost_strength: float = 0.5     # Keyword boost factor (lower = stronger, 0.1-1.0)
    entity_boost_strength: float = 0.001    # Entity boost factor (EXTREME boost for exact matches)
    require_query_terms: bool = False       # Filter out papers without query terms
    
    def to_dict(self) -> dict:
        """Convert to dictionary for easy inspection"""
        return {
            'n_papers': self.n_papers,
            'chunks_per_paper': self.chunks_per_paper,
            'enable_reranking': self.enable_reranking,
            'enable_hybrid': self.enable_hybrid,
            'hybrid_balance': self.hybrid_balance,
            'llm_model': self.llm_model
        }


class CitationAssistant:
    """Search research papers and suggest citations for manuscripts"""

    def __init__(
        self,
        embeddings_dir: str,
        collection_name: str = "research_papers",
        embedding_model: str = "pritamdeka/S-PubMedBert-MS-MARCO",
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-12-v2",
        config: GlobalConfig = None
    ):
        """
        Initialize Citation Assistant
        
        Args:
            embeddings_dir: Directory containing ChromaDB embeddings
            collection_name: Name of the ChromaDB collection
            embedding_model: Sentence transformer model for embeddings
            reranker_model: Cross-encoder model for re-ranking
            config: Global configuration (creates default if None)
        """
        self.embeddings_dir = Path(embeddings_dir)
        self.collection_name = collection_name
        
        # Initialize or use provided config
        self.config = config if config is not None else GlobalConfig()

        # Initialize embedding model
        print(f"Loading embedding model: {embedding_model}")
        self.embedding_model = SentenceTransformer(embedding_model)

        # Initialize cross-encoder re-ranker (loaded on-demand if enabled)
        self.reranker_model_name = reranker_model
        self.reranker = None

        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(
            path=str(self.embeddings_dir),
            settings=Settings(anonymized_telemetry=False)
        )

        try:
            self.collection = self.client.get_collection(name=self.collection_name)
            print(f"Loaded collection '{self.collection_name}' with {self.collection.count()} documents")
        except Exception as e:
            print(f"Error: Collection '{self.collection_name}' not found. Please run indexer first.")
            raise e

        # Load BM25 index if available (for hybrid search)
        self.bm25_index = None
        self.bm25_doc_map = None
        bm25_path = self.embeddings_dir / "bm25_index.pkl"
        if bm25_path.exists():
            try:
                with open(bm25_path, 'rb') as f:
                    bm25_data = pickle.load(f)
                    self.bm25_index = bm25_data['index']
                    self.bm25_doc_map = bm25_data['doc_map']
                print(f"✓ Loaded BM25 index for hybrid search ({len(self.bm25_doc_map)} documents)")
            except Exception as e:
                print(f"⚠ Could not load BM25 index: {e}")

    def _ensure_reranker_loaded(self):
        """Load cross-encoder re-ranker on demand"""
        if self.config.enable_reranking and self.reranker is None:
            print(f"Loading cross-encoder re-ranker: {self.reranker_model_name}")
            self.reranker = CrossEncoder(self.reranker_model_name)
            print("✓ Re-ranking enabled (improves precision by 10-15%)")

    def update_config(self, **kwargs):
        """
        Update global configuration parameters
        
        Example:
            assistant.update_config(n_papers=15, enable_reranking=True)
        """
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
                print(f"✓ Updated {key} = {value}")
            else:
                print(f"⚠ Unknown config parameter: {key}")

    def get_config(self) -> dict:
        """Get current configuration as dictionary"""
        return self.config.to_dict()

    # ========================================================================
    # ENTITY DETECTION AND QUERY PREPROCESSING
    # ========================================================================

    def _detect_entities(self, query: str) -> List[str]:
        """
        Extract entity names (compounds, drugs, proteins, etc.) from query
        
        Returns list of terms that should be treated as exact-match entities:
        - Capitalized terms (e.g., "Golgicide A", "Protein X")
        - Abbreviations (e.g., "CRISPR", "mRNA")
        - Drug/compound patterns
        """
        entities = []
        
        # Extract capitalized phrases (including those starting with lowercase like "golgicide A")
        cap_phrases = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z0-9]?\b)*', query)
        entities.extend(cap_phrases)
        
        # Also extract lowercase word + letter/number patterns (e.g., "golgicide A", "compound 1")
        lowercase_compound_patterns = re.findall(r'\b[a-z]+\s+[A-Z0-9]\b', query)
        entities.extend(lowercase_compound_patterns)
        
        # Extract abbreviations (2-5 uppercase letters)
        abbrevs = re.findall(r'\b[A-Z]{2,5}\b', query)
        entities.extend(abbrevs)
        
        # Extract compound patterns (e.g., "compound A", "drug 123")
        compound_patterns = re.findall(r'\b\w+\s+[A-Z0-9]\b', query)
        entities.extend(compound_patterns)
        
        # Special handling for drug names ending in 'cide', 'cin', etc.
        drug_suffixes = ['cide', 'cin', 'mycin', 'cycline', 'fenac', 'pril', 'olol',
                        'dipine', 'statin', 'zole', 'mab', 'nib', 'tinib']
        for suffix in drug_suffixes:
            pattern = rf'\b\w+{suffix}\b(?:\s+[A-Z0-9]\b)?'
            matches = re.findall(pattern, query, re.IGNORECASE)
            entities.extend(matches)
        
        return list(set(entities))  # Remove duplicates

    def _is_compound_query(self, query: str) -> bool:
        """
        Detect if query is asking about a specific compound/drug
        Returns True if query contains drug names or compound patterns
        """
        query_lower = query.lower()
        
        # Check for discovery/mechanism keywords with entity
        if any(word in query_lower for word in ['discovery', 'discovered', 'mechanism', 'action']):
            if self._detect_entities(query):
                return True
        
        # Check for common drug suffixes
        drug_suffixes = [
            'cide', 'cin', 'mycin', 'cycline', 'fenac', 'pril', 'olol',
            'dipine', 'statin', 'zole', 'mab', 'nib', 'tinib'
        ]
        if any(query_lower.endswith(suffix) for suffix in drug_suffixes):
            return True
        
        # Check for chemical compound patterns
        if re.search(r'\b(compound|drug|inhibitor|molecule)\s+[A-Z0-9]', query, re.IGNORECASE):
            return True
        
        return False

    def _preprocess_query(self, query: str, boost_keywords: str = "") -> Dict:
        """
        Preprocess query to extract important information
        
        Returns dict with:
        - original: original query string
        - terms: individual query terms (stopwords removed)
        - entities: detected entity names
        - boost_terms: terms to boost (entities + user keywords)
        - required_terms: terms that must appear (for compound queries)
        - is_compound_query: whether this is a compound/drug query
        """
        query_lower = query.lower()
        
        # Extract terms (remove stopwords)
        stopwords = {'the', 'a', 'an', 'of', 'in', 'on', 'at', 'to', 'for', 'is', 'was', 'are', 'by'}
        terms = [t for t in query_lower.split() if t not in stopwords and len(t) > 2]
        
        # Detect entities
        entities = self._detect_entities(query)
        is_compound = self._is_compound_query(query)
        
        # Build boost terms list
        boost_terms = entities.copy()
        if boost_keywords:
            boost_terms.extend([kw.strip() for kw in boost_keywords.split(',') if kw.strip()])
        
        # Determine required terms (for strict matching)
        required_terms = []
        if is_compound and entities:
            # For compound queries, require the entity names
            required_terms = [e.lower() for e in entities]
        elif self.config.require_query_terms:
            # Otherwise use main query terms if configured
            required_terms = terms[:2]  # Require at least 2 main terms
        
        return {
            'original': query,
            'terms': terms,
            'entities': entities,
            'boost_terms': boost_terms,
            'required_terms': required_terms,
            'is_compound_query': is_compound
        }

    # ========================================================================
    # SCORING AND FILTERING
    # ========================================================================

    def _check_required_terms(self, text: str, required_terms: List[str]) -> Tuple[bool, int]:
        """
        Check if text contains required terms
        Returns (passes_filter, num_matches)
        """
        if not required_terms:
            return True, 0
        
        text_lower = text.lower()
        matches = sum(1 for term in required_terms if term in text_lower)
        return matches > 0, matches

    def _score_exact_matches(self, text: str, query_info: Dict) -> float:
        """
        Calculate boost multiplier for exact term matches
        Lower multiplier = better (will be multiplied with distance)
        
        Returns:
            Multiplier in range [0.01, 1.0]
            - 0.01-0.05: Excellent match (multiple entity mentions)
            - 0.1-0.3: Strong match (entity or multiple boost terms)
            - 0.5-0.7: Moderate match (some query terms)
            - 1.0: No exact matches
        """
        text_lower = text.lower()
        multiplier = 1.0
        
        # Entity matches (highest priority)
        if query_info['entities']:
            entity_matches = 0
            for entity in query_info['entities']:
                entity_lower = entity.lower()
                # Count exact matches (case-insensitive)
                count = text_lower.count(entity_lower)
                entity_matches += count
                
                # Also count with spaces/punctuation variations
                if ' ' in entity_lower:
                    # For compound names like "golgicide A", also check without space
                    entity_no_space = entity_lower.replace(' ', '')
                    count = text_lower.count(entity_no_space)
                    entity_matches += count
            
            if entity_matches > 0:
                # Exponential boost for entity matches (VERY strong boost)
                # With entity_boost_strength = 0.01, each match reduces distance by 99%
                multiplier *= (self.config.entity_boost_strength ** min(entity_matches, 10))
        
        # Boost term matches
        if query_info['boost_terms']:
            boost_matches = sum(
                1 for term in query_info['boost_terms']
                if term.lower() in text_lower
            )
            if boost_matches > 0:
                multiplier *= (self.config.keyword_boost_strength ** boost_matches)
        
        # General query term matches
        term_matches = sum(
            1 for term in query_info['terms']
            if term in text_lower
        )
        if term_matches > 0:
            multiplier *= (0.8 ** term_matches)
        
        return multiplier

    def _filter_and_score_chunks(
        self,
        chunks: List[Dict],
        query_info: Dict
    ) -> Tuple[List[Dict], int]:
        """
        Filter chunks by required terms and apply exact match scoring
        
        Returns:
            (filtered_chunks, num_filtered_out)
        """
        filtered_chunks = []
        num_filtered = 0
        
        for chunk in chunks:
            text = chunk['text']
            
            # Check required terms
            passes, num_matches = self._check_required_terms(text, query_info['required_terms'])
            if not passes:
                num_filtered += 1
                continue
            
            chunk['required_term_matches'] = num_matches
            
            # Calculate exact match boost
            exact_match_multiplier = self._score_exact_matches(text, query_info)
            
            # Apply to distance
            chunk['original_distance'] = chunk['distance']
            chunk['distance'] = chunk['distance'] * exact_match_multiplier
            chunk['similarity'] = 1 / (1 + chunk['distance'])
            chunk['exact_match_multiplier'] = exact_match_multiplier
            
            filtered_chunks.append(chunk)
        
        return filtered_chunks, num_filtered

    # ========================================================================
    # CORE SEARCH LOGIC
    # ========================================================================

    def _check_duplicate_papers(
        self,
        text1: str,
        text2: str,
        threshold: float = None
    ) -> bool:
        """
        Check if two texts are from duplicate papers using cosine similarity
        
        Args:
            threshold: Similarity threshold (uses config default if None)
                0.80: Aggressive - removes papers with similar topics
                0.90: Moderate - removes very similar papers
                0.95: Conservative - removes near-duplicates (default)
                0.98: Minimal - only removes nearly identical papers
                1.00: Exact - only removes 100% identical papers
        """
        if threshold is None:
            threshold = self.config.duplicate_threshold
        
        # Validate threshold
        if not 0.8 <= threshold <= 1.0:
            threshold = 0.95
        
        # Use first 2000 chars for comparison
        sample1 = text1[:2000] if len(text1) > 2000 else text1
        sample2 = text2[:2000] if len(text2) > 2000 else text2
        
        # Encode both texts
        embeddings = self.embedding_model.encode([sample1, sample2])
        
        # Calculate cosine similarity
        from numpy import dot
        from numpy.linalg import norm
        cosine_sim = dot(embeddings[0], embeddings[1]) / (norm(embeddings[0]) * norm(embeddings[1]))
        
        return cosine_sim > threshold

    def _fetch_chunks_from_chromadb(self, query: str, fetch_count: int) -> Dict:
        """Fetch chunks from ChromaDB vector database"""
        query_embedding = self.embedding_model.encode([query])[0]
        
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=min(fetch_count, self.collection.count()),
            include=["documents", "metadatas", "distances"]
        )
        
        return results

    def _select_diverse_chunks(
        self,
        chunks: List[Dict],
        chunks_per_paper: int
    ) -> List[Dict]:
        """
        Select diverse chunks from different parts of a document
        Ensures we get content from multiple sections, not just similar passages
        """
        if len(chunks) <= chunks_per_paper:
            return chunks
        
        # Sort by document position
        chunks.sort(key=lambda x: x.get('chunk_index', 0))
        
        # Always include the best scoring chunk
        best_chunk = min(chunks, key=lambda x: x['distance'])
        selected = [best_chunk]
        remaining = [c for c in chunks if c != best_chunk]
        
        if chunks_per_paper > 1 and remaining:
            # Add evenly spaced chunks from the document
            step = max(1, len(remaining) // (chunks_per_paper - 1))
            for i in range(0, len(remaining), step):
                if len(selected) < chunks_per_paper:
                    selected.append(remaining[i])
        
        # Sort by position for readability
        selected.sort(key=lambda x: x.get('chunk_index', 0))
        return selected

    def unified_search(
        self,
        query: str,
        boost_keywords: str = "",
        boost_haslam: bool = True,
        override_config: Dict = None
    ) -> List[Dict]:
        """
        Unified search method with improved relevance ranking
        Uses global config but allows temporary overrides
        
        Args:
            query: Search query string
            boost_keywords: Additional keywords to boost (comma-separated)
            boost_haslam: Whether to boost papers mentioning "Haslam"
            override_config: Temporary config overrides (e.g., {'n_papers': 5})
        
        Returns:
            List of relevant papers with enhanced ranking
        """
        # Apply temporary config overrides if provided
        original_config = {}
        if override_config:
            for key, value in override_config.items():
                if hasattr(self.config, key):
                    original_config[key] = getattr(self.config, key)
                    setattr(self.config, key, value)
        
        try:
            return self._unified_search_impl(query, boost_keywords, boost_haslam)
        finally:
            # Restore original config
            for key, value in original_config.items():
                setattr(self.config, key, value)

    def _unified_search_impl(
        self,
        query: str,
        boost_keywords: str,
        boost_haslam: bool
    ) -> List[Dict]:
        """Internal implementation of unified search"""
        
        # Preprocess query
        query_info = self._preprocess_query(query, boost_keywords)
        
        # Print query analysis
        if query_info['is_compound_query']:
            print(f"🔬 Detected compound/drug query")
        if query_info['entities']:
            print(f"🎯 Entities: {', '.join(query_info['entities'])}")
        if query_info['required_terms']:
            print(f"✓ Requiring terms: {', '.join(query_info['required_terms'])}")
        
        # Calculate fetch count
        chunks_per_paper = self.config.chunks_per_paper
        fetch_count = min(
            self.config.n_papers * self.config.fetch_multiplier * chunks_per_paper,
            self.config.max_fetch
        )
        
        # Fetch chunks from ChromaDB
        results = self._fetch_chunks_from_chromadb(query, fetch_count)
        
        # Process chunks into list
        all_chunks = []
        for i in range(len(results['ids'][0])):
            chunk = {
                'id': results['ids'][0][i],
                'text': results['documents'][0][i],
                'filename': results['metadatas'][0][i]['filename'],
                'distance': results['distances'][0][i],
                'similarity': 1 / (1 + results['distances'][0][i]),
                'chunk_index': results['metadatas'][0][i].get('chunk_index', i),
                'metadata': results['metadatas'][0][i]
            }
            all_chunks.append(chunk)
        
        # Filter and score chunks
        filtered_chunks, num_filtered = self._filter_and_score_chunks(all_chunks, query_info)
        
        if num_filtered > 0:
            print(f"⚠ Filtered out {num_filtered} chunks (missing required terms)")
        
        # Apply Haslam boost if enabled
        if boost_haslam:
            for chunk in filtered_chunks:
                if 'haslam' in chunk['text'].lower():
                    chunk['distance'] *= 0.5
                    chunk['similarity'] = 1 / (1 + chunk['distance'])
        
        # Group chunks by paper
        papers_chunks = {}
        for chunk in filtered_chunks:
            filename = chunk['filename']
            if filename not in papers_chunks:
                papers_chunks[filename] = []
            papers_chunks[filename].append(chunk)
        
        # Select chunks per paper and combine
        final_papers = []
        for filename, chunks in papers_chunks.items():
            if chunks_per_paper == 1:
                # Single best chunk
                best_chunk = min(chunks, key=lambda x: x['distance'])
                paper_text = best_chunk['text']
                selected_chunks = [best_chunk]
            else:
                # Multiple diverse chunks
                selected_chunks = self._select_diverse_chunks(chunks, chunks_per_paper)
                paper_text = "\n\n---[Chunk Break]---\n\n".join([
                    f"[Section {c['chunk_index']}]\n{c['text']}"
                    for c in selected_chunks
                ])
            
            best_chunk = min(selected_chunks, key=lambda x: x['distance'])
            
            final_papers.append({
                'filename': filename,
                'text': paper_text,
                'similarity': best_chunk['similarity'],
                'distance': best_chunk['distance'],
                'num_chunks': len(selected_chunks),
                'source': best_chunk['metadata'].get('source', ''),
                'exact_match_multiplier': best_chunk.get('exact_match_multiplier', 1.0),
                'has_haslam': 'haslam' in paper_text.lower()
            })
        
        # Apply BM25 hybrid search if enabled
        if self.config.enable_hybrid and self.bm25_index:
            query_tokens = query.lower().split()
            bm25_scores = self.bm25_index.get_scores(query_tokens)
            
            # Create filename -> BM25 score mapping
            bm25_score_map = {}
            for idx, score in enumerate(bm25_scores):
                if idx < len(self.bm25_doc_map):
                    filename = self.bm25_doc_map[idx]
                    if filename not in bm25_score_map or score > bm25_score_map[filename]:
                        bm25_score_map[filename] = score
            
            # Normalize BM25 scores
            max_bm25 = max(bm25_score_map.values()) if bm25_score_map else 1.0
            if max_bm25 > 0:
                bm25_score_map = {k: v / max_bm25 for k, v in bm25_score_map.items()}
            
            # Combine scores using hybrid balance
            # hybrid_balance: 0.0 = pure BM25, 1.0 = pure vector
            for paper in final_papers:
                vector_score = paper['similarity']
                bm25_score = bm25_score_map.get(paper['filename'], 0.0)
                paper['hybrid_score'] = (
                    self.config.hybrid_balance * vector_score +
                    (1 - self.config.hybrid_balance) * bm25_score
                )
                paper['bm25_score'] = bm25_score
        
        # Remove duplicates
        unique_papers = []
        for paper in final_papers:
            is_duplicate = False
            for existing in unique_papers:
                if self._check_duplicate_papers(paper['text'], existing['text']):
                    # Keep the better scoring one
                    if paper['distance'] < existing['distance']:
                        unique_papers.remove(existing)
                        unique_papers.append(paper)
                    is_duplicate = True
                    break
            if not is_duplicate:
                unique_papers.append(paper)
        
        final_papers = unique_papers
        
        # Apply cross-encoder re-ranking if enabled
        if self.config.enable_reranking and len(final_papers) > 0:
            self._ensure_reranker_loaded()
            
            # Re-rank top candidates
            rerank_candidates = final_papers[:min(len(final_papers), self.config.n_papers * 3)]
            pairs = [[query, paper['text']] for paper in rerank_candidates]
            rerank_scores = self.reranker.predict(pairs)
            
            for i, paper in enumerate(rerank_candidates):
                paper['rerank_score'] = float(rerank_scores[i])
            
            # Sort by rerank scores
            final_papers = sorted(
                rerank_candidates,
                key=lambda x: x['rerank_score'],
                reverse=True
            )
            
            # Add remaining papers
            remaining = [p for p in final_papers[len(rerank_candidates):] if 'rerank_score' not in p]
            final_papers.extend(remaining)
        else:
            # Sort by similarity or hybrid score
            if self.config.enable_hybrid and final_papers and 'hybrid_score' in final_papers[0]:
                final_papers.sort(key=lambda x: x.get('hybrid_score', x['similarity']), reverse=True)
            else:
                final_papers.sort(key=lambda x: x['similarity'], reverse=True)
        
        # Print search summary
        print(f"📊 Found {len(final_papers)} unique papers")
        if self.config.enable_hybrid:
            print(f"   Hybrid search (balance: {self.config.hybrid_balance:.1f})")
        if self.config.enable_reranking:
            print(f"   Cross-encoder re-ranking: ON")
        
        return final_papers[:self.config.n_papers]

    # ========================================================================
    # PUBLIC SEARCH METHODS
    # ========================================================================

    def search_papers(
        self,
        query: str,
        boost_keywords: str = "",
        boost_haslam: bool = True
    ) -> List[Dict]:
        """
        Search for relevant papers
        Uses global config (n_papers, chunks_per_paper, etc.)
        
        Args:
            query: Search query string
            boost_keywords: Additional keywords to boost (comma-separated)
            boost_haslam: Whether to boost papers mentioning "Haslam"
        
        Returns:
            List of relevant papers with metadata
        """
        return self.unified_search(query, boost_keywords, boost_haslam)

    def summarize_research(
        self,
        query: str,
        boost_keywords: str = ""
    ) -> str:
        """
        Generate comprehensive summary of research on a topic
        Uses global config for search parameters and LLM model
        
        Args:
            query: Research topic to summarize
            boost_keywords: Additional keywords to boost
        
        Returns:
            Formatted summary with inline citations and references
        """
        # Search for papers
        papers = self.unified_search(query, boost_keywords)
        
        if not papers:
            return "No relevant papers found in your library."
        
        # Prepare context with numbered citations
        context_parts = []
        reference_list = []
        
        for i, p in enumerate(papers, 1):
            context_parts.append(f"[{i}] {p['filename']}\nExcerpt: {p['text']}")
            reference_list.append(f"[{i}] {p['filename']}")
        
        context = "\n\n".join(context_parts)
        references = "\n".join(reference_list)
        
        # Create prompt for LLM
        prompt = f"""Based on the following research papers from my library, provide a comprehensive and detailed summary addressing this topic: "{query}"

Research excerpts:
{context}

Please provide an EXTENSIVE summary that includes:
1. A detailed synthesis of the key findings from these papers
2. Common themes, methodologies, and consensus across studies
3. Any contradictions, debates, or conflicting results
4. Implications and significance of the findings
5. Gaps in the current research and future directions
6. Technical details and specific results when relevant

CRITICAL CITATION REQUIREMENTS:
- After EVERY statement or claim, include an inline citation using the format [1], [2], etc.
- Use the paper numbers from the excerpts above
- Multiple papers can support one statement: [1, 3]
- Be generous with citations - cite frequently throughout
- Every paragraph should have multiple citations

After your summary, include a "References" section listing all cited papers."""

        # Query LLM
        print(f"\n🤖 Querying {self.config.llm_model}...")
        response = ollama.chat(
            model=self.config.llm_model,
            messages=[{'role': 'user', 'content': prompt}]
        )
        
        summary = response['message']['content']
        
        # Append references if not included
        if "References" not in summary and "REFERENCES" not in summary:
            summary += f"\n\n{'='*80}\nREFERENCES\n{'='*80}\n{references}"
        
        return summary

    def suggest_citations_for_manuscript(
        self,
        manuscript_text: str,
        n_suggestions_per_statement: int = 3
    ) -> List[Dict]:
        """
        Analyze manuscript and suggest relevant citations
        Uses global config for search parameters and LLM model
        
        Args:
            manuscript_text: Manuscript excerpt to analyze
            n_suggestions_per_statement: Number of paper suggestions per statement
        
        Returns:
            List of citation suggestions with confidence scores
        """
        # Use LLM to identify claims needing citations
        prompt = f"""Analyze the following manuscript excerpt and identify specific claims, findings, or statements that would benefit from citations.

Manuscript:
{manuscript_text}

Please list each statement that needs a citation, one per line, in this format:
CLAIM: [the specific statement that needs support]

Be concise and specific. Focus on factual claims, methodological choices, and assertions that would typically require scholarly support."""

        print(f"\n🤖 Analyzing manuscript with {self.config.llm_model}...")
        response = ollama.chat(
            model=self.config.llm_model,
            messages=[{'role': 'user', 'content': prompt}]
        )
        
        # Parse identified claims
        claims_text = response['message']['content']
        claims = []
        for line in claims_text.split('\n'):
            if line.strip().startswith('CLAIM:'):
                claim = line.replace('CLAIM:', '').strip()
                if claim:
                    claims.append(claim)
        
        if not claims:
            print("No claims identified that need citations.")
            return []
        
        print(f"✓ Identified {len(claims)} claims needing citations")
        
        # Find relevant papers for each claim
        suggestions = []
        for claim in claims:
            # Temporarily override n_papers for citation suggestions
            papers = self.unified_search(
                query=claim,
                override_config={'n_papers': n_suggestions_per_statement}
            )
            
            if papers:
                suggestions.append({
                    'statement': claim,
                    'suggested_papers': papers,
                    'confidence': papers[0]['similarity'] if papers else 0
                })
        
        return suggestions

    def format_citation_suggestions(self, suggestions: List[Dict]) -> str:
        """Format citation suggestions for display"""
        if not suggestions:
            return "No citation suggestions generated."
        
        output = "CITATION SUGGESTIONS\n" + "=" * 80 + "\n\n"
        
        for i, suggestion in enumerate(suggestions, 1):
            output += f"{i}. STATEMENT NEEDING CITATION:\n"
            output += f"   \"{suggestion['statement']}\"\n\n"
            output += f"   SUGGESTED REFERENCES (confidence: {suggestion['confidence']:.2%}):\n"
            
            for j, paper in enumerate(suggestion['suggested_papers'], 1):
                output += f"   [{j}] {paper['filename']} (similarity: {paper['similarity']:.2%})\n"
                output += f"       Relevant excerpt: {paper['text'][:200]}...\n\n"
            
            output += "-" * 80 + "\n\n"
        
        return output

    def write_document(
        self,
        topic: str,
        style: str = "academic",
        length: str = "long",
        keywords: str = ""
    ) -> str:
        """
        Write a comprehensive document on a topic using papers from library
        Uses global config for search parameters and LLM model
        
        Args:
            topic: Document topic/title
            style: Writing style ("academic" or "grant")
            length: Document length ("short", "medium", or "long")
            keywords: Additional keywords to boost in search
        
        Returns:
            Formatted document with inline citations and references
        """
        # Search for papers
        papers = self.unified_search(topic, boost_keywords=keywords)
        
        if not papers:
            return "No relevant papers found in your library for this topic."
        
        # Prepare context with numbered citations
        context_parts = []
        reference_list = []
        
        for i, p in enumerate(papers, 1):
            context_parts.append(f"[{i}] {p['filename']}\nExcerpt: {p['text']}")
            reference_list.append(f"[{i}] {p['filename']}")
        
        context = "\n\n".join(context_parts)
        references = "\n".join(reference_list)
        
        # Define length targets
        length_guidance = {
            "short": "approximately 500-750 words (2-3 paragraphs)",
            "medium": "approximately 1000-1500 words (4-6 paragraphs)",
            "long": "approximately 2000-3000 words (8-12 paragraphs)"
        }
        
        # Define style guidance
        if style == "grant":
            style_guidance = """
GRANT PROPOSAL STYLE:
- Write in a compelling, persuasive tone
- Emphasize significance and innovation
- Highlight knowledge gaps
- Frame research in terms of broader impacts"""
        else:
            style_guidance = """
FORMAL ACADEMIC STYLE:
- Write in formal, objective academic prose
- Present balanced analysis
- Discuss methodologies and implications"""
        
        # Create prompt for LLM
        prompt = f"""You are writing a comprehensive scholarly document on the topic: "{topic}"

Based EXCLUSIVELY on the research papers provided below, write a {length_guidance.get(length, 'long')} document.

Research Papers:
{context}

{style_guidance}

CRITICAL REQUIREMENTS:
- After EVERY statement, include citations [1], [2], etc.
- Base EVERY statement on the provided excerpts
- Never add information not present in the excerpts

End with a REFERENCES section listing the papers.

Write the complete document now:"""

        # Query LLM
        print(f"\n🤖 Generating {length} {style} document with {self.config.llm_model}...")
        print(f"📚 Using {len(papers)} papers from your library")
        
        response = ollama.chat(
            model=self.config.llm_model,
            messages=[{'role': 'user', 'content': prompt}]
        )
        
        document = response['message']['content']
        
        # Append references if not included
        if "References" not in document and "REFERENCES" not in document:
            document += f"\n\n{'='*80}\nREFERENCES\n{'='*80}\n{references}"
        
        return document


# ============================================================================
# EXAMPLE USAGE AND TESTING
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("ENHANCED CITATION ASSISTANT - UNIFIED CONFIGURATION")
    print("=" * 80)
    
    # Create custom configuration
    config = GlobalConfig(
        n_papers=10,
        chunks_per_paper=2,  # Get 2 chunks per paper for better context
        enable_reranking=True,
        enable_hybrid=True,
        hybrid_balance=0.5,  # Balanced vector + BM25
        llm_model="gemma2:27b"
    )
    
    print("\nConfiguration:")
    for key, value in config.to_dict().items():
        print(f"  {key}: {value}")
    
    # Initialize assistant
    assistant = CitationAssistant(
        embeddings_dir="/fastpool/rag_embeddings",
        config=config
    )
    
    # Test query
    query = "golgicide A discovery"
    print(f"\n{'='*80}")
    print(f"Testing query: '{query}'")
    print('='*80)
    
    # Search for papers
    results = assistant.search_papers(query)
    
    print(f"\n📋 Top {len(results)} Results:")
    print("-" * 80)
    for i, paper in enumerate(results, 1):
        print(f"\n{i}. {paper['filename']}")
        print(f"   Similarity: {paper['similarity']:.4f}")
        print(f"   Exact match boost: {paper.get('exact_match_multiplier', 1.0):.4f}")
        if 'hybrid_score' in paper:
            print(f"   Hybrid score: {paper['hybrid_score']:.4f}")
        print(f"   Num chunks: {paper['num_chunks']}")
        
        # Check if golgicide is mentioned
        has_golgicide = 'golgicide' in paper['text'].lower()
        mention_count = paper['text'].lower().count('golgicide')
        print(f"   Contains 'golgicide': {'✓ YES' if has_golgicide else '✗ NO'}", end='')
        if mention_count > 0:
            print(f" ({mention_count} mentions)")
        else:
            print()
        
        # Show snippet
        snippet = paper['text'][:150].replace('\n', ' ')
        print(f"   Snippet: {snippet}...")
    
    print("\n" + "=" * 80)
    print("Test complete! Global config is working across all functions.")
    print("=" * 80)