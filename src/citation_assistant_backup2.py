#!/usr/bin/env python3
"""
Citation Assistant - Refactored with Unified Search Pipeline
All search methods use the same core logic with consistent defaults
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from enum import Enum

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


class SearchMode(Enum):
    """Search modes for different use cases"""
    STANDARD = "standard"      # Single best chunk per paper
    MULTI_CHUNK = "multi_chunk" # Multiple diverse chunks per paper
    DISCOVERY = "discovery"     # Optimized for discovery/factual queries
    HYBRID = "hybrid"          # Combines vector and BM25


class CitationAssistant:
    """Search research papers and suggest citations for manuscripts"""

    def __init__(
        self,
        embeddings_dir: str,
        collection_name: str = "research_papers",
        embedding_model: str = "pritamdeka/S-PubMedBert-MS-MARCO",
        llm_model: str = "gemma2:27b",
        enable_reranking: bool = False,
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-12-v2"
    ):
        self.embeddings_dir = Path(embeddings_dir)
        self.collection_name = collection_name
        self.llm_model = llm_model

        # Initialize embedding model
        print(f"Loading embedding model: {embedding_model}")
        self.embedding_model = SentenceTransformer(embedding_model)

        # Initialize cross-encoder re-ranker (optional)
        self.enable_reranking = enable_reranking
        self.reranker = None
        if enable_reranking:
            print(f"Loading cross-encoder re-ranker: {reranker_model}")
            self.reranker = CrossEncoder(reranker_model)
            print("Re-ranking enabled: will improve precision by 10-15%")

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

    def _check_duplicate_papers(
        self, 
        text1: str, 
        text2: str, 
        similarity_threshold: float = 0.95
    ) -> bool:
        """
        Check if two texts are from duplicate papers.
        
        Threshold scale (0.8-1.0):
        - 0.80: Aggressive - removes papers with similar topics
        - 0.90: Moderate - removes very similar papers
        - 0.95: Conservative - removes near-duplicates (default)
        - 0.98: Minimal - only removes nearly identical papers
        - 1.00: Exact - only removes 100% identical papers
        """
        # Validate threshold
        if not 0.8 <= similarity_threshold <= 1.0:
            print(f"Warning: Invalid duplicate threshold {similarity_threshold}, using 0.95")
            similarity_threshold = 0.95
        # Use more text for better comparison
        sample1 = text1[:2000] if len(text1) > 2000 else text1
        sample2 = text2[:2000] if len(text2) > 2000 else text2
        
        # Encode both texts
        embeddings = self.embedding_model.encode([sample1, sample2])
        
        # Calculate cosine similarity
        from numpy import dot
        from numpy.linalg import norm
        cosine_sim = dot(embeddings[0], embeddings[1]) / (norm(embeddings[0]) * norm(embeddings[1]))
        
        return cosine_sim > similarity_threshold

    def _fetch_chunks_from_chromadb(
        self,
        query: str,
        fetch_count: int = 2000
    ) -> Dict:
        """
        Fetch raw chunks from ChromaDB
        """
        query_embedding = self.embedding_model.encode([query])[0]
        
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=min(fetch_count, self.collection.count()),
            include=["documents", "metadatas", "distances"]
        )
        
        return results

    def _boost_chunks_by_keywords(
        self,
        chunks: List[Dict],
        keywords: List[str],
        boost_strength: float = 0.7
    ) -> List[Dict]:
        """
        Boost chunks that contain specific keywords
        boost_strength: Lower = stronger boost (0.7 = moderate, 0.5 = strong)
        """
        for chunk in chunks:
            text_lower = chunk['text'].lower()
            keyword_count = sum(1 for kw in keywords if kw.lower() in text_lower)
            
            if keyword_count > 0:
                # Apply exponential boost based on keyword matches
                chunk['distance'] *= (boost_strength ** keyword_count)
                chunk['similarity'] = 1 / (1 + chunk['distance'])
                chunk['keyword_matches'] = keyword_count
        
        return chunks

    def _select_diverse_chunks(
        self,
        chunks: List[Dict],
        chunks_per_paper: int
    ) -> List[Dict]:
        """
        Select diverse chunks from different parts of the document
        instead of just the best-scoring ones
        """
        if len(chunks) <= chunks_per_paper:
            return chunks
        
        # Sort by position in document
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
        
        # Sort by document position for readability
        selected.sort(key=lambda x: x.get('chunk_index', 0))
        return selected

    def unified_search(
        self,
        query: str,
        n_results: int = 10,
        mode: SearchMode = SearchMode.STANDARD,
        chunks_per_paper: int = 1,
        boost_keywords: str = "",
        boost_haslam: bool = True,
        remove_duplicates: bool = True,  # Enable by default
        duplicate_threshold: float = 0.95,  # Default: moderate duplicate detection (0.8-1.0 scale)
        use_reranking: bool = None,
        fetch_multiplier: int = 50,
        max_fetch: int = 5000,
        keyword_boost_strength: float = 0.7,
        bm25_weight: float = 0.5
    ) -> List[Dict]:
        """
        Unified search method used by all functions
        
        Args:
            query: Search query
            n_results: Number of papers to return
            mode: Search mode (STANDARD, MULTI_CHUNK, DISCOVERY, HYBRID)
            chunks_per_paper: For MULTI_CHUNK mode
            boost_keywords: Comma-separated keywords to boost
            boost_haslam: Whether to boost papers with "Haslam"
            remove_duplicates: Whether to remove duplicate papers
            duplicate_threshold: Similarity threshold for duplicates (0.8-1.0 scale)
                0.80 = Aggressive: Remove papers with similar topics
                0.90 = Moderate: Remove very similar papers  
                0.95 = Conservative: Remove near-duplicates (default)
                0.98 = Minimal: Only remove nearly identical papers
                1.00 = Exact: Only remove 100% identical papers
            use_reranking: Use cross-encoder re-ranking
            fetch_multiplier: How many chunks to fetch initially
            max_fetch: Maximum chunks to fetch
            keyword_boost_strength: How strongly to boost keywords (lower = stronger)
            bm25_weight: Weight for BM25 in hybrid search (0-1)
        """
        # Determine actual fetch count based on mode
        if mode == SearchMode.MULTI_CHUNK:
            fetch_count = min(n_results * fetch_multiplier * chunks_per_paper, max_fetch)
        elif mode == SearchMode.DISCOVERY:
            fetch_count = min(n_results * 100, max_fetch)  # More aggressive for discovery
        else:
            fetch_count = min(n_results * fetch_multiplier, max_fetch)
        
        # Fetch chunks from ChromaDB
        results = self._fetch_chunks_from_chromadb(query, fetch_count)
        
        # Process all chunks into a list
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
        
        # Apply keyword boosting
        if boost_keywords or mode == SearchMode.DISCOVERY:
            keywords = []
            
            # Parse provided keywords
            if boost_keywords:
                keywords.extend([kw.strip() for kw in boost_keywords.split(',') if kw.strip()])
            
            # Add discovery keywords for DISCOVERY mode
            if mode == SearchMode.DISCOVERY:
                discovery_keywords = [
                    "discovered", "discovery", "first identified", "first described",
                    "novel", "shown", "found", "identified", "demonstrated",
                    "screen", "screening", "high-throughput"
                ]
                keywords.extend(discovery_keywords)
            
            if keywords:
                all_chunks = self._boost_chunks_by_keywords(
                    all_chunks, keywords, keyword_boost_strength
                )
        
        # Apply Haslam boost
        if boost_haslam:
            for chunk in all_chunks:
                if 'haslam' in chunk['text'].lower():
                    chunk['distance'] *= 0.5  # Strong boost for Haslam papers
                    chunk['similarity'] = 1 / (1 + chunk['distance'])
        
        # Group chunks by paper
        papers_chunks = {}
        for chunk in all_chunks:
            filename = chunk['filename']
            if filename not in papers_chunks:
                papers_chunks[filename] = []
            papers_chunks[filename].append(chunk)
        
        # Process based on mode
        final_papers = []
        
        if mode == SearchMode.MULTI_CHUNK:
            # Select diverse chunks for each paper
            for filename, chunks in papers_chunks.items():
                selected_chunks = self._select_diverse_chunks(chunks, chunks_per_paper)
                
                # Combine chunks
                combined_text = "\n\n---[Next Chunk]---\n\n".join([
                    f"[Chunk {c['chunk_index']}]\n{c['text']}"
                    for c in selected_chunks
                ])
                
                best_chunk = min(selected_chunks, key=lambda x: x['distance'])
                
                final_papers.append({
                    'filename': filename,
                    'text': combined_text,
                    'similarity': best_chunk['similarity'],
                    'distance': best_chunk['distance'],
                    'num_chunks': len(selected_chunks),
                    'source': best_chunk['metadata'].get('source', ''),
                    'has_haslam': 'haslam' in combined_text.lower()
                })
        
        elif mode == SearchMode.DISCOVERY:
            # Take top 2 chunks per paper for discovery context
            for filename, chunks in papers_chunks.items():
                # Sort by adjusted distance (after keyword boosting)
                chunks.sort(key=lambda x: x['distance'])
                top_chunks = chunks[:2]
                
                combined_text = "\n\n---[Discovery Context]---\n\n".join([
                    c['text'] for c in top_chunks
                ])
                
                best_chunk = top_chunks[0]
                
                final_papers.append({
                    'filename': filename,
                    'text': combined_text,
                    'similarity': best_chunk['similarity'],
                    'distance': best_chunk['distance'],
                    'num_chunks': len(top_chunks),
                    'keywords_found': sum(c.get('keyword_matches', 0) for c in top_chunks),
                    'source': best_chunk['metadata'].get('source', ''),
                    'has_haslam': 'haslam' in combined_text.lower()
                })
        
        else:  # STANDARD mode
            # Keep only the best chunk per paper
            for filename, chunks in papers_chunks.items():
                best_chunk = min(chunks, key=lambda x: x['distance'])
                
                final_papers.append({
                    'filename': filename,
                    'text': best_chunk['text'],
                    'similarity': best_chunk['similarity'],
                    'distance': best_chunk['distance'],
                    'num_chunks': 1,
                    'source': best_chunk['metadata'].get('source', ''),
                    'has_haslam': 'haslam' in best_chunk['text'].lower()
                })
        
        # Apply BM25 for HYBRID mode
        if mode == SearchMode.HYBRID and self.bm25_index:
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
            
            # Combine scores
            for paper in final_papers:
                vector_score = paper['similarity']
                bm25_score = bm25_score_map.get(paper['filename'], 0.0)
                paper['hybrid_score'] = bm25_weight * vector_score + (1 - bm25_weight) * bm25_score
                paper['bm25_score'] = bm25_score
        
        # Remove duplicates if requested
        if remove_duplicates and len(final_papers) > 1:
            unique_papers = []
            for paper in final_papers:
                is_duplicate = False
                for existing in unique_papers:
                    if self._check_duplicate_papers(paper['text'], existing['text'], duplicate_threshold):
                        # Keep the better scoring one
                        if paper['distance'] < existing['distance']:
                            unique_papers.remove(existing)
                            unique_papers.append(paper)
                        is_duplicate = True
                        break
                
                if not is_duplicate:
                    unique_papers.append(paper)
            
            final_papers = unique_papers
        
        # Apply re-ranking if enabled
        if use_reranking is None:
            use_reranking = self.enable_reranking
        
        if use_reranking and self.reranker and len(final_papers) > 0:
            # Re-rank top candidates
            rerank_candidates = final_papers[:min(len(final_papers), n_results * 3)]
            pairs = [[query, paper['text']] for paper in rerank_candidates]
            rerank_scores = self.reranker.predict(pairs)
            
            for i, paper in enumerate(rerank_candidates):
                paper['rerank_score'] = float(rerank_scores[i])
            
            # Sort by rerank scores
            final_papers = sorted(rerank_candidates, key=lambda x: x['rerank_score'], reverse=True)
            
            # Add remaining papers not re-ranked
            remaining = [p for p in final_papers[len(rerank_candidates):] if 'rerank_score' not in p]
            final_papers.extend(remaining)
        else:
            # Sort by similarity (or hybrid score if available)
            if mode == SearchMode.HYBRID and 'hybrid_score' in final_papers[0]:
                final_papers.sort(key=lambda x: x.get('hybrid_score', x['similarity']), reverse=True)
            else:
                final_papers.sort(key=lambda x: x['similarity'], reverse=True)
        
        return final_papers[:n_results]

    # Now the public methods just call unified_search with appropriate parameters
    
    def search_papers(
        self,
        query: str,
        n_results: int = 10,
        boost_haslam: bool = True,
        boost_keywords: str = "",
        use_reranking: bool = None,
        duplicate_threshold: float = 0.95
    ) -> List[Dict]:
        """Standard paper search - returns best chunk per paper"""
        return self.unified_search(
            query=query,
            n_results=n_results,
            mode=SearchMode.STANDARD,
            boost_keywords=boost_keywords,
            boost_haslam=boost_haslam,
            remove_duplicates=True,
            duplicate_threshold=duplicate_threshold,
            use_reranking=use_reranking
        )

    def search_papers_multi_chunk(
        self,
        query: str,
        n_results: int = 10,
        chunks_per_paper: int = 3,
        duplicate_threshold: float = 0.95
    ) -> List[Dict]:
        """Multi-chunk search - returns multiple diverse chunks per paper"""
        return self.unified_search(
            query=query,
            n_results=n_results,
            mode=SearchMode.MULTI_CHUNK,
            chunks_per_paper=chunks_per_paper,
            remove_duplicates=True,
            duplicate_threshold=duplicate_threshold
        )

    def search_papers_factual(
        self,
        query: str,
        n_results: int = 10,
        duplicate_threshold: float = 0.95
    ) -> List[Dict]:
        """Discovery/factual search - optimized for finding discovery narratives"""
        return self.unified_search(
            query=query,
            n_results=n_results,
            mode=SearchMode.DISCOVERY,
            remove_duplicates=True,
            duplicate_threshold=duplicate_threshold
        )

    def hybrid_search(
        self,
        query: str,
        n_results: int = 10,
        alpha: float = 0.5,
        duplicate_threshold: float = 0.95
    ) -> List[Dict]:
        """Hybrid search combining vector and BM25"""
        return self.unified_search(
            query=query,
            n_results=n_results,
            mode=SearchMode.HYBRID,
            bm25_weight=alpha,
            remove_duplicates=True,
            duplicate_threshold=duplicate_threshold
        )

    def summarize_research(
        self,
        query: str,
        n_papers: int = 10,
        search_mode: SearchMode = SearchMode.MULTI_CHUNK,
        llm_model: str = None
    ) -> str:
        """Summarize research findings on a topic"""
        # Use multi-chunk by default for better context
        papers = self.unified_search(
            query=query,
            n_results=n_papers,
            mode=search_mode,
            chunks_per_paper=2 if search_mode == SearchMode.MULTI_CHUNK else 1,
            remove_duplicates=True,
            duplicate_threshold=0.95  # Conservative default
        )

        if not papers:
            return "No relevant papers found in your library."

        # Prepare context
        context_parts = []
        reference_list = []

        for i, p in enumerate(papers, 1):
            context_parts.append(f"[{i}] {p['filename']}\nExcerpt: {p['text']}")
            reference_list.append(f"[{i}] {p['filename']}")

        context = "\n\n".join(context_parts)
        references = "\n".join(reference_list)

        # Create prompt
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
        model_to_use = llm_model if llm_model else self.llm_model
        print(f"\nQuerying {model_to_use}...")
        response = ollama.chat(
            model=model_to_use,
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
        n_suggestions_per_statement: int = 3,
        llm_model: str = None
    ) -> List[Dict]:
        """Analyze manuscript and suggest relevant citations"""
        # Use LLM to identify claims
        prompt = f"""Analyze the following manuscript excerpt and identify specific claims, findings, or statements that would benefit from citations.

Manuscript:
{manuscript_text}

Please list each statement that needs a citation, one per line, in this format:
CLAIM: [the specific statement that needs support]

Be concise and specific. Focus on factual claims, methodological choices, and assertions that would typically require scholarly support."""

        model_to_use = llm_model if llm_model else self.llm_model
        print(f"\nAnalyzing manuscript with {model_to_use}...")
        response = ollama.chat(
            model=model_to_use,
            messages=[{'role': 'user', 'content': prompt}]
        )

        # Parse claims
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

        print(f"Identified {len(claims)} claims needing citations")

        # Find papers for each claim
        suggestions = []
        for claim in claims:
            papers = self.unified_search(
                query=claim,
                n_results=n_suggestions_per_statement,
                mode=SearchMode.STANDARD,
                remove_duplicates=True,
                duplicate_threshold=0.95  # Conservative default
            )

            if papers:
                suggestions.append({
                    'statement': claim,
                    'suggested_papers': papers,
                    'confidence': papers[0]['similarity'] if papers else 0
                })

        return suggestions

    def write_document(
        self,
        topic: str,
        style: str = "academic",
        length: str = "long",
        n_papers: int = 15,
        keywords: str = "",
        search_method: str = "multi_chunk",  # Changed back to search_method for compatibility
        chunks_per_paper: int = 3,
        llm_model: str = None
    ) -> str:
        """Write a comprehensive document on a topic"""
        # Map string search method to enum (using search_method for backward compatibility)
        mode_map = {
            "standard": SearchMode.STANDARD,
            "multi_chunk": SearchMode.MULTI_CHUNK,
            "factual": SearchMode.DISCOVERY,
            "discovery": SearchMode.DISCOVERY,
            "hybrid": SearchMode.HYBRID,
            "multi-chunk": SearchMode.MULTI_CHUNK,  # Handle hyphenated version
        }
        mode = mode_map.get(search_method.lower(), SearchMode.MULTI_CHUNK)
        
        # Search for papers
        papers = self.unified_search(
            query=topic,
            n_results=n_papers,
            mode=mode,
            chunks_per_paper=chunks_per_paper if mode == SearchMode.MULTI_CHUNK else 1,
            boost_keywords=keywords,
            remove_duplicates=True,
            duplicate_threshold=0.95,  # Conservative default
            fetch_multiplier=100 if mode == SearchMode.DISCOVERY else 50
        )

        if not papers:
            return "No relevant papers found in your library for this topic."

        # Prepare context
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

        # Create prompt
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
        model_to_use = llm_model if llm_model else self.llm_model
        print(f"\nGenerating {length} {style} document with {model_to_use}...")
        print(f"Using {len(papers)} papers from your library")
        
        response = ollama.chat(
            model=model_to_use,
            messages=[{'role': 'user', 'content': prompt}]
        )

        document = response['message']['content']

        # Append references if not included
        if "References" not in document and "REFERENCES" not in document:
            document += f"\n\n{'='*80}\nREFERENCES\n{'='*80}\n{references}"

        return document

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


if __name__ == "__main__":
    # Example usage
    assistant = CitationAssistant(
        embeddings_dir="/fastpool/rag_embeddings"
    )

    print("\n" + "=" * 80)
    print("REFACTORED CITATION ASSISTANT TEST")
    print("=" * 80)
    
    # Test unified search with different modes
    query = "golgicide discovery"
    
    print(f"\nQuery: '{query}'")
    print("-" * 40)
    
    # Standard search
    results_standard = assistant.search_papers(query, n_results=5)
    print(f"Standard search: {len(results_standard)} papers")
    
    # Multi-chunk search
    results_multi = assistant.search_papers_multi_chunk(query, n_results=5, chunks_per_paper=3)
    print(f"Multi-chunk search: {len(results_multi)} papers")
    
    # Discovery search
    results_discovery = assistant.search_papers_factual(query, n_results=5)
    print(f"Discovery search: {len(results_discovery)} papers")
    
    if results_discovery:
        print(f"\nTop discovery result: {results_discovery[0]['filename']}")
        if 'keywords_found' in results_discovery[0]:
            print(f"Discovery keywords found: {results_discovery[0]['keywords_found']}")