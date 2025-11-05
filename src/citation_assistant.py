#!/usr/bin/env python3
"""
Citation Assistant
Search for relevant citations and suggest references for manuscripts
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Tuple

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


class CitationAssistant:
    """Search research papers and suggest citations for manuscripts"""

    def __init__(
        self,
        embeddings_dir: str,
        collection_name: str = "research_papers",
        embedding_model: str = "pritamdeka/S-PubMedBert-MS-MARCO",
        llm_model: str = "gemma2:27b",
        default_fetch_multiplier: int = 50,
        default_max_fetch: int = 2000,
        default_keyword_boost: float = 0.7,
        enable_reranking: bool = False,
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-12-v2"
    ):
        self.embeddings_dir = Path(embeddings_dir)
        self.collection_name = collection_name
        self.llm_model = llm_model

        # Store default search parameters (Phase 1 optimizations)
        self.default_fetch_multiplier = default_fetch_multiplier
        self.default_max_fetch = default_max_fetch
        self.default_keyword_boost = default_keyword_boost

        # Initialize embedding model
        print(f"Loading embedding model: {embedding_model}")
        self.embedding_model = SentenceTransformer(embedding_model)

        # Initialize cross-encoder re-ranker (optional, for improved precision)
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
        self.bm25_doc_map = None  # Maps BM25 index positions to filenames
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
                print("  Hybrid search will fall back to vector-only mode")
        else:
            print("ℹ BM25 index not found. Hybrid search disabled.")

    def _are_vectors_similar(self, distance1: float, distance2: float, threshold: float = 0.1) -> bool:
        """
        Check if two papers are likely duplicates based on their vector distances from the query

        This is more robust than filename matching because:
        - Uses actual content similarity (semantic embeddings)
        - Doesn't rely on filename conventions
        - Catches duplicate papers even with different filenames

        Args:
            distance1: L2 distance of first paper from query
            distance2: L2 distance of second paper from query
            threshold: Maximum difference in distances to consider papers as duplicates (default: 0.1)

        Returns:
            True if papers have very similar distances (likely duplicates)
        """
        # If two papers have nearly identical distances from the query,
        # they likely contain very similar content (duplicates)
        return abs(distance1 - distance2) < threshold

    def hybrid_search(
        self,
        query: str,
        n_results: int = 10,
        alpha: float = 0.5,
        **kwargs
    ) -> List[Dict]:
        """
        Hybrid search combining vector similarity (semantic) and BM25 (keyword) search

        This provides the best of both worlds:
        - Vector search: semantic similarity ("gut microbiome" → "intestinal flora")
        - BM25: exact keyword matches ("Clostridioides difficile", drug names)

        Args:
            query: Search query string
            n_results: Number of unique papers to return
            alpha: Weight for vector search (0-1). 0.5 = equal weight, 0.7 = favor vector, 0.3 = favor BM25
            **kwargs: Additional arguments passed to search_papers (use_reranking, etc.)

        Returns:
            List of papers with combined scores
        """
        if self.bm25_index is None:
            print("⚠ BM25 index not available, falling back to vector-only search")
            return self.search_papers(query, n_results=n_results, **kwargs)

        # Stage 1: Get vector search results (semantic similarity)
        # Fetch more candidates to ensure good coverage after merging
        vector_papers = self.search_papers(query, n_results=n_results * 3, **kwargs)

        # Stage 2: Get BM25 scores (keyword matching)
        # Tokenize query
        query_tokens = query.lower().split()

        # Get BM25 scores for all documents
        bm25_scores = self.bm25_index.get_scores(query_tokens)

        # Create a mapping of filename -> BM25 score
        bm25_score_map = {}
        for idx, score in enumerate(bm25_scores):
            if idx < len(self.bm25_doc_map):
                filename = self.bm25_doc_map[idx]
                if filename not in bm25_score_map or score > bm25_score_map[filename]:
                    bm25_score_map[filename] = score

        # Normalize BM25 scores to 0-1 range
        max_bm25 = max(bm25_score_map.values()) if bm25_score_map else 1.0
        if max_bm25 > 0:
            bm25_score_map = {k: v / max_bm25 for k, v in bm25_score_map.items()}

        # Stage 3: Combine scores using weighted average
        for paper in vector_papers:
            filename = paper['filename']
            vector_score = paper['similarity']  # Already normalized 0-1
            bm25_score = bm25_score_map.get(filename, 0.0)

            # Weighted combination: alpha * vector + (1-alpha) * BM25
            paper['hybrid_score'] = alpha * vector_score + (1 - alpha) * bm25_score
            paper['bm25_score'] = bm25_score

        # Sort by combined score
        hybrid_papers = sorted(vector_papers, key=lambda x: x['hybrid_score'], reverse=True)

        return hybrid_papers[:n_results]

    def search_papers(
        self,
        query: str,
        n_results: int = 10,
        boost_haslam: bool = True,
        boost_keywords: str = "",
        fetch_multiplier: int = None,
        max_fetch: int = None,
        keyword_boost_strength: float = None,
        use_reranking: bool = None
    ) -> List[Dict]:
        """Search for relevant papers given a query (deduplicated by filename)

        Args:
            query: Search query string
            n_results: Number of unique papers to return
            boost_haslam: If True, boost papers with "Haslam" as author (default: True)
            boost_keywords: Optional comma-separated keywords for aggressive boosting (e.g., "golgicide, brefeldin")
            fetch_multiplier: Multiplier for initial fetch (None = use default)
            max_fetch: Maximum chunks to fetch (None = use default)
            keyword_boost_strength: Boost factor for keyword matches (None = use default)
                Lower = stronger boost. 0.7 = moderate, 0.5 = strong, 0.1 = very aggressive
            use_reranking: If True, use cross-encoder re-ranking (None = use instance setting)
        """
        # Use instance defaults if not specified
        if fetch_multiplier is None:
            fetch_multiplier = self.default_fetch_multiplier
        if max_fetch is None:
            max_fetch = self.default_max_fetch
        if keyword_boost_strength is None:
            keyword_boost_strength = self.default_keyword_boost

        # Generate query embedding
        query_embedding = self.embedding_model.encode([query])[0]

        # Search ChromaDB with more results to account for deduplication
        # Increased from 10x/500 to 50x/2000 for better coverage
        fetch_count = min(n_results * fetch_multiplier, max_fetch)
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=fetch_count,
            include=["documents", "metadatas", "distances"]
        )

        # Extract key terms for keyword matching
        # If boost_keywords provided, use those; otherwise use query terms
        if boost_keywords.strip():
            # Parse comma-separated keywords
            query_terms = set(kw.strip().lower() for kw in boost_keywords.split(',') if kw.strip())
        else:
            query_terms = set(query.lower().split())

        # Deduplicate by filename, keeping only the best-matching chunk per paper
        # Also check for vector similarity to catch duplicate papers with different filenames
        unique_papers = {}
        for i in range(len(results['ids'][0])):
            filename = results['metadatas'][0][i]['filename']
            distance = results['distances'][0][i]
            text = results['documents'][0][i]
            text_lower = text.lower()

            # Check if we already have this exact filename
            if filename in unique_papers:
                # If this chunk is better, replace it
                if distance < unique_papers[filename]['distance']:
                    # Keep the existing paper but update with better chunk
                    unique_papers[filename]['id'] = results['ids'][0][i]
                    unique_papers[filename]['text'] = text
                    unique_papers[filename]['source'] = results['metadatas'][0][i]['source']
                    unique_papers[filename]['chunk_index'] = results['metadatas'][0][i]['chunk_index']
                    unique_papers[filename]['distance'] = distance
                    unique_papers[filename]['similarity'] = 1 / (1 + distance)
                continue

            # Check if we have a paper with very similar vector distance (likely duplicate/same paper)
            # This is more robust than filename matching
            duplicate_found = False
            for existing_filename, existing_paper in list(unique_papers.items()):
                if self._are_vectors_similar(distance, existing_paper['distance']):
                    # Found a paper with very similar embedding - likely a duplicate
                    # Keep the one with the better (lower) distance
                    if distance < existing_paper['distance']:
                        # This new paper is better, remove old one and add new one
                        del unique_papers[existing_filename]
                        duplicate_found = False  # Will add this one below
                        break
                    else:
                        # Existing paper is better, skip this one
                        duplicate_found = True
                        break

            if not duplicate_found:
                # Check if "Haslam" appears in the text (author detection)
                has_haslam = 'haslam' in text_lower

                # Check for keyword matches (exact terms from query)
                keyword_matches = sum(1 for term in query_terms if len(term) > 3 and term in text_lower)

                unique_papers[filename] = {
                    'id': results['ids'][0][i],
                    'text': text,
                    'filename': filename,
                    'source': results['metadatas'][0][i]['source'],
                    'chunk_index': results['metadatas'][0][i]['chunk_index'],
                    'distance': distance,
                    'similarity': 1 / (1 + distance),  # Normalize L2 distance to 0-1 range
                    'has_haslam': has_haslam,
                    'keyword_matches': keyword_matches
                }

        # Apply keyword boost (stronger than Haslam boost for relevance)
        for paper in unique_papers.values():
            if paper['keyword_matches'] > 0:
                # Boost papers with keyword matches
                # Using configurable exponential boost (default 0.7^n is gentler than old 0.1^n)
                # This prevents keyword matches from overwhelming semantic similarity
                paper['distance'] *= keyword_boost_strength ** paper['keyword_matches']
                paper['similarity'] = 1 / (1 + paper['distance'])

        # Apply Haslam boost to distances if enabled
        if boost_haslam:
            for paper in unique_papers.values():
                if paper['has_haslam']:
                    # Reduce distance by 50% (strong boost for Haslam papers)
                    paper['distance'] = paper['distance'] * 0.5
                    paper['similarity'] = 1 / (1 + paper['distance'])

        # Convert to list and sort by distance (best matches first)
        papers = sorted(unique_papers.values(), key=lambda x: x['distance'])

        # Apply cross-encoder re-ranking if enabled
        # Determine if we should use re-ranking (instance setting or parameter override)
        should_rerank = use_reranking if use_reranking is not None else self.enable_reranking

        if should_rerank and self.reranker is not None and len(papers) > 0:
            # Stage 2: Re-rank with cross-encoder for improved precision
            # Fetch more candidates for re-ranking (typically 3-5x final results)
            rerank_candidates = papers[:min(len(papers), n_results * 3)]

            # Create query-document pairs for cross-encoder
            pairs = [[query, paper['text']] for paper in rerank_candidates]

            # Get cross-encoder scores (higher = more relevant)
            rerank_scores = self.reranker.predict(pairs)

            # Add re-rank scores to papers
            for i, paper in enumerate(rerank_candidates):
                paper['rerank_score'] = float(rerank_scores[i])

            # Re-sort by cross-encoder scores (descending)
            papers = sorted(rerank_candidates, key=lambda x: x['rerank_score'], reverse=True)

            # Append remaining papers (not re-ranked) if any
            if len(papers) < len(unique_papers):
                remaining = [p for p in unique_papers.values() if 'rerank_score' not in p]
                papers.extend(sorted(remaining, key=lambda x: x['distance']))

        # Return only the requested number of results
        return papers[:n_results]

    def summarize_research(
        self,
        query: str,
        n_papers: int = 10,
        fetch_multiplier: int = None,
        keyword_boost_strength: float = None
    ) -> str:
        """Summarize research findings on a topic with inline citations

        Args:
            query: Search query string
            n_papers: Number of papers to use for summary (default: 10, leverages Gemma2's 8K context)
            fetch_multiplier: Multiplier for initial chunk fetch (None = use default)
            keyword_boost_strength: Boost factor for keyword matches (None = use default)
        """
        papers = self.search_papers(
            query,
            n_results=n_papers,
            fetch_multiplier=fetch_multiplier,
            keyword_boost_strength=keyword_boost_strength
        )

        if not papers:
            return "No relevant papers found in your library."

        # Prepare numbered context from papers with reference list
        context_parts = []
        reference_list = []

        for i, p in enumerate(papers[:n_papers], 1):
            # Send full chunk for maximum context (avg ~2000 chars with semantic chunking)
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

IMPORTANT:
- Base your summary ONLY on the research excerpts provided above.
- If the excerpts don't adequately address the topic, start your response with: "The papers in your library do not directly address this topic. However, based on my general knowledge..."
- Clearly distinguish between information from the library papers and any general knowledge you add.
- Make the summary detailed and comprehensive, not brief.

After your summary, include a "References" section listing all cited papers."""

        # Query Ollama
        print(f"\nQuerying {self.llm_model}...")
        response = ollama.chat(
            model=self.llm_model,
            messages=[{'role': 'user', 'content': prompt}]
        )

        summary = response['message']['content']

        # Append reference list if not already included
        if "References" not in summary and "REFERENCES" not in summary:
            summary += f"\n\n{'='*80}\nREFERENCES\n{'='*80}\n{references}"

        return summary

    def suggest_citations_for_manuscript(
        self,
        manuscript_text: str,
        n_suggestions_per_statement: int = 3,
        fetch_multiplier: int = None,
        keyword_boost_strength: float = None
    ) -> List[Dict]:
        """
        Analyze manuscript and suggest relevant citations

        Returns list of suggestions with:
        - statement: the claim/statement needing citation
        - suggested_papers: list of relevant papers
        - confidence: confidence score
        """
        # Use LLM to identify claims needing citations
        prompt = f"""Analyze the following manuscript excerpt and identify specific claims, findings, or statements that would benefit from citations.

Manuscript:
{manuscript_text}

Please list each statement that needs a citation, one per line, in this format:
CLAIM: [the specific statement that needs support]

Be concise and specific. Focus on factual claims, methodological choices, and assertions that would typically require scholarly support."""

        print(f"\nAnalyzing manuscript with {self.llm_model}...")
        response = ollama.chat(
            model=self.llm_model,
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

        print(f"Identified {len(claims)} claims needing citations")

        # Find relevant papers for each claim
        suggestions = []
        for claim in claims:
            papers = self.search_papers(
                claim,
                n_results=n_suggestions_per_statement,
                fetch_multiplier=fetch_multiplier,
                keyword_boost_strength=keyword_boost_strength
            )

            if papers:
                # Group by unique filename (deduplicate chunks from same paper)
                unique_papers = {}
                for paper in papers:
                    filename = paper['filename']
                    if filename not in unique_papers:
                        unique_papers[filename] = paper
                    else:
                        # Keep the most relevant chunk
                        if paper['similarity'] > unique_papers[filename]['similarity']:
                            unique_papers[filename] = paper

                suggestions.append({
                    'statement': claim,
                    'suggested_papers': list(unique_papers.values())[:n_suggestions_per_statement],
                    'confidence': unique_papers[list(unique_papers.keys())[0]]['similarity'] if unique_papers else 0
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
        n_papers: int = 15,
        keywords: str = "",
        fetch_multiplier: int = None,
        keyword_boost_strength: float = None
    ) -> str:
        """
        Write a comprehensive document on a topic using only papers from the library

        Args:
            topic: The topic/title for the document
            style: Writing style - "academic" (formal research article) or "grant" (grant proposal)
            length: Document length - "short" (~500 words), "medium" (~1000 words), "long" (~2000+ words)
            n_papers: Number of papers to use as sources (default: 15)
            keywords: Optional comma-separated keywords for aggressive boosting (e.g., "golgicide, brefeldin")
            fetch_multiplier: Multiplier for initial chunk fetch (None = use default)
            keyword_boost_strength: Boost factor for keyword matches (None = use default)

        Returns:
            Formatted document with inline citations and references
        """
        # Search for relevant papers
        papers = self.search_papers(
            topic,
            n_results=n_papers,
            boost_keywords=keywords,
            fetch_multiplier=fetch_multiplier,
            keyword_boost_strength=keyword_boost_strength
        )

        if not papers:
            return "No relevant papers found in your library for this topic."

        # Prepare numbered context from papers with reference list
        context_parts = []
        reference_list = []

        for i, p in enumerate(papers, 1):
            # Send full chunk for maximum context (avg ~2000 chars with semantic chunking)
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

        # Define style-specific guidance
        if style == "grant":
            style_guidance = """
GRANT PROPOSAL STYLE:
- Write in a compelling, persuasive tone appropriate for grant applications
- Emphasize significance, innovation, and impact
- Use clear topic sentences and strong transitions
- Highlight knowledge gaps and unmet needs
- Frame the research in terms of broader impacts
- Use active voice and confident language
- Structure: Background → Knowledge Gaps → Significance → Future Directions"""
        else:  # academic
            style_guidance = """
FORMAL ACADEMIC STYLE:
- Write in formal, objective academic prose
- Use complex sentence structures and sophisticated vocabulary
- Maintain third-person perspective
- Present balanced analysis of the literature
- Discuss methodologies, findings, and implications
- Acknowledge limitations and controversies
- Structure: Introduction → Current Understanding → Recent Advances → Future Directions"""

        # Create comprehensive writing prompt
        prompt = f"""You are writing a comprehensive scholarly document on the topic: "{topic}"

Based EXCLUSIVELY on the research papers from my library provided below, write a {length_guidance[length]} document.

Research Papers from Library:
{context}

DOCUMENT REQUIREMENTS:

{style_guidance}

LENGTH: Write {length_guidance[length]}

CRITICAL CITATION REQUIREMENTS:
- After EVERY statement, claim, or fact, include an inline citation using [1], [2], etc.
- Use the paper numbers from the excerpts above
- Cite multiple papers when appropriate: [1, 3, 5]
- Be EXTREMELY generous with citations - every sentence should have at least one citation
- Never make unsupported statements

CONTENT REQUIREMENTS:
- Base EVERY statement on the provided research excerpts
- Synthesize information across multiple papers
- Identify patterns, consensus, and disagreements in the literature
- Discuss methodologies where relevant
- Highlight key findings and their implications
- Identify knowledge gaps
- Do NOT add any information not present in the excerpts
- If the papers don't adequately cover the topic, state: "The available papers in your library provide limited coverage of [aspect]. Based on the available literature..."

STRUCTURE:
- Start with a strong opening that frames the topic
- Develop ideas logically across multiple paragraphs
- Use clear topic sentences for each paragraph
- Build to a conclusion about current state and future directions
- End with a "REFERENCES" section that lists the exact PDF filenames for each citation number

REFERENCES SECTION FORMAT (CRITICAL - COPY THESE EXACTLY):
At the end of your document, include this EXACT reference section by copying the filenames below:

REFERENCES
================================================================================
{references}

IMPORTANT: Simply COPY-PASTE the filenames above into your references section.
Do NOT modify the filenames. Do NOT create placeholder names like "filename1.pdf".
Use the EXACT filenames shown above for each [1], [2], [3], etc.

Write the complete document now:"""

        # Query Ollama
        print(f"\nGenerating {length} {style} document with {self.llm_model}...")
        print(f"Using {len(papers)} papers from your library")
        response = ollama.chat(
            model=self.llm_model,
            messages=[{'role': 'user', 'content': prompt}]
        )

        document = response['message']['content']

        # Append reference list if not already included
        if "References" not in document and "REFERENCES" not in document:
            document += f"\n\n{'='*80}\nREFERENCES\n{'='*80}\n{references}"

        return document

    def write_document_from_files(
        self,
        topic: str,
        style: str = "academic",
        length: str = "long",
        files: List[str] = None
    ) -> str:
        """
        Write a comprehensive document on a topic using a list of files

        Args:
            topic: The topic/title for the document
            style: Writing style - "academic" (formal research article) or "grant" (grant proposal)
            length: Document length - "short" (~500 words), "medium" (~1000 words), "long" (~2000+ words)
            files: List of file paths to use as sources

        Returns:
            Formatted document with inline citations and references
        """
        papers = []
        for file_path in files:
            # This is a simplified way to get paper info. 
            # A more robust implementation would extract text and metadata.
            papers.append({
                'filename': Path(file_path).name,
                'text': Path(file_path).read_text(),
                'full_path': file_path
            })

        if not papers:
            return "No files provided."

        # Prepare numbered context from papers with reference list
        context_parts = []
        reference_list = []

        for i, p in enumerate(papers, 1):
            # Send full chunk for maximum context (avg ~2000 chars with semantic chunking)
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

        # Define style-specific guidance
        if style == "grant":
            style_guidance = """
GRANT PROPOSAL STYLE:
- Write in a compelling, persuasive tone appropriate for grant applications
- Emphasize significance, innovation, and impact
- Use clear topic sentences and strong transitions
- Highlight knowledge gaps and unmet needs
- Frame the research in terms of broader impacts
- Use active voice and confident language
- Structure: Background → Knowledge Gaps → Significance → Future Directions"""
        else:  # academic
            style_guidance = """
FORMAL ACADEMIC STYLE:
- Write in formal, objective academic prose
- Use complex sentence structures and sophisticated vocabulary
- Maintain third-person perspective
- Present balanced analysis of the literature
- Discuss methodologies, findings, and implications
- Acknowledge limitations and controversies
- Structure: Introduction → Current Understanding → Recent Advances → Future Directions"""

        # Create comprehensive writing prompt
        prompt = f"""You are writing a comprehensive scholarly document on the topic: "{topic}"

Based EXCLUSIVELY on the research papers from my library provided below, write a {length_guidance[length]} document.

Research Papers from Library:
{context}

DOCUMENT REQUIREMENTS:

{style_guidance}

LENGTH: Write {length_guidance[length]}

CRITICAL CITATION REQUIREMENTS:
- After EVERY statement, claim, or fact, include an inline citation using [1], [2], etc.
- Use the paper numbers from the excerpts above
- Cite multiple papers when appropriate: [1, 3, 5]
- Be EXTREMELY generous with citations - every sentence should have at least one citation
- Never make unsupported statements

CONTENT REQUIREMENTS:
- Base EVERY statement on the provided research excerpts
- Synthesize information across multiple papers
- Identify patterns, consensus, and disagreements in the literature
- Discuss methodologies where relevant
- Highlight key findings and their implications
- Identify knowledge gaps
- Do NOT add any information not present in the excerpts
- If the papers don't adequately cover the topic, state: "The available papers in your library provide limited coverage of [aspect]. Based on the available literature..."

STRUCTURE:
- Start with a strong opening that frames the topic
- Develop ideas logically across multiple paragraphs
- Use clear topic sentences for each paragraph
- Build to a conclusion about current state and future directions
- End with a "REFERENCES" section that lists the exact PDF filenames for each citation number

REFERENCES SECTION FORMAT (CRITICAL - COPY THESE EXACTLY):
At the end of your document, include this EXACT reference section by copying the filenames below:

REFERENCES
================================================================================
{references}

IMPORTANT: Simply COPY-PASTE the filenames above into your references section.
Do NOT modify the filenames. Do NOT create placeholder names like "filename1.pdf".
Use the EXACT filenames shown above for each [1], [2], [3], etc.

Write the complete document now:"""

        # Query Ollama
        print(f"\nGenerating {length} {style} document with {self.llm_model}...")
        print(f"Using {len(papers)} papers from your library")
        response = ollama.chat(
            model=self.llm_model,
            messages=[{'role': 'user', 'content': prompt}]
        )

        document = response['message']['content']

        # Append reference list if not already included
        if "References" not in document and "REFERENCES" not in document:
            document += f"\n\n{'='*80}\nREFERENCES\n{'='*80}\n{references}"

        return document



if __name__ == "__main__":
    # Example usage
    assistant = CitationAssistant(
        embeddings_dir="/fastpool/rag_embeddings"
    )

    # Example 1: Search for papers
    print("\n" + "=" * 80)
    print("EXAMPLE 1: Search for papers on a topic")
    print("=" * 80)
    query = "machine learning for genomics"
    papers = assistant.search_papers(query, n_results=5)
    print(f"\nTop 5 papers for query: '{query}'\n")
    for i, paper in enumerate(papers, 1):
        print(f"{i}. {paper['filename']} (similarity: {paper['similarity']:.2%})")
        print(f"   {paper['text'][:150]}...\n")

    # Example 2: Summarize research
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Summarize research on a topic")
    print("=" * 80)
    summary = assistant.summarize_research("microbiome analysis methods")
    print(summary)

    # Example 3: Suggest citations for manuscript
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Suggest citations for manuscript excerpt")
    print("=" * 80)
    manuscript_sample = """
    Recent advances in next-generation sequencing have revolutionized our understanding
    of the human microbiome. Metagenomic approaches allow us to characterize entire
    microbial communities without cultivation. Machine learning techniques have shown
    promise in predicting disease states from microbiome profiles.
    """
    suggestions = assistant.suggest_citations_for_manuscript(manuscript_sample)
    print(assistant.format_citation_suggestions(suggestions))
