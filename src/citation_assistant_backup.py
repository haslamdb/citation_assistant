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

    def _are_chunks_duplicate(self, text1: str, text2: str, threshold: float = 0.95) -> bool:
        """
        Check if two chunks are likely from duplicate papers based on content similarity

        This is more robust than filename matching because:
        - Uses actual content similarity (semantic embeddings)
        - Doesn't rely on filename conventions
        - Catches duplicate papers even with different filenames

        Args:
            text1: Text content of first chunk
            text2: Text content of second chunk
            threshold: Minimum cosine similarity to consider as duplicates (default: 0.95 - very high similarity)

        Returns:
            True if chunks have very high similarity (likely from duplicate papers)
        """
        # Encode both texts - use more text for better comparison
        embeddings = self.embedding_model.encode([text1[:2000], text2[:2000]])  # Use first 2000 chars for better accuracy
        
        # Calculate cosine similarity
        from numpy import dot
        from numpy.linalg import norm
        cosine_sim = dot(embeddings[0], embeddings[1]) / (norm(embeddings[0]) * norm(embeddings[1]))
        
        # Very high similarity indicates duplicate papers (0.95 = nearly identical)
        return cosine_sim > threshold

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
        use_reranking: bool = None,
        check_duplicates: bool = True
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
            check_duplicates: If True, check for duplicate papers based on content similarity (default: True)
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

            # Check if we have a paper with very similar content (likely duplicate/same paper)
            # This is more robust than filename matching
            duplicate_found = False
            if check_duplicates:
                for existing_filename, existing_paper in list(unique_papers.items()):
                    # Skip if it's the same filename
                    if existing_filename == filename:
                        continue
                        
                    # Check content similarity
                    if self._are_chunks_duplicate(text, existing_paper['text']):
                        # Found a paper with very similar content - likely a duplicate
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
        keyword_boost_strength: float = None,
        llm_model: str = None,
        check_duplicates: bool = False
    ) -> str:
        """Summarize research findings on a topic with inline citations

        Args:
            query: Search query string
            n_papers: Number of papers to use for summary (default: 10, leverages Gemma2's 8K context)
            fetch_multiplier: Multiplier for initial chunk fetch (None = use default)
            keyword_boost_strength: Boost factor for keyword matches (None = use default)
            check_duplicates: If True, check for duplicate papers (default: False)
        """
        papers = self.search_papers(
            query,
            n_results=n_papers,
            fetch_multiplier=fetch_multiplier,
            keyword_boost_strength=keyword_boost_strength,
            check_duplicates=check_duplicates
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
        # Use provided model or fall back to instance default
        model_to_use = llm_model if llm_model else self.llm_model
        print(f"\nQuerying {model_to_use}...")
        response = ollama.chat(
            model=model_to_use,
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
        keyword_boost_strength: float = None,
        llm_model: str = None,
        use_reranking: bool = False,
        check_duplicates: bool = False
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

        # Use provided model or fall back to instance default
        model_to_use = llm_model if llm_model else self.llm_model
        print(f"\nAnalyzing manuscript with {model_to_use}...")
        response = ollama.chat(
            model=model_to_use,
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
                keyword_boost_strength=keyword_boost_strength,
                use_reranking=use_reranking,
                check_duplicates=check_duplicates
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
        keyword_boost_strength: float = None,
        llm_model: str = None,
        search_method: str = "multi_chunk",
        use_reranking: bool = False,
        chunks_per_paper: int = 3,
        check_duplicates: bool = False
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
            llm_model: LLM model to use for generation
            search_method: Method for searching papers - "default", "multi_chunk", "factual"
            use_reranking: If True, use cross-encoder re-ranking for better precision
            check_duplicates: If True, check for duplicate papers (default: False)

        Returns:
            Formatted document with inline citations and references
        """
        # Search for relevant papers using specified method
        if search_method == "multi_chunk":
            # Build kwargs dict for optional parameters
            kwargs = {}
            if fetch_multiplier is not None:
                kwargs['fetch_multiplier'] = fetch_multiplier
            kwargs['use_reranking'] = use_reranking
            
            papers = self.search_papers_multi_chunk(
                topic,
                n_results=n_papers,
                chunks_per_paper=chunks_per_paper,
                check_duplicates=check_duplicates,
                **kwargs
            )
        elif search_method == "factual":
            # Build kwargs dict for optional parameters
            kwargs = {}
            if fetch_multiplier is not None:
                kwargs['fetch_multiplier'] = fetch_multiplier
            # Note: factual search doesn't use reranking (has its own keyword boosting)
                
            papers = self.search_papers_factual(
                topic,
                n_results=n_papers,
                check_duplicates=check_duplicates,
                **kwargs
            )
        else:
            papers = self.search_papers(
                topic,
                n_results=n_papers,
                boost_keywords=keywords,
                fetch_multiplier=fetch_multiplier,
                keyword_boost_strength=keyword_boost_strength,
                use_reranking=use_reranking,
                check_duplicates=check_duplicates
            )

        if not papers:
            return "No relevant papers found in your library for this topic."
        
        # Deduplicate similar papers (vector similarity threshold)
        if len(papers) > 1:
            import numpy as np
            from sentence_transformers import util
            
            # Encode paper texts for similarity comparison
            paper_embeddings = self.embedding_model.encode([p['text'][:1000] for p in papers])
            
            # Keep track of papers to keep
            keep_indices = [0]  # Always keep the first (best) paper
            
            for i in range(1, len(papers)):
                # Check similarity with all kept papers
                is_duplicate = False
                for kept_idx in keep_indices:
                    similarity = util.cos_sim(paper_embeddings[i], paper_embeddings[kept_idx])[0][0].item()
                    if similarity > 0.9:  # 90% similarity threshold
                        is_duplicate = True
                        break
                
                if not is_duplicate:
                    keep_indices.append(i)
            
            # Filter papers
            papers = [papers[i] for i in keep_indices]
            print(f"Deduplicated from {len(paper_embeddings)} to {len(papers)} unique papers")

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
        # Use provided model or fall back to instance default
        model_to_use = llm_model if llm_model else self.llm_model
        print(f"\nGenerating {length} {style} document with {model_to_use}...")
        print(f"Using {len(papers)} papers from your library")
        response = ollama.chat(
            model=model_to_use,
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
        # Use provided model or fall back to instance default
        model_to_use = llm_model if llm_model else self.llm_model
        print(f"\nGenerating {length} {style} document with {model_to_use}...")
        print(f"Using {len(papers)} papers from your library")
        response = ollama.chat(
            model=model_to_use,
            messages=[{'role': 'user', 'content': prompt}]
        )

        document = response['message']['content']

        # Append reference list if not already included
        if "References" not in document and "REFERENCES" not in document:
            document += f"\n\n{'='*80}\nREFERENCES\n{'='*80}\n{references}"

        return document

    def search_papers_multi_chunk(
        self,
        query: str,
        n_results: int = 10,
        chunks_per_paper: int = 2,
        check_duplicates: bool = False,
        **kwargs
    ) -> List[Dict]:
        """
        Retrieve multiple chunks per paper instead of just the best one.
        
        This fixes the golgicide discovery retrieval issue by preserving
        discovery narratives that score lower than mechanism details.
        
        Args:
            query: Search query
            n_results: Number of papers to return
            chunks_per_paper: Chunks per paper to retrieve (recommend 2-3)
            check_duplicates: If True, check for duplicate papers (default: False - less strict)
            **kwargs: Additional parameters (fetch_multiplier, max_fetch, use_reranking)
        
        Returns:
            List of dicts with keys: filename, text, similarity, num_chunks
        """
        # Use instance defaults if not specified
        fetch_multiplier = kwargs.get('fetch_multiplier', self.default_fetch_multiplier)
        max_fetch = kwargs.get('max_fetch', self.default_max_fetch)
        use_reranking = kwargs.get('use_reranking', self.enable_reranking)
        
        # Embed the query
        query_embedding = self.embedding_model.encode([query])[0]
        
        # Fetch more chunks upfront (need more to have options)
        fetch_count = min(n_results * fetch_multiplier * chunks_per_paper, max_fetch)
        
        # Query ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=fetch_count,
            include=["documents", "metadatas", "distances"]
        )
        
        # Group ALL chunks by filename first
        papers_all_chunks = {}  # filename -> list of ALL chunks
        
        for i in range(len(results['ids'][0])):
            filename = results['metadatas'][0][i]['filename']
            text = results['documents'][0][i]
            distance = results['distances'][0][i]
            chunk_index = results['metadatas'][0][i].get('chunk_index', i)
            
            if filename not in papers_all_chunks:
                papers_all_chunks[filename] = []
            
            papers_all_chunks[filename].append({
                'text': text,
                'distance': distance,
                'similarity': 1 / (1 + distance),
                'chunk_index': chunk_index,
                'metadata': results['metadatas'][0][i]
            })
        
        # Now for each paper, select diverse chunks (not just best scoring)
        papers_chunks = {}
        for filename, chunks in papers_all_chunks.items():
            if len(chunks) <= chunks_per_paper:
                # If we have fewer chunks than requested, keep all
                papers_chunks[filename] = chunks
            else:
                # Select diverse chunks: best chunk + evenly spaced others
                # This ensures we get content from different sections
                chunks.sort(key=lambda x: x['chunk_index'])  # Sort by position in document
                
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
                
                # Sort selected chunks by document position for readability
                selected.sort(key=lambda x: x['chunk_index'])
                papers_chunks[filename] = selected
        
        # Flatten results and remove duplicates
        flat_results = []
        seen_papers = {}  # Track papers we've already added
        
        for filename, chunks in papers_chunks.items():
            # Find best chunk for this paper (for ranking)
            best_chunk = min(chunks, key=lambda x: x['distance'])
            
            # Combine all chunks with separators and metadata
            combined_text = "\n\n---[Next Chunk]---\n\n".join([
                f"[Chunk {c['chunk_index']}]\n{c['text']}"
                for c in chunks
            ])
            
            # Check for duplicates with already added papers (if enabled)
            is_duplicate = False
            if check_duplicates:
                duplicate_of = None
                for seen_filename, seen_text in seen_papers.items():
                    if self._are_chunks_duplicate(chunks[0]['text'], seen_text):
                        is_duplicate = True
                        duplicate_of = seen_filename
                        break
            
            if not is_duplicate:
                flat_results.append({
                    'filename': filename,
                    'text': combined_text,
                    'similarity': best_chunk['similarity'],
                    'distance': best_chunk['distance'],
                    'num_chunks': len(chunks),
                    'source': best_chunk['metadata'].get('source', ''),
                    'has_haslam': 'haslam' in combined_text.lower(),
                    'keyword_matches': sum(1 for term in query.lower().split() 
                                          if len(term) > 3 and term in combined_text.lower())
                })
                # Remember this paper for duplicate checking
                seen_papers[filename] = chunks[0]['text']
        
        # Apply cross-encoder re-ranking if enabled
        if use_reranking and self.reranker is not None and len(flat_results) > 0:
            # Re-rank with cross-encoder for improved precision
            rerank_candidates = flat_results[:min(len(flat_results), n_results * 3)]
            
            # Create query-document pairs for cross-encoder
            pairs = [[query, paper['text']] for paper in rerank_candidates]
            
            # Get cross-encoder scores
            rerank_scores = self.reranker.predict(pairs)
            
            # Add re-rank scores to papers
            for i, paper in enumerate(rerank_candidates):
                paper['rerank_score'] = float(rerank_scores[i])
            
            # Re-sort by cross-encoder scores (descending)
            flat_results = sorted(rerank_candidates, key=lambda x: x['rerank_score'], reverse=True)
            
            # Append remaining papers (not re-ranked) if any
            if len(flat_results) < len(papers_chunks):
                remaining = [p for p in flat_results if 'rerank_score' not in p]
                flat_results.extend(sorted(remaining, key=lambda x: x['distance']))
        else:
            # Sort by similarity and return top N
            flat_results.sort(key=lambda x: x['similarity'], reverse=True)
        
        return flat_results[:n_results]

    def search_papers_factual(
        self,
        query: str,
        n_results: int = 10,
        check_duplicates: bool = False,
        **kwargs
    ) -> List[Dict]:
        """
        Specialized search for factual/discovery queries.
        
        Boosts chunks containing discovery-related keywords:
        "discovered", "first identified", "novel", "shown", "found"
        
        Use this for queries like:
        - "How was golgicide discovered?"
        - "First identification of X"
        - "Discovery of the mechanism"
        
        Args:
            query: Search query
            n_results: Number of papers to return
            check_duplicates: If True, check for duplicate papers (default: False - less strict)
            **kwargs: Additional parameters
        
        Returns:
            List of dicts with keys: filename, text, similarity, num_chunks
        """
        # Discovery keywords to boost
        discovery_keywords = [
            "discovered", "discovery", "first identified", "first described",
            "novel", "shown", "found", "identified", "demonstrated",
            "introduction of", "history of", "screen", "screening",
            "high-throughput", "isolated", "purified", "synthesized"
        ]
        
        # Use instance defaults if not specified
        fetch_multiplier = kwargs.get('fetch_multiplier', 100)  # More aggressive for factual
        max_fetch = kwargs.get('max_fetch', 5000)
        
        # Embed query
        query_embedding = self.embedding_model.encode([query])[0]
        
        # Initial retrieval with aggressive fetch
        fetch_count = min(n_results * fetch_multiplier, max_fetch)
        
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=fetch_count,
            include=["documents", "metadatas", "distances"]
        )
        
        # Score chunks and boost for discovery keywords
        papers_chunks = {}
        
        for i in range(len(results['ids'][0])):
            filename = results['metadatas'][0][i]['filename']
            text = results['documents'][0][i]
            distance = results['distances'][0][i]
            
            # Count keyword matches (case-insensitive)
            text_lower = text.lower()
            keyword_matches = 0
            for keyword in discovery_keywords:
                if keyword in text_lower:
                    keyword_matches += text_lower.count(keyword)
            
            # Also check for query terms
            query_terms = set(query.lower().split())
            query_matches = sum(1 for term in query_terms if len(term) > 3 and term in text_lower)
            
            # Adjust distance: reduce (improve rank) for keyword matches
            adjusted_distance = distance
            if keyword_matches > 0:
                # Exponential boost: more keywords = stronger boost
                adjusted_distance = distance * (0.5 ** min(keyword_matches, 3))
            if query_matches > 0:
                adjusted_distance = adjusted_distance * (0.7 ** query_matches)
            
            if filename not in papers_chunks:
                papers_chunks[filename] = []
            
            papers_chunks[filename].append({
                'text': text,
                'distance': distance,
                'adjusted_distance': adjusted_distance,
                'similarity': 1 / (1 + adjusted_distance),
                'keyword_matches': keyword_matches,
                'query_matches': query_matches,
                'metadata': results['metadatas'][0][i]
            })
        
        # Keep best 2 chunks per paper (adjusted ranking) and remove duplicates
        papers_best = {}
        seen_papers = {}  # Track papers for duplicate detection
        
        for filename, chunks in papers_chunks.items():
            # Sort by adjusted distance
            chunks.sort(key=lambda x: x['adjusted_distance'])
            # Take top 2 chunks
            top_chunks = chunks[:2]
            
            # Check for duplicates before adding (if enabled)
            is_duplicate = False
            duplicate_to_remove = None
            
            if check_duplicates:
                for seen_filename, seen_text in list(seen_papers.items()):  # Convert to list to avoid modification during iteration
                    if self._are_chunks_duplicate(top_chunks[0]['text'], seen_text):
                        # This is a duplicate - only keep if it has a better score
                        if top_chunks[0]['adjusted_distance'] < papers_best[seen_filename]['distance']:
                            # This duplicate is better, mark the old one for removal
                            duplicate_to_remove = seen_filename
                            break
                        else:
                            # Existing paper is better, skip this one
                            is_duplicate = True
                            break
                
                # Remove the old duplicate if needed
                if duplicate_to_remove:
                    del papers_best[duplicate_to_remove]
                    del seen_papers[duplicate_to_remove]
            
            if not is_duplicate:
                # Combine text from multiple chunks
                combined_text = "\n\n---[Discovery Context]---\n\n".join([c['text'] for c in top_chunks])
                
                # Use best chunk's score for ranking
                best_chunk = top_chunks[0]
                
                papers_best[filename] = {
                    'filename': filename,
                    'text': combined_text,
                    'similarity': best_chunk['similarity'],
                    'distance': best_chunk['adjusted_distance'],
                    'num_chunks': len(top_chunks),
                    'keywords_found': sum(c['keyword_matches'] for c in top_chunks),
                    'source': best_chunk['metadata'].get('source', ''),
                    'has_haslam': 'haslam' in combined_text.lower()
                }
                # Remember this paper for duplicate checking
                seen_papers[filename] = top_chunks[0]['text']
        
        # Sort by adjusted similarity
        sorted_papers = sorted(papers_best.values(),
                             key=lambda x: x['similarity'],
                             reverse=True)
        
        return sorted_papers[:n_results]



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
