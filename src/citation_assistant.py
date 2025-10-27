#!/usr/bin/env python3
"""
Citation Assistant
Search for relevant citations and suggest references for manuscripts
"""

import os
from pathlib import Path
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import ollama


class CitationAssistant:
    """Search research papers and suggest citations for manuscripts"""

    def __init__(
        self,
        embeddings_dir: str,
        collection_name: str = "research_papers",
        embedding_model: str = "pritamdeka/S-PubMedBert-MS-MARCO",
        llm_model: str = "gemma2:27b"
    ):
        self.embeddings_dir = Path(embeddings_dir)
        self.collection_name = collection_name
        self.llm_model = llm_model

        # Initialize embedding model
        print(f"Loading embedding model: {embedding_model}")
        self.embedding_model = SentenceTransformer(embedding_model)

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

    def search_papers(self, query: str, n_results: int = 10, boost_haslam: bool = True, boost_keywords: str = "") -> List[Dict]:
        """Search for relevant papers given a query (deduplicated by filename)

        Args:
            query: Search query string
            n_results: Number of unique papers to return
            boost_haslam: If True, boost papers with "Haslam" as author (default: True)
            boost_keywords: Optional comma-separated keywords for aggressive boosting (e.g., "golgicide, brefeldin")
        """
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query])[0]

        # Search ChromaDB with more results to account for deduplication
        # Fetch more results to find papers with specific keywords
        fetch_count = min(n_results * 10, 500)
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
        unique_papers = {}
        for i in range(len(results['ids'][0])):
            filename = results['metadatas'][0][i]['filename']
            distance = results['distances'][0][i]
            text = results['documents'][0][i]
            text_lower = text.lower()

            # If this paper hasn't been seen, or this chunk is better than the previous one
            if filename not in unique_papers or distance < unique_papers[filename]['distance']:
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
                    'similarity': 1 - distance,  # Convert distance to similarity
                    'has_haslam': has_haslam,
                    'keyword_matches': keyword_matches
                }

        # Apply keyword boost (stronger than Haslam boost for relevance)
        for paper in unique_papers.values():
            if paper['keyword_matches'] > 0:
                # Aggressively boost papers with keyword matches
                # Reduce distance significantly for each keyword match
                paper['distance'] *= 0.1 ** paper['keyword_matches']
                paper['similarity'] = 1 - paper['distance']

        # Apply Haslam boost to distances if enabled
        if boost_haslam:
            for paper in unique_papers.values():
                if paper['has_haslam']:
                    # Reduce distance by 50% (strong boost for Haslam papers)
                    paper['distance'] = paper['distance'] * 0.5
                    paper['similarity'] = 1 - paper['distance']

        # Convert to list and sort by distance (best matches first)
        papers = sorted(unique_papers.values(), key=lambda x: x['distance'])

        # Return only the requested number of results
        return papers[:n_results]

    def summarize_research(self, query: str, n_papers: int = 5) -> str:
        """Summarize research findings on a topic with inline citations"""
        papers = self.search_papers(query, n_results=n_papers)

        if not papers:
            return "No relevant papers found in your library."

        # Prepare numbered context from papers with reference list
        context_parts = []
        reference_list = []

        for i, p in enumerate(papers[:n_papers], 1):
            context_parts.append(f"[{i}] {p['filename']}\nExcerpt: {p['text'][:800]}...")
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
        n_suggestions_per_statement: int = 3
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
            papers = self.search_papers(claim, n_results=n_suggestions_per_statement)

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
        keywords: str = ""
    ) -> str:
        """
        Write a comprehensive document on a topic using only papers from the library

        Args:
            topic: The topic/title for the document
            style: Writing style - "academic" (formal research article) or "grant" (grant proposal)
            length: Document length - "short" (~500 words), "medium" (~1000 words), "long" (~2000+ words)
            n_papers: Number of papers to use as sources (default: 15)
            keywords: Optional comma-separated keywords for aggressive boosting (e.g., "golgicide, brefeldin")

        Returns:
            Formatted document with inline citations and references
        """
        # Search for relevant papers
        papers = self.search_papers(topic, n_results=n_papers, boost_keywords=keywords)

        if not papers:
            return "No relevant papers found in your library for this topic."

        # Prepare numbered context from papers with reference list
        context_parts = []
        reference_list = []

        for i, p in enumerate(papers, 1):
            # Use longer excerpts for document writing (1000 chars)
            context_parts.append(f"[{i}] {p['filename']}\nExcerpt: {p['text'][:1000]}...")
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
            # Use longer excerpts for document writing (1000 chars)
            context_parts.append(f"[{i}] {p['filename']}\nExcerpt: {p['text'][:1000]}...")
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
