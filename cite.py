#!/usr/bin/env python3
"""
Simple CLI for Citation Assistant
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from pdf_indexer import PDFIndexer
from citation_assistant import CitationAssistant


def cmd_index(args):
    """Index EndNote library"""
    indexer = PDFIndexer(
        endnote_pdf_dir=args.pdf_dir,
        embeddings_dir=args.embeddings_dir
    )

    if args.stats:
        stats = indexer.get_stats()
        print("\nCurrent Index Statistics:")
        print("=" * 60)
        for key, value in stats.items():
            print(f"  {key}: {value}")
        return

    print("\nIndexing EndNote Library")
    print("=" * 60)
    results = indexer.index_all_new()


def cmd_search(args):
    """Search for papers"""
    assistant = CitationAssistant(embeddings_dir=args.embeddings_dir)

    print(f"\nSearching for: '{args.query}'")
    print("=" * 80)

    papers = assistant.search_papers(args.query, n_results=args.num)

    if not papers:
        print("No papers found.")
        return

    for i, paper in enumerate(papers, 1):
        print(f"\n{i}. {paper['filename']}")
        print(f"   Similarity: {paper['similarity']:.2%}")
        print(f"   Source: {paper['source']}")
        if args.verbose:
            print(f"   Excerpt: {paper['text'][:300]}...")


def cmd_summarize(args):
    """Summarize research on a topic"""
    assistant = CitationAssistant(embeddings_dir=args.embeddings_dir)

    print(f"\nSummarizing research on: '{args.query}'")
    print("=" * 80)
    print("(This may take a minute...)\n")

    summary = assistant.summarize_research(args.query, n_papers=args.num)
    print(summary)


def cmd_suggest(args):
    """Suggest citations for manuscript"""
    assistant = CitationAssistant(embeddings_dir=args.embeddings_dir)

    # Read manuscript
    manuscript_path = Path(args.manuscript)
    if not manuscript_path.exists():
        print(f"Error: File not found: {args.manuscript}")
        return

    with open(manuscript_path, 'r') as f:
        manuscript_text = f.read()

    print(f"\nAnalyzing manuscript: {manuscript_path.name}")
    print("=" * 80)
    print("(This may take a few minutes...)\n")

    suggestions = assistant.suggest_citations_for_manuscript(
        manuscript_text,
        n_suggestions_per_statement=args.num
    )

    output = assistant.format_citation_suggestions(suggestions)
    print(output)

    # Optionally save to file
    if args.output:
        output_path = Path(args.output)
        with open(output_path, 'w') as f:
            f.write(output)
        print(f"\nSuggestions saved to: {output_path}")


def cmd_write_doc(args):
    """Write a document on a topic"""
    assistant = CitationAssistant(embeddings_dir=args.embeddings_dir)

    print(f"\nWriting document on: '{args.topic}'")
    print("=" * 80)
    print("(This may take a few minutes...)\n")

    document = assistant.write_document(
        topic=args.topic,
        style=args.style,
        length=args.length,
        n_papers=args.n_papers,
        keywords=args.keywords
    )
    print(document)


def main():
    parser = argparse.ArgumentParser(
        description="Citation Assistant - AI-powered research paper management",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Index EndNote library (run after adding new papers)
  %(prog)s index

  # Show index statistics
  %(prog)s index --stats

  # Search for papers
  %(prog)s search "CRISPR gene editing"

  # Summarize research on a topic
  %(prog)s summarize "microbiome analysis methods"

  # Suggest citations for a manuscript
  %(prog)s suggest my_manuscript.txt

  # Save suggestions to file
  %(prog)s suggest my_manuscript.txt -o citations.txt

  # Write a document
  %(prog)s write-doc "The role of AI in drug discovery" --style academic --length long --n-papers 20 --keywords "AI, drug discovery"
        """
    )

    # Default paths
    default_pdf_dir = "/home/david/projects/EndNote_Library/PDF"
    default_embeddings_dir = "/fastpool/rag_embeddings"

    # Global options
    parser.add_argument(
        '--pdf-dir',
        default=default_pdf_dir,
        help=f"EndNote PDF directory (default: {default_pdf_dir})"
    )
    parser.add_argument(
        '--embeddings-dir',
        default=default_embeddings_dir,
        help=f"Embeddings directory (default: {default_embeddings_dir})"
    )

    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # Index command
    parser_index = subparsers.add_parser('index', help='Index EndNote library')
    parser_index.add_argument('--stats', action='store_true', help='Show statistics only')
    parser_index.set_defaults(func=cmd_index)

    # Search command
    parser_search = subparsers.add_parser('search', help='Search for papers')
    parser_search.add_argument('query', help='Search query')
    parser_search.add_argument('-n', '--num', type=int, default=10, help='Number of results (default: 10)')
    parser_search.add_argument('-v', '--verbose', action='store_true', help='Show excerpts')
    parser_search.set_defaults(func=cmd_search)

    # Summarize command
    parser_summarize = subparsers.add_parser('summarize', help='Summarize research on a topic')
    parser_summarize.add_argument('query', help='Research topic')
    parser_summarize.add_argument('-n', '--num', type=int, default=5, help='Number of papers to use (default: 5)')
    parser_summarize.set_defaults(func=cmd_summarize)

    # Suggest command
    parser_suggest = subparsers.add_parser('suggest', help='Suggest citations for manuscript')
    parser_suggest.add_argument('manuscript', help='Path to manuscript file')
    parser_suggest.add_argument('-n', '--num', type=int, default=3, help='Citations per statement (default: 3)')
    parser_suggest.add_argument('-o', '--output', help='Save suggestions to file')
    parser_suggest.set_defaults(func=cmd_suggest)

    # Write doc command
    parser_write_doc = subparsers.add_parser('write-doc', help='Write a document on a topic')
    parser_write_doc.add_argument('topic', help='Topic for the document')
    parser_write_doc.add_argument('--style', default='academic', help='Writing style (academic or grant)')
    parser_write_doc.add_argument('--length', default='long', help='Document length (short, medium, or long)')
    parser_write_doc.add_argument('--n-papers', type=int, default=15, help='Number of papers to use')
    parser_write_doc.add_argument('--keywords', default='', help='Keywords for aggressive boosting')
    parser_write_doc.set_defaults(func=cmd_write_doc)

    # Parse arguments
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    # Run command
    args.func(args)


if __name__ == "__main__":
    main()
