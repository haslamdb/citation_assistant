#!/usr/bin/env python3
"""
Test script to compare different LLM models for summarization quality
Tests Gemma2:27b vs Qwen2.5:72b vs other models
"""

import sys
from pathlib import Path
import time

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from citation_assistant import CitationAssistant


def test_model_comparison():
    """Compare different LLM models on the same query"""

    test_query = "antibiotic resistance mechanisms in gut bacteria"
    n_papers = 5

    # Models to test
    models = [
        ("gemma2:27b", "Gemma 2 27B (current)"),
        ("qwen2.5:72b-instruct-q4_K_M", "Qwen 2.5 72B (new, SOTA for RAG)"),
    ]

    print("=" * 80)
    print("LLM MODEL COMPARISON TEST")
    print("=" * 80)
    print(f"\nTest query: '{test_query}'")
    print(f"Papers to use: {n_papers}")
    print("\n" + "=" * 80)

    results = {}

    for model_name, description in models:
        print(f"\n{'='*80}")
        print(f"Testing: {description}")
        print(f"Model: {model_name}")
        print("=" * 80)

        try:
            # Initialize assistant with this model
            print(f"\nLoading {model_name}...")
            assistant = CitationAssistant(
                embeddings_dir="/fastpool/rag_embeddings",
                llm_model=model_name,
                enable_reranking=False
            )

            # Time the summarization
            start_time = time.time()
            summary = assistant.summarize_research(test_query, n_papers=n_papers)
            elapsed_time = time.time() - start_time

            results[model_name] = {
                "description": description,
                "summary": summary,
                "time": elapsed_time,
                "success": True
            }

            print(f"\n✓ Summary generated in {elapsed_time:.1f} seconds")
            print(f"\nSummary length: {len(summary)} characters")
            print(f"Summary preview (first 500 chars):")
            print("-" * 80)
            print(summary[:500] + "...")
            print("-" * 80)

        except Exception as e:
            print(f"\n✗ Error with {model_name}: {e}")
            results[model_name] = {
                "description": description,
                "summary": None,
                "time": None,
                "success": False,
                "error": str(e)
            }

    # Print comparison summary
    print("\n" + "=" * 80)
    print("COMPARISON SUMMARY")
    print("=" * 80)

    for model_name, result in results.items():
        if result["success"]:
            print(f"\n{result['description']}:")
            print(f"  Model: {model_name}")
            print(f"  Time: {result['time']:.1f}s")
            print(f"  Length: {len(result['summary'])} chars")

            # Count citations
            import re
            citations = len(re.findall(r'\[\d+\]', result['summary']))
            print(f"  Citations: {citations}")
        else:
            print(f"\n{result['description']}: FAILED")
            print(f"  Error: {result.get('error', 'Unknown')}")

    print("\n" + "=" * 80)
    print("QUALITY ASSESSMENT")
    print("=" * 80)
    print("\nPlease compare the summaries above and assess:")
    print("1. Comprehensiveness - Does it cover the key points?")
    print("2. Citation quality - Are citations used appropriately?")
    print("3. Biomedical accuracy - Does it understand the domain?")
    print("4. Clarity - Is it well-written and coherent?")
    print()

    # Save full results for review
    output_file = Path(__file__).parent / "llm_comparison_results.txt"
    with open(output_file, 'w') as f:
        f.write("LLM MODEL COMPARISON RESULTS\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Query: {test_query}\n")
        f.write(f"Papers used: {n_papers}\n\n")

        for model_name, result in results.items():
            f.write("=" * 80 + "\n")
            f.write(f"{result['description']}\n")
            f.write(f"Model: {model_name}\n")
            f.write("=" * 80 + "\n\n")

            if result["success"]:
                f.write(f"Time: {result['time']:.1f} seconds\n")
                f.write(f"Length: {len(result['summary'])} characters\n\n")
                f.write("FULL SUMMARY:\n")
                f.write("-" * 80 + "\n")
                f.write(result['summary'])
                f.write("\n\n")
            else:
                f.write(f"ERROR: {result.get('error', 'Unknown')}\n\n")

    print(f"Full results saved to: {output_file}")
    print()


if __name__ == "__main__":
    test_model_comparison()
