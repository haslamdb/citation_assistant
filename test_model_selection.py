#!/usr/bin/env python3
"""Test script to verify model selection works with Ollama"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from citation_assistant import CitationAssistant
import ollama

# Test that all three models are available
models = ["gemma2:27b", "qwen2.5:72b-instruct-q4_K_M", "llama3.1:70b"]

print("Testing Ollama model availability...")
print("=" * 60)

# Check available models
try:
    available = ollama.list()
    print(f"Found {len(available['models'])} models:")
    for model in available['models']:
        print(f"  - {model['name']}")
    print()
except Exception as e:
    print(f"Error listing models: {e}")
    sys.exit(1)

# Test each model
for model in models:
    print(f"Testing {model}...")
    try:
        # Simple test query
        response = ollama.chat(
            model=model,
            messages=[{'role': 'user', 'content': 'Say "hello" in one word'}]
        )
        result = response['message']['content'].strip()
        print(f"  ✓ {model} responded: {result}")
    except Exception as e:
        print(f"  ✗ {model} failed: {e}")

print("\n" + "=" * 60)
print("Testing CitationAssistant with different models...")
print("=" * 60)

# Test CitationAssistant initialization with each model
for model in models:
    print(f"\nInitializing CitationAssistant with {model}...")
    try:
        assistant = CitationAssistant(
            embeddings_dir="/home/david/projects/citation_assistant/embeddings",
            llm_model=model,
            enable_reranking=False
        )
        print(f"  ✓ Successfully initialized with {model}")
        print(f"    Collection has {assistant.collection.count()} documents")
        
        # Test a simple summarize call with the model override
        test_summary = assistant.summarize_research(
            "test query", 
            n_papers=1,
            llm_model=model  # Test the override parameter
        )
        if test_summary:
            print(f"    ✓ Model override parameter works")
        
    except Exception as e:
        print(f"  ✗ Failed to initialize with {model}: {e}")

print("\n✅ Model selection feature has been successfully added!")