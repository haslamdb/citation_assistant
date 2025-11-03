#!/bin/bash
# Script to check progress of background tasks

echo "=========================================="
echo "Citation Assistant - Progress Check"
echo "=========================================="
echo ""

# Check Qwen model download
echo "1. Qwen 2.5 72B Model Download:"
if [ -d ~/models/manifests/registry.ollama.ai/library/qwen2.5 ]; then
    echo "   ✓ Model directory exists"
    du -sh ~/models/manifests/registry.ollama.ai/library/qwen2.5 2>/dev/null || echo "   Size: checking..."
else
    echo "   ⏳ Not started or in progress"
fi
echo ""

# Check BM25 index build
echo "2. BM25 Index Build:"
if [ -f /fastpool/rag_embeddings/bm25_index.pkl ]; then
    size=$(du -h /fastpool/rag_embeddings/bm25_index.pkl | cut -f1)
    echo "   ✓ COMPLETE! Index file exists ($size)"
elif screen -list | grep -q bm25_build; then
    echo "   ⏳ Running in screen session 'bm25_build'"
    if [ -f /home/david/projects/citation_assistant/bm25_build.log ]; then
        echo "   Last log entries:"
        tail -5 /home/david/projects/citation_assistant/bm25_build.log | sed 's/^/      /'
    else
        echo "   (Log file not yet created)"
    fi
else
    echo "   ❌ Not running"
fi
echo ""

# Screen sessions
echo "3. Active Screen Sessions:"
screen -list | grep -E "bm25_build|nicu_batch" | sed 's/^/   /'
echo ""

echo "=========================================="
echo "Commands:"
echo "  - Reconnect to BM25 build: screen -r bm25_build"
echo "  - View BM25 log: tail -f ~/projects/citation_assistant/bm25_build.log"
echo "  - Check Qwen model: ollama list | grep qwen"
echo "=========================================="
