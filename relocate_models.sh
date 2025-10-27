#!/bin/bash
# Relocate models to ~/models for better organization
# Run this AFTER re-indexing completes

set -e  # Exit on error

MODELS_DIR="$HOME/models"
HF_MODELS_DIR="$MODELS_DIR/huggingface"
OLLAMA_MODELS_DIR="$MODELS_DIR/ollama"

echo "================================================================"
echo "Model Relocation Script"
echo "================================================================"
echo ""
echo "This will move:"
echo "  1. PubMedBERT → $HF_MODELS_DIR"
echo "  2. Ollama models → $OLLAMA_MODELS_DIR"
echo ""
echo "Current locations:"
echo "  PubMedBERT: ~/.cache/huggingface/"
echo "  Ollama: ~/.ollama/"
echo ""

# Check if re-indexing is still running
if screen -list | grep -q "reindex"; then
    echo "⚠️  WARNING: Re-indexing is still running!"
    echo "   Please wait for it to complete before relocating models."
    echo ""
    read -p "Continue anyway? (yes/no): " response
    if [ "$response" != "yes" ]; then
        echo "Aborted."
        exit 1
    fi
fi

echo ""
read -p "Proceed with model relocation? (yes/no): " response
if [ "$response" != "yes" ]; then
    echo "Aborted."
    exit 0
fi

echo ""
echo "================================================================"
echo "Step 1: Create directory structure"
echo "================================================================"
mkdir -p "$HF_MODELS_DIR"
mkdir -p "$OLLAMA_MODELS_DIR"
echo "✓ Created $MODELS_DIR"

echo ""
echo "================================================================"
echo "Step 2: Move PubMedBERT (HuggingFace models)"
echo "================================================================"

# Check if HuggingFace cache exists
if [ -d "$HOME/.cache/huggingface/hub" ]; then
    echo "Moving HuggingFace cache..."

    # Calculate size
    HF_SIZE=$(du -sh "$HOME/.cache/huggingface/hub" | cut -f1)
    echo "  Size: $HF_SIZE"

    # Move the entire hub directory
    mv "$HOME/.cache/huggingface/hub" "$HF_MODELS_DIR/"

    # Create symlink for backward compatibility
    ln -s "$HF_MODELS_DIR/hub" "$HOME/.cache/huggingface/hub"

    echo "✓ Moved to $HF_MODELS_DIR/hub"
    echo "✓ Created symlink for backward compatibility"
else
    echo "⚠️  HuggingFace cache not found at ~/.cache/huggingface/hub"
fi

echo ""
echo "================================================================"
echo "Step 3: Move Ollama models"
echo "================================================================"

# Check if Ollama models exist
if [ -d "$HOME/.ollama/models" ]; then
    echo "Moving Ollama models..."

    # Calculate size
    OLLAMA_SIZE=$(du -sh "$HOME/.ollama/models" | cut -f1)
    echo "  Size: $OLLAMA_SIZE"

    # Move models directory
    mv "$HOME/.ollama/models" "$OLLAMA_MODELS_DIR/"

    # Create symlink for backward compatibility
    ln -s "$OLLAMA_MODELS_DIR/models" "$HOME/.ollama/models"

    echo "✓ Moved to $OLLAMA_MODELS_DIR/models"
    echo "✓ Created symlink for backward compatibility"
else
    echo "⚠️  Ollama models not found at ~/.ollama/models"
fi

echo ""
echo "================================================================"
echo "Step 4: Create environment configuration"
echo "================================================================"

# Create env file for Citation Assistant
cat > "$HOME/projects/citation_assistant/.env.models" << 'EOF'
# Model locations for Citation Assistant
# Source this file in your shell or add to ~/.bashrc

# HuggingFace transformers cache (PubMedBERT)
export TRANSFORMERS_CACHE="$HOME/models/huggingface"
export HF_HOME="$HOME/models/huggingface"

# Ollama models
export OLLAMA_MODELS="$HOME/models/ollama/models"

# Sentence transformers cache (uses HuggingFace)
export SENTENCE_TRANSFORMERS_HOME="$HOME/models/huggingface"
EOF

echo "✓ Created .env.models configuration file"
echo ""
echo "Add this to your ~/.bashrc to make it permanent:"
echo "  source ~/projects/citation_assistant/.env.models"
echo ""

echo "================================================================"
echo "Step 5: Update Citation Assistant configuration"
echo "================================================================"

# Create a config file for the project
cat > "$HOME/projects/citation_assistant/model_config.py" << 'EOF'
"""
Model location configuration for Citation Assistant
This file sets environment variables before importing models
"""
import os
from pathlib import Path

# Set model directories
MODELS_BASE = Path.home() / "models"
HF_MODELS_DIR = MODELS_BASE / "huggingface"
OLLAMA_MODELS_DIR = MODELS_BASE / "ollama" / "models"

# Set environment variables
os.environ["TRANSFORMERS_CACHE"] = str(HF_MODELS_DIR)
os.environ["HF_HOME"] = str(HF_MODELS_DIR)
os.environ["SENTENCE_TRANSFORMERS_HOME"] = str(HF_MODELS_DIR)
os.environ["OLLAMA_MODELS"] = str(OLLAMA_MODELS_DIR)

# Create directories if they don't exist
HF_MODELS_DIR.mkdir(parents=True, exist_ok=True)
OLLAMA_MODELS_DIR.mkdir(parents=True, exist_ok=True)

print(f"✓ Model directories configured:")
print(f"  HuggingFace: {HF_MODELS_DIR}")
print(f"  Ollama: {OLLAMA_MODELS_DIR}")
EOF

echo "✓ Created model_config.py"

echo ""
echo "================================================================"
echo "COMPLETE!"
echo "================================================================"
echo ""
echo "Model locations:"
echo "  $MODELS_DIR/"
echo "  ├── huggingface/          (~420 MB - PubMedBERT)"
echo "  └── ollama/models/        (~16 GB - Gemma2)"
echo ""
echo "Symlinks created for backward compatibility:"
echo "  ~/.cache/huggingface/hub → $HF_MODELS_DIR/hub"
echo "  ~/.ollama/models → $OLLAMA_MODELS_DIR/models"
echo ""
echo "Next steps:"
echo "  1. Add to ~/.bashrc:"
echo "     source ~/projects/citation_assistant/.env.models"
echo ""
echo "  2. Your Citation Assistant will automatically use the new locations"
echo "     via model_config.py imports"
echo ""
echo "  3. Test with: python3 cite.py search \"test query\""
echo ""
