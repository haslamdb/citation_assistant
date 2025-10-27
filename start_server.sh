#!/bin/bash

# Citation Assistant Server Startup Script

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Activate conda environment
source /home/david/miniforge3/etc/profile.d/conda.sh
conda activate rag

# Start server
echo "Starting Citation Assistant Server..."
echo "Access at: http://$(hostname -I | awk '{print $1}'):8000"
echo "Press Ctrl+C to stop"
echo ""

python server.py
