#!/bin/bash

# Citation Assistant Secure Server Startup Script

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Activate conda environment
source /home/david/miniforge3/etc/profile.d/conda.sh
conda activate rag

echo "=" * 60
echo "🔒 Starting Secure Citation Assistant Server"
echo "=" * 60
echo ""
echo "Server IP: 192.168.1.163:8000"
echo "Access: http://192.168.1.163:8000"
echo "API Docs: http://192.168.1.163:8000/docs"
echo ""
echo "⚠️  SECURITY ENABLED - Authentication Required"
echo ""
echo "Create users with: python manage_users.py"
echo "Press Ctrl+C to stop"
echo ""

# Start secure server
python server_secure.py
