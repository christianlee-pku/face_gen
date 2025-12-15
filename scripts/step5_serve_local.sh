#!/bin/bash
set -e

# ==============================================================================
# Face Generation Pipeline - Step 3: Run API Server (Local)
# ==============================================================================
# This script starts the FastAPI server locally using the exported ONNX model.
# ==============================================================================

# Configuration
MODEL_PATH="work_dirs/stylegan3.onnx"
PORT=8080
TOKEN="dev-token"

echo "========================================================"
echo "Starting Local API Server"
echo "Model: ${MODEL_PATH}"
echo "Port: ${PORT}"
echo "========================================================"

# Check if model exists
if [ ! -f "${MODEL_PATH}" ]; then
    echo "Error: Model file not found at ${MODEL_PATH}. Please run scripts/step2_export.sh first."
    exit 1
fi

# Set Environment Variables
export MODEL_PATH=${MODEL_PATH}
export API_AUTH_TOKEN=${TOKEN}
export AIP_HTTP_PORT=${PORT}
export PYTHONPATH=$(pwd)

# Run Server
echo "Starting uvicorn server..."
python src/apis/vertex_entry.py
