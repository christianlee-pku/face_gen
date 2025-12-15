#!/bin/bash
set -e

# ==============================================================================
# Face Generation Pipeline - Step 2: Export to ONNX
# ==============================================================================
# This script exports the trained PyTorch model to ONNX format.
# ==============================================================================

# Configuration
CONFIG_PATH="configs/stylegan3_celeba.py"
CHECKPOINT_PATH="work_dirs/stylegan3_celeba/epoch_3.pth"
OUTPUT_PATH="work_dirs/stylegan3.onnx"

echo "========================================================"
echo "Starting Export Pipeline"
echo "Checkpoint: ${CHECKPOINT_PATH}"
echo "Output: ${OUTPUT_PATH}"
echo "========================================================"

# Check if checkpoint exists, if not create dummy
if [ ! -f "${CHECKPOINT_PATH}" ]; then
    echo "Checkpoint not found. Creating dummy checkpoint for testing..."
    export PYTHONPATH=$(pwd)
    python src/tools/create_dummy_checkpoint.py
fi

# Run Export
echo "Running export script..."
export PYTHONPATH=$(pwd)
python src/tools/export.py ${CONFIG_PATH} ${CHECKPOINT_PATH} --out ${OUTPUT_PATH}

echo "Export completed."
