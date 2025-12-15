#!/bin/bash
set -e

# ==============================================================================
# Face Generation Pipeline - Step 5: Validation (Evaluation)
# ==============================================================================
# This script runs the model evaluation (e.g. generating images and computing metrics).
# ==============================================================================

# Configuration
CONFIG_PATH="configs/stylegan3_celeba.py"
CHECKPOINT_PATH="work_dirs/stylegan3_celeba/epoch_3.pth"

echo "========================================================"
echo "Starting Validation Pipeline"
echo "Config: ${CONFIG_PATH}"
echo "Checkpoint: ${CHECKPOINT_PATH}"
echo "========================================================"

# Check if checkpoint exists, if not create dummy
if [ ! -f "${CHECKPOINT_PATH}" ]; then
    echo "Checkpoint not found. Creating dummy checkpoint for testing..."
    export PYTHONPATH=$(pwd)
    python src/tools/create_dummy_checkpoint.py
fi

# Run Evaluation
echo "Running evaluation script..."
export PYTHONPATH=$(pwd)
python src/tools/evaluate.py ${CONFIG_PATH} ${CHECKPOINT_PATH}

echo "Validation completed."
