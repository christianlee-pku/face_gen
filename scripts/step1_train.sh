#!/bin/bash
set -e

# ==============================================================================
# Face Generation Pipeline - Step 1: Train
# ==============================================================================
# This script runs the training process locally.
# It uses the CelebA configuration by default.
# ==============================================================================

# Configuration
CONFIG_PATH="configs/stylegan3_celeba.py"

echo "========================================================"
echo "Starting Training Pipeline"
echo "Config: ${CONFIG_PATH}"
echo "========================================================"

# Run Training
echo "Running training script..."
export PYTHONPATH=$(pwd)
python src/tools/train.py ${CONFIG_PATH}

echo "Training completed."
