#!/bin/bash
set -e

# ==============================================================================
# Face Generation Pipeline - Step 6: Inference (Generate Images)
# ==============================================================================
# This script generates images using the trained model (PyTorch version).
# For ONNX/API inference, see step3_serve_local.sh.
# ==============================================================================

# Configuration
CONFIG_PATH="configs/stylegan3_celeba.py"
CHECKPOINT_PATH="work_dirs/stylegan3_celeba/epoch_5.pth"
SEEDS="42,100,2023"
TRUNCATION_PSI=0.7
OUTDIR="work_dirs/inference_output"

echo "========================================================"
echo "Starting Inference Pipeline"
echo "Config: ${CONFIG_PATH}"
echo "Checkpoint: ${CHECKPOINT_PATH}"
echo "Seeds: ${SEEDS}"
echo "========================================================"

# Check if checkpoint exists, if not create dummy
if [ ! -f "${CHECKPOINT_PATH}" ]; then
    echo "Checkpoint not found. Creating dummy checkpoint for testing..."
    export PYTHONPATH=$(pwd)
    python src/tools/create_dummy_checkpoint.py
fi

# Run Inference
# Note: Since we don't have a dedicated `generate.py` script in the initial plan,
# we reuse `evaluate.py` or similar, or assume `run_inference.py` in src/deploy (moved there previously)
# Let's check where the inference logic resides.
# Based on file structure, `src/deploy/run_inference.py` exists.

echo "Running inference script..."
export PYTHONPATH=$(pwd)
python src/deploy/run_inference.py \
    --config ${CONFIG_PATH} \
    --checkpoint ${CHECKPOINT_PATH} \
    --seeds ${SEEDS} \
    --truncation ${TRUNCATION_PSI} \
    --outdir ${OUTDIR}

echo "Inference completed. Results saved to ${OUTDIR}"
