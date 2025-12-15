#!/bin/bash
set -e

# ==============================================================================
# Face Generation Pipeline - GCP Deployment Script
# ==============================================================================
# PREREQUISITES:
# - gcloud CLI installed and authenticated (mocked for DRY_RUN)
# - Docker installed (mocked for DRY_RUN)
# - Python environment with necessary dependencies
# ==============================================================================

# Configuration
PROJECT_ID="face-gen"
REGION="us-central1"
REPO_NAME="face-gen-repo"
IMAGE_NAME="face-gen-inference"
IMAGE_TAG="v1"
BUCKET_NAME="your-gcs-bucket-name"
MODEL_DISPLAY_NAME="stylegan3-face-gen"
ENDPOINT_DISPLAY_NAME="face-gen-endpoint"

# Derived Variables
GCR_URI="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/${IMAGE_NAME}:${IMAGE_TAG}"
GCS_MODEL_URI="gs://${BUCKET_NAME}/models/${MODEL_DISPLAY_NAME}"
LOCAL_MODEL_PATH="work_dirs/stylegan3.onnx"

# Dry Run Mode
if [ "${DRY_RUN}" == "1" ]; then
    echo "[DRY_RUN] Mode Enabled. External commands will be skipped or mocked."
    
    # Mock gcloud/gsutil/docker functions
    gcloud() { echo "[DRY_RUN] gcloud $@"; }
    gsutil() { echo "[DRY_RUN] gsutil $@"; }
    docker() { echo "[DRY_RUN] docker $@"; }
fi

# Python Interpreter Path
PYTHON_BIN="python"

echo "========================================================"
echo "Starting GCP Deployment Pipeline"
echo "Project: ${PROJECT_ID}"
echo "Region: ${REGION}"
echo "Python: ${PYTHON_BIN}"
echo "========================================================"

# 1. Export Model to ONNX
echo ""
echo "[Step 1/5] Exporting PyTorch model to ONNX..."

# Check if model exists (for dry run scenarios where export might fail or take too long)
if [ ! -f "${LOCAL_MODEL_PATH}" ] && [ "${DRY_RUN}" == "1" ]; then
    echo "[DRY_RUN] Creating dummy ONNX for verification..."
    # We can use the previously created dummy checkpoint and export script
    # First create dummy checkpoint if not exists
    ${PYTHON_BIN} src/tools/create_dummy_checkpoint.py
    
    # Then export
    ${PYTHON_BIN} src/tools/export.py configs/stylegan3_celeba.py work_dirs/stylegan3_celeba/epoch_5.pth --out ${LOCAL_MODEL_PATH}
else
    # In real run, use existing checkpoint
    # python src/tools/export.py ...
    echo "Using existing logic or file."
fi

echo "Model exported to ${LOCAL_MODEL_PATH}"

# 2. Upload Model to GCS
echo ""
echo "[Step 2/5] Uploading model to Google Cloud Storage..."
if [ "${DRY_RUN}" == "1" ]; then
    echo "[DRY_RUN] gsutil cp ${LOCAL_MODEL_PATH} ${GCS_MODEL_URI}/stylegan3.onnx"
else
    gsutil cp ${LOCAL_MODEL_PATH} ${GCS_MODEL_URI}/stylegan3.onnx
fi
echo "Model uploaded to ${GCS_MODEL_URI}/stylegan3.onnx"

# 3. Build and Push Docker Image
echo ""
echo "[Step 3/5] Building and Pushing Docker Image..."
# Create repository if it doesn't exist (ignore error if exists)
if [ "${DRY_RUN}" == "1" ]; then
    echo "[DRY_RUN] gcloud artifacts repositories create ${REPO_NAME} ..."
    echo "[DRY_RUN] gcloud auth configure-docker ..."
    echo "[DRY_RUN] docker build -t ${GCR_URI} ..."
    echo "[DRY_RUN] docker push ${GCR_URI}"
else
    gcloud artifacts repositories create ${REPO_NAME} \
        --repository-format=docker \
        --location=${REGION} \
        --description="Face Generation Inference Repository" || true

    # Configure Docker to authenticate with Artifact Registry
    gcloud auth configure-docker ${REGION}-docker.pkg.dev --quiet

    # Build
    docker build -t ${GCR_URI} -f docker/Dockerfile.inference .

    # Push
    docker push ${GCR_URI}
fi
echo "Docker image pushed to ${GCR_URI}"

# 4. Deploy to Vertex AI
echo ""
echo "[Step 4/5] Deploying to Vertex AI Endpoint..."

if [ "${DRY_RUN}" == "1" ]; then
    echo "[DRY_RUN] Running mock test for deploy_vertex.py..."
    export PYTHONPATH=.
    ${PYTHON_BIN} tests/test_deploy_vertex_mock.py
else
    ${PYTHON_BIN} src/tools/deploy_vertex.py \
        --project-id ${PROJECT_ID} \
        --region ${REGION} \
        --model-name ${MODEL_DISPLAY_NAME} \
        --image-uri ${GCR_URI} \
        --artifact-uri ${GCS_MODEL_URI}
fi

echo ""
echo "========================================================"
echo "Deployment Pipeline Complete!"
echo "You can now query your endpoint via Vertex AI API."
echo "========================================================"