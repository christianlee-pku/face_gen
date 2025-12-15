#!/bin/bash
set -e

# ==============================================================================
# Face Generation Pipeline - Step 4: Deploy to GCP
# ==============================================================================
# This script builds the Docker image and deploys it to GCP Vertex AI.
# It assumes you have gcloud and docker configured.
# ==============================================================================

# Configuration (Customize these)
PROJECT_ID="your-gcp-project-id"
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

# Dry Run Check
if [ "${DRY_RUN}" == "1" ]; then
    echo "[DRY_RUN] Mode Enabled. Commands will be mocked."
    gsutil() { echo "[DRY_RUN] gsutil $@"; }
    gcloud() { echo "[DRY_RUN] gcloud $@"; }
    docker() { echo "[DRY_RUN] docker $@"; }
fi

echo "========================================================"
echo "Starting GCP Deployment"
echo "Project: ${PROJECT_ID}"
echo "========================================================"

# 1. Upload Model
echo "[1/3] Uploading model to GCS..."
if [ "${DRY_RUN}" == "1" ]; then
    echo "[DRY_RUN] gsutil cp ${LOCAL_MODEL_PATH} ${GCS_MODEL_URI}/stylegan3.onnx"
else
    gsutil cp ${LOCAL_MODEL_PATH} ${GCS_MODEL_URI}/stylegan3.onnx
fi

# 2. Build & Push Image
echo "[2/3] Building and Pushing Docker Image..."
if [ "${DRY_RUN}" == "1" ]; then
    echo "[DRY_RUN] gcloud artifacts repositories create..."
    echo "[DRY_RUN] docker build -t ${GCR_URI} ..."
    echo "[DRY_RUN] docker push ${GCR_URI}"
else
    gcloud artifacts repositories create ${REPO_NAME} \
        --repository-format=docker \
        --location=${REGION} \
        --description="Face Generation Inference Repository" || true
    gcloud auth configure-docker ${REGION}-docker.pkg.dev --quiet
    docker build -t ${GCR_URI} -f docker/Dockerfile.inference .
    docker push ${GCR_URI}
fi

# 3. Deploy Vertex AI
echo "[3/3] Deploying to Vertex AI..."
export PYTHONPATH=$(pwd)
if [ "${DRY_RUN}" == "1" ]; then
    echo "[DRY_RUN] Mocking Vertex AI deployment..."
    python tests/test_deploy_vertex_mock.py
else
    python src/tools/deploy_vertex.py \
        --project-id ${PROJECT_ID} \
        --region ${REGION} \
        --model-name ${MODEL_DISPLAY_NAME} \
        --image-uri ${GCR_URI} \
        --artifact-uri ${GCS_MODEL_URI}
fi

echo "Deployment pipeline finished."
