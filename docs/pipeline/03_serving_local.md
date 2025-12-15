# Pipeline Step 3: Local API Serving

## Overview
Before deploying to the cloud, it is crucial to verify the inference service locally. This step starts a FastAPI server that loads the ONNX model and exposes a REST API for image generation.

## Prerequisites
-   **ONNX Model**: `work_dirs/stylegan3.onnx` from Step 2.
-   **Environment Variables**: `MODEL_PATH`, `API_AUTH_TOKEN` (optional).

## Execution
Run the local serving script:
```bash
bash scripts/step3_serve_local.sh
```
Or manually:
```bash
export MODEL_PATH=work_dirs/stylegan3.onnx
export AIP_HTTP_PORT=8080
python src/apis/vertex_entry.py
```

## Key Components
-   `src/apis/vertex_entry.py`: Entry point compatible with Vertex AI container requirements.
-   `src/apis/app.py`: FastAPI application definition.
-   `src/apis/inference.py`: Wraps ONNX Runtime for efficient inference.
-   `src/apis/postprocessing.py`: Applies invisible watermarking.

## Endpoints
-   `GET /health`: Health check.
-   `POST /generate`: Generate image (JSON body: `{"seed": 42}`).
