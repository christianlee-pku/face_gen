# Pipeline Step 4: GCP Deployment

## Overview
This final step automates the deployment of the face generation service to Google Cloud Platform (GCP) Vertex AI. It handles model uploading, container building, and endpoint creation with auto-scaling.

## Prerequisites
-   **GCP Account**: Active project with billing enabled.
-   **Tools**: `gcloud` CLI, `docker`, `gsutil` installed and authenticated.
-   **Permissions**: Access to GCS, Artifact Registry, and Vertex AI.

## Execution
Run the deployment script (edit variables inside first):
```bash
bash scripts/step4_deploy_gcp.sh
```

**Dry Run Mode**:
To verify the logic without actual cloud operations:
```bash
DRY_RUN=1 bash scripts/step4_deploy_gcp.sh
```

## Workflow
1.  **Upload Model**: Copies `stylegan3.onnx` to a GCS bucket.
2.  **Build Container**: Builds the Docker image using `docker/Dockerfile.inference`.
3.  **Push Image**: Pushes the image to GCP Artifact Registry.
4.  **Deploy Vertex AI**:
    -   Creates a **Model** resource in Vertex AI pointing to the GCS artifact and Docker image.
    -   Creates an **Endpoint** resource.
    -   Deploys the Model to the Endpoint with auto-scaling configuration.

## Key Components
-   `src/tools/deploy_vertex.py`: Python script using `google-cloud-aiplatform` SDK to manage Vertex AI resources.
-   `docker/Dockerfile.inference`: Defines the runtime environment.
