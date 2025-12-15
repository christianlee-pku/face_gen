import argparse
from google.cloud import aiplatform

def deploy_vertex_endpoint(
    project_id,
    region,
    model_display_name,
    serving_container_image_uri,
    artifact_uri,
    machine_type='n1-standard-4',
    min_replica_count=1,
    max_replica_count=4,
    endpoint_display_name='face-gen-endpoint'
):
    """
    Deploys the Face Generation model to GCP Vertex AI.
    
    Args:
        project_id (str): GCP Project ID.
        region (str): GCP Region (e.g., us-central1).
        model_display_name (str): Display name for the Model in Vertex AI.
        serving_container_image_uri (str): URI of the Docker image in Artifact Registry/Container Registry.
        artifact_uri (str): GCS URI where the model artifacts (stylegan3.onnx) are stored.
        machine_type (str): Compute Engine machine type.
        min_replica_count (int): Minimum number of replicas.
        max_replica_count (int): Maximum number of replicas.
        endpoint_display_name (str): Display name for the Endpoint.
    """
    
    print(f"Initializing Vertex AI SDK for project {project_id} in {region}...")
    aiplatform.init(project=project_id, location=region)
    
    # 1. Upload Model to Vertex AI Model Registry
    print(f"Uploading Model: {model_display_name}")
    model = aiplatform.Model.upload(
        display_name=model_display_name,
        artifact_uri=artifact_uri,
        serving_container_image_uri=serving_container_image_uri,
        serving_container_predict_route="/generate", # Assuming we will have this
        serving_container_health_route="/health",
        serving_container_ports=[8080]
    )
    print(f"Model uploaded: {model.resource_name}")
    
    # 2. Create Endpoint
    # Check if endpoint exists or create new one
    endpoints = aiplatform.Endpoint.list(filter=f'display_name="{endpoint_display_name}"')
    if endpoints:
        endpoint = endpoints[0]
        print(f"Using existing Endpoint: {endpoint.resource_name}")
    else:
        print(f"Creating Endpoint: {endpoint_display_name}")
        endpoint = aiplatform.Endpoint.create(
            display_name=endpoint_display_name,
            project=project_id,
            location=region
        )
        print(f"Endpoint created: {endpoint.resource_name}")
    
    # 3. Deploy Model to Endpoint with Auto-scaling
    print("Deploying Model to Endpoint...")
    model.deploy(
        endpoint=endpoint,
        deployed_model_display_name=model_display_name,
        machine_type=machine_type,
        min_replica_count=min_replica_count,
        max_replica_count=max_replica_count,
        traffic_split={"0": 100},
        sync=True
    )
    
    print(f"Model deployed to Endpoint: {endpoint.resource_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deploy Face Gen Model to GCP Vertex AI")
    parser.add_argument("--project-id", required=True, help="GCP Project ID")
    parser.add_argument("--region", required=True, help="GCP Region")
    parser.add_argument("--model-name", default="face-gen-model", help="Model Display Name")
    parser.add_argument("--image-uri", required=True, help="Serving Container Image URI")
    parser.add_argument("--artifact-uri", required=True, help="GCS URI for model artifacts")
    
    args = parser.parse_args()
    
    deploy_vertex_endpoint(
        project_id=args.project_id,
        region=args.region,
        model_display_name=args.model_name,
        serving_container_image_uri=args.image_uri,
        artifact_uri=args.artifact_uri
    )
