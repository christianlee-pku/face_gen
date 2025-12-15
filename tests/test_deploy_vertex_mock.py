import unittest
from unittest.mock import patch, MagicMock
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from src.tools.deploy_vertex import deploy_vertex_endpoint

class TestDeployVertex(unittest.TestCase):
    
    @patch('src.tools.deploy_vertex.aiplatform')
    def test_deploy_vertex_endpoint(self, mock_aiplatform):
        # Setup mocks
        mock_model = MagicMock()
        mock_model.resource_name = "projects/123/locations/us-central1/models/456"
        mock_aiplatform.Model.upload.return_value = mock_model
        
        mock_endpoint = MagicMock()
        mock_endpoint.resource_name = "projects/123/locations/us-central1/endpoints/789"
        mock_aiplatform.Endpoint.create.return_value = mock_endpoint
        mock_aiplatform.Endpoint.list.return_value = [] # No existing endpoints
        
        # Execute
        deploy_vertex_endpoint(
            project_id="test-project",
            region="us-central1",
            model_display_name="test-model",
            serving_container_image_uri="gcr.io/test-project/test-image:latest",
            artifact_uri="gs://test-bucket/model",
            endpoint_display_name="test-endpoint"
        )
        
        # Verify interactions
        mock_aiplatform.init.assert_called_once_with(project="test-project", location="us-central1")
        
        mock_aiplatform.Model.upload.assert_called_once_with(
            display_name="test-model",
            artifact_uri="gs://test-bucket/model",
            serving_container_image_uri="gcr.io/test-project/test-image:latest",
            serving_container_predict_route="/generate",
            serving_container_health_route="/health",
            serving_container_ports=[8080]
        )
        
        mock_aiplatform.Endpoint.create.assert_called_once_with(
            display_name="test-endpoint",
            project="test-project",
            location="us-central1"
        )
        
        mock_model.deploy.assert_called_once()
        print("\nTestDeployVertex passed: Logic verified with mocks.")

if __name__ == "__main__":
    unittest.main()
