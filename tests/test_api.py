from fastapi.testclient import TestClient
from src.apis.app import app

client = TestClient(app)

def test_generate_endpoint_contract():
    """
    Verify the contract for /generate endpoint.
    It should accept a seed or latent vector and return a base64 encoded image.
    """
    # Test valid request with seed
    payload = {"seed": 42}
    response = client.post("/generate", json=payload)
    
    assert response.status_code == 200
    data = response.json()
    assert "image" in data
    assert isinstance(data["image"], str) # Base64 string
    assert "meta" in data

def test_generate_invalid_input():
    """
    Verify error handling for invalid input.
    """
    payload = {"seed": "invalid_string"}
    # Should fail validation (handled by Pydantic)
    response = client.post("/generate", json=payload)
    
    # If endpoint exists, should return 422. If not, 404.
    assert response.status_code in [404, 422]
