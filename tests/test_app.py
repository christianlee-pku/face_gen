from fastapi.testclient import TestClient
from src.apis.app import app

client = TestClient(app)

def test_health_check():
    """
    Test the /health endpoint.
    """
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy", "version": "1.0.0"}

def test_root():
    """
    Test the root endpoint.
    """
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()
