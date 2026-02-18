"""
Test suite for serving module.
"""

import pytest
from fastapi.testclient import TestClient
from serving.app import app


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


def test_health_check(client):
    """Test health endpoint."""
    response = client.get("/health")
    
    assert response.status_code == 200
    assert response.json()["status"] in ["ready", "loading"]


def test_health_models_loaded(client):
    """Test health endpoint returns loaded models."""
    response = client.get("/health")
    
    assert "models_loaded" in response.json()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
