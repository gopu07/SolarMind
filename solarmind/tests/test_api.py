"""
Tests for FastAPI Backend Layer 7.
"""

import pytest
from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_health_returns_200_or_207():
    """Health endpoint should return 200 or 207 when components are reachable."""
    response = client.get("/health")
    assert response.status_code in (200, 207, 503), f"Unexpected status {response.status_code}"
    data = response.json()
    assert "status" in data
    assert "checks" in data


def test_predict_returns_404_for_unknown_inverter():
    """POST /predict should return 404 for an unknown inverter_id."""
    # First get a valid token
    token_resp = client.post("/auth/token", data={"username": "admin", "password": "admin"})
    assert token_resp.status_code == 200
    token = token_resp.json()["access_token"]

    resp = client.post(
        "/predict",
        json={"inverter_id": "FAKE_INV_999", "generate_narrative": False},
        headers={"Authorization": f"Bearer {token}"}
    )
    assert resp.status_code == 404, f"Expected 404 for unknown inverter, got {resp.status_code}"


def test_unauthenticated_request_returns_401():
    """Unauthenticated requests to protected endpoints should return 401."""
    resp = client.post("/predict", json={"inverter_id": "INV_001"})
    assert resp.status_code == 401


def test_auth_token_endpoint():
    """POST /auth/token should return a JWT for valid credentials."""
    resp = client.post("/auth/token", data={"username": "admin", "password": "admin"})
    assert resp.status_code == 200
    data = resp.json()
    assert "access_token" in data
    assert data["token_type"] == "bearer"


def test_auth_token_invalid_credentials():
    """POST /auth/token should return 401 for bad credentials."""
    resp = client.post("/auth/token", data={"username": "baduser", "password": "badpass"})
    assert resp.status_code == 401


def test_predict_returns_risk_score_in_range():
    """Test 1: POST /predict returns a risk score in the [0.0, 1.0] range."""
    from unittest.mock import patch
    
    token_resp = client.post("/auth/token", data={"username": "admin", "password": "admin"})
    token = token_resp.json()["access_token"]
    
    with patch("models.predict.predict_inverter") as mock_predict:
        mock_predict.return_value = {
            "inverter_id": "INV_TEST",
            "plant_id": "P1",
            "risk_score": 0.75,
            "risk_level": "HIGH",
            "shap_top5": [],
            "error": "NONE"
        }
        
        resp = client.post(
            "/predict",
            json={"inverter_id": "INV_TEST", "generate_narrative": False},
            headers={"Authorization": f"Bearer {token}"}
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "risk_score" in data
        assert 0.0 <= data["risk_score"] <= 1.0
        assert data["risk_score"] == 0.75


def test_predict_missing_fields_returns_422():
    """Test 2: POST /predict missing fields returns HTTP 422."""
    token_resp = client.post("/auth/token", data={"username": "admin", "password": "admin"})
    token = token_resp.json()["access_token"]
    
    # Missing 'inverter_id' which is required
    resp = client.post(
        "/predict",
        json={"generate_narrative": False},
        headers={"Authorization": f"Bearer {token}"}
    )
    assert resp.status_code == 422
