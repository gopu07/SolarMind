import pytest
from app_config.settings import mask_secret, Settings
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_mask_secret():
    assert mask_secret(None) == "NOT SET"
    assert mask_secret("") == "NOT SET"
    assert mask_secret("short") == "****"
    assert mask_secret("sk-1234567890abcdef") == "sk-1****cdef"

def test_config_status_endpoint():
    response = client.get("/config/status")
    assert response.status_code == 200
    data = response.json()
    assert "environment" in data
    assert "openai_api_key" in data
    assert "gemini_api_key" in data
    
    # Verify masking
    if data["openai_api_key"] != "NOT SET":
        assert "****" in data["openai_api_key"]
    if data["gemini_api_key"] != "NOT SET":
        assert "****" in data["gemini_api_key"]

def test_settings_load():
    settings = Settings()
    # Basic check that defaults or env values are present
    assert settings.model_path is not None
    assert settings.environment in ["development", "staging", "production", "test"]
