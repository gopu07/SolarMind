"""
Tests for Phase 2: Core Platform Stabilization & ML Pipeline Enhancement.

Verifies:
  - PlantStateManager correctly stores and updates telemetry per inverter.
  - Predict pipeline loads IsolationForest and generates anomaly_score.
  - Final risk score successfully blends XGBoost and Anomaly scores.
  - Replay engine correctly structures plant-level payloads.
"""

import json
import pytest
from datetime import datetime

import pandas as pd
import numpy as np

# We import the manager to test state mutability
from api.state import PlantStateManager, state_manager
from models.predict import _error_result

# =====================================================================
# PlantStateManager Tests
# =====================================================================
class TestPlantStateManager:
    def test_initial_state(self):
        manager = PlantStateManager()
        state = manager.get_state()
        
        assert state["timestamp"] is None
        assert state["inverter_count"] == 0
        assert isinstance(state["inverters"], dict)
        assert len(state["inverters"]) == 0

    def test_update_state(self):
        manager = PlantStateManager()
        
        # Simulating first frame
        updates = {
            "timestamp": "2026-03-05T12:00:00Z",
            "inverters": {
                "INV_001": {"temperature": 45.0, "power": 1200.0, "risk_score": 0.1},
                "INV_002": {"temperature": 46.0, "power": 1180.0, "risk_score": 0.15}
            }
        }
        
        manager.update_state(updates)
        state = manager.get_state()
        
        assert state["timestamp"] == "2026-03-05T12:00:00Z"
        assert state["inverter_count"] == 2
        assert "INV_001" in state["inverters"]
        
    def test_partial_update_preserves_others(self):
        manager = PlantStateManager()
        
        # Setup initial state
        manager.update_state({
            "timestamp": "2026-03-05T12:00:00Z",
            "inverters": {
                "INV_001": {"temperature": 45.0, "power": 1200.0, "risk_score": 0.1},
                "INV_002": {"temperature": 46.0, "power": 1180.0, "risk_score": 0.15}
            }
        })
        
        # New frame only updates INV_001
        manager.update_state({
            "timestamp": "2026-03-05T12:15:00Z",
            "inverters": {
                "INV_001": {"temperature": 50.0, "power": 1300.0, "risk_score": 0.2, "anomaly_score": 0.05}
            }
        })
        
        state = manager.get_state()
        assert state["inverter_count"] == 2
        # Verify INV_001 got updated
        assert state["inverters"]["INV_001"]["temperature"] == 50.0
        assert state["inverters"]["INV_001"]["anomaly_score"] == 0.05
        # Verify INV_002 remained conceptually intact but existing 
        assert state["inverters"]["INV_002"]["temperature"] == 46.0

    def test_single_inverter_update(self):
        manager = PlantStateManager()
        manager.update_inverter_state("INV_999", {"temperature": 80.0, "risk_score": 0.9})
        
        inv = manager.get_inverter_state("INV_999")
        assert inv is not None
        assert inv["temperature"] == 80.0
        assert inv["risk_score"] == 0.9

from models.predict import predict_inverter

# =====================================================================
# Model Prediction Tests
# =====================================================================
class TestPredictionSchemas:
    def test_error_result_schema(self):
        from models.predict import _error_result
        res = _error_result("INV_ERR", "Testing error schema")
        assert res["inverter_id"] == "INV_ERR"
        assert res["risk_score"] == 0.0
        assert res["risk_level"] == "LOW"
        assert res["error"] == "Testing error schema"

    def test_predict_inverter_metrics(self):
        # This will test the live loading of the model and running inference
        # We assume master_labelled.parquet exists (which it must for training to have worked)
        import pandas as pd
        import config
        master_path = config.PROCESSED_DIR / "master_labelled.parquet"
        if not master_path.exists():
            pytest.skip("master_labelled.parquet missing")
            
        # Get a real inverter ID
        df = pd.read_parquet(master_path, columns=["inverter_id"]).head(1)
        inverter_id = df["inverter_id"].iloc[0]
        
        result = predict_inverter(inverter_id)
        
        assert "risk_score" in result
        assert "anomaly_score" in result
        assert "final_risk_score" in result
        assert isinstance(result["anomaly_score"], float)
        assert isinstance(result["final_risk_score"], float)
        # Verify the blend logic basically
        # final_risk_score = 0.7 * risk_score + 0.3 * anomaly_score
        expected = round(0.7 * result["risk_score"] + 0.3 * result["anomaly_score"], 4)
        assert abs(result["final_risk_score"] - expected) < 0.001
