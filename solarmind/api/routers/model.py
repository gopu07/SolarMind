from fastapi import APIRouter, HTTPException
from typing import Dict, Any
from models.drift import drift_monitor
from api.state import state_manager
import pandas as pd

router = APIRouter(prefix="/model", tags=["Model"])

@router.get("/drift")
async def get_model_drift():
    """Calculate and return model drift based on the last 100 inverter states."""
    plant_state = state_manager.get_state()
    inverters = plant_state.get("inverters", {})
    
    if not inverters:
        return {"status": "error", "message": "No active telemetry in state manager"}

    # Convert mapping to list of dicts for calculation
    current_telemetry = []
    for inv_id, data in inverters.items():
        # Map state keys to feature names expected by drift monitor
        current_telemetry.append({
            "inverter_temperature": data.get("temperature"),
            "pv1_power": data.get("power"),
            "conversion_efficiency": data.get("efficiency")
        })
    
    return drift_monitor.calculate_drift(current_telemetry)
