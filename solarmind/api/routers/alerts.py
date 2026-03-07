"""
Layer 7 - Alerts API Router.

Exposes active alerts based on inverter risk scores.
"""
from typing import List
from fastapi import APIRouter
import time

from api.schemas.models import Alert
from api.state import state_manager

router = APIRouter(prefix="/alerts", tags=["Alerts"])

@router.get("", response_model=List[Alert])
async def get_active_alerts():
    """Get currently active alerts based on plant state."""
    plant_state = state_manager.get_state()
    inverters = plant_state.get("inverters", {})
    
    alerts = []
    
    for inv_id, inv_data in inverters.items():
        risk_score = float(inv_data.get("final_risk_score", inv_data.get("risk_score", 0.0)))
        plant_id = inv_data.get("plant_id", "PLANT_1")
        timestamp = plant_state.get("timestamp", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        
        if risk_score > 0.8:
            alerts.append(
                Alert(
                    id=f"ALT-{inv_id}-{timestamp}",
                    inverter_id=inv_id,
                    plant_id=plant_id,
                    risk_score=risk_score,
                    level="critical",
                    message=f"Critical risk detected for {inv_id}. Immediate attention required.",
                    timestamp=timestamp
                )
            )
        elif risk_score > 0.6:
            alerts.append(
                Alert(
                    id=f"ALT-{inv_id}-{timestamp}",
                    inverter_id=inv_id,
                    plant_id=plant_id,
                    risk_score=risk_score,
                    level="warning",
                    message=f"Warning: Elevated risk score ({risk_score:.2f}) for {inv_id}.",
                    timestamp=timestamp
                )
            )
            
    return alerts
