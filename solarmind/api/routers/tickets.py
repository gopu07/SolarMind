"""
Layer 7 - Tickets API Router.

Generates draft maintenance tickets for critical alerts.
"""
from typing import List
from fastapi import APIRouter
import time

from api.schemas.models import Ticket
from api.state import state_manager

router = APIRouter(prefix="/tickets", tags=["Tickets"])

@router.get("", response_model=List[Ticket])
async def get_ticket_drafts():
    """Get auto-generated ticket drafts for critical alerts."""
    plant_state = state_manager.get_state()
    inverters = plant_state.get("inverters", {})
    
    tickets = []
    
    for inv_id, inv_data in inverters.items():
        risk_score = float(inv_data.get("final_risk_score", inv_data.get("risk_score", 0.0)))
        top_features = inv_data.get("top_features", [])
        plant_id = inv_data.get("plant_id", "PLANT_1")
        timestamp = plant_state.get("timestamp", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        
        if risk_score > 0.8:
            suspected_issue = "Unknown anomaly"
            recommended_action = "Inspect immediately."
            
            if top_features:
                top_feature = top_features[0]
                feature_name = top_feature.get("feature", "unknown").lower()
                if "temperature" in feature_name:
                    suspected_issue = "Cooling fan degradation or overheating"
                    recommended_action = "Inspect cooling system within 24 hours"
                elif "efficiency" in feature_name:
                    suspected_issue = "Panel soiling or DC mismatch"
                    recommended_action = "Schedule panel cleaning / inspect DC lines"
                elif "voltage" in feature_name:
                    suspected_issue = "String voltage mismatch"
                    recommended_action = "Check fuse and diode continuity"
                else:
                    suspected_issue = f"High risk anomaly in {feature_name}"
                    recommended_action = "Perform full diagnostic check"
                    
            tickets.append(
                Ticket(
                    id=f"TKT-{inv_id[-4:]}-{int(time.time()*1000)%100000}",
                    inverter_id=inv_id,
                    plant_id=plant_id,
                    risk_score=risk_score,
                    suspected_issue=suspected_issue,
                    recommended_action=recommended_action,
                    status="open",
                    created_at=timestamp
                )
            )
            
    return tickets
