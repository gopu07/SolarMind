from fastapi import APIRouter
from typing import List
from datetime import datetime, timedelta, timezone

from api.schemas.models import TimelineEvent
from api.state import state_manager

router = APIRouter(tags=["Maintenance & Timeline"])

@router.get("/timeline", response_model=List[TimelineEvent])
async def get_timeline_events():
    """Returns predictive timeline events based on the live plant state TTF heuristic."""
    state = state_manager.get_state()
    inverters = state.get("inverters", {})
    
    # Try parsing the current state timestamp, fallback to now
    try:
        base_time = datetime.fromisoformat(state.get("timestamp", "").replace("Z", "+00:00"))
    except:
        base_time = datetime.now(timezone.utc)
    
    events = []
    for inv_id, inv_data in inverters.items():
        ttf_hours = inv_data.get("predicted_failure_hours")
        
        if ttf_hours is not None:
            failure_time = base_time + timedelta(hours=ttf_hours)
            
            # Determine likely failure type based on top features
            failure_type = "degradation"
            top_feats = inv_data.get("top_features", [])
            if top_feats:
                feat_name = str(top_feats[0].get("feature", "")).lower()
                if "temperature" in feat_name or "thermal" in feat_name:
                    failure_type = "thermal_issue"
                elif "efficiency" in feat_name:
                    failure_type = "conversion_loss"
                elif "mismatch" in feat_name or "voltage" in feat_name:
                    failure_type = "electrical_anomaly"
            
            events.append(
                TimelineEvent(
                    inverter_id=inv_id,
                    predicted_failure_time=failure_time.isoformat(),
                    predicted_failure_hours=ttf_hours,
                    risk_score=float(inv_data.get("risk_score", 0.0)),
                    failure_type=failure_type
                )
            )
            
    # Sort events so the most immediate failures appear first
    events.sort(key=lambda x: x.predicted_failure_hours)
    return events
