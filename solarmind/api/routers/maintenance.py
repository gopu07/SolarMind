from fastapi import APIRouter
from typing import List
import uuid
from datetime import datetime, timedelta, timezone

from api.schemas.models import MaintenanceTask
from api.state import state_manager

router = APIRouter(tags=["Maintenance & Timeline"])

@router.get("/maintenance_schedule", response_model=List[MaintenanceTask])
async def get_maintenance_schedule():
    """Generates scheduled maintenance tasks prioritizing time-to-failure."""
    state = state_manager.get_state()
    inverters = state.get("inverters", {})
    
    try:
        base_time = datetime.fromisoformat(state.get("timestamp", "").replace("Z", "+00:00"))
    except:
        base_time = datetime.now(timezone.utc)
        
    tasks = []
    for inv_id, inv_data in inverters.items():
        ttf_hours = inv_data.get("predicted_failure_hours")
        if ttf_hours is None:
            continue
            
        priority = "MEDIUM"
        action = "Schedule routine inspection within 72 hours."
        rec_time = base_time + timedelta(hours=72)
        
        if ttf_hours <= 12:
            priority = "CRITICAL"
            action = "Immediate physical inspection and thermal imaging required."
            rec_time = base_time + timedelta(hours=1)
        elif ttf_hours <= 24:
            priority = "HIGH"
            action = "Dispatch technician for targeted inspection within 24 hours."
            rec_time = base_time + timedelta(hours=24)
        elif ttf_hours <= 48:
            priority = "MEDIUM"
            action = "Schedule preventive maintenance check within 72 hours."
            rec_time = base_time + timedelta(hours=72)
            
        # Refine action based on top SHAP feature
        top_feats = inv_data.get("top_features", [])
        if top_feats:
            f = str(top_feats[0].get("feature", "")).lower()
            if "temperature" in f:
                action += " Focus on cooling units and thermal hotspots."
            elif "efficiency" in f:
                action += " Inspect panel string connections."

        tasks.append(
            MaintenanceTask(
                maintenance_id=f"MNT-{inv_id.split('_')[-1]}-{uuid.uuid4().hex[:6].upper()}",
                inverter_id=inv_id,
                recommended_time=rec_time.isoformat(),
                priority=priority,
                recommended_action=action
            )
        )
        
    # Sort tasks by severity/time: Critical first, High second, Medium third
    priority_map = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2}
    tasks.sort(key=lambda x: (priority_map.get(x.priority, 99), x.recommended_time))
    
    return tasks
