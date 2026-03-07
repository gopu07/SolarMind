"""
Centralized state manager for the SolarMind API.

This module provides the single source of truth for real-time telemetry,
replacing direct reads of Parquet files for API endpoints and WebSockets.
"""

from typing import Dict, Any, Optional, List
from datetime import datetime

class PlantStateManager:
    def __init__(self):
        self.plant_state: Dict[str, Any] = {
            "timestamp": None,
            "inverter_count": 0,
            "inverters": {},
            "history": {}  # Stores last 192 samples (48 hours @ 15min) per inverter
        }
        self.last_updated: float = 0.0

    def update_state(self, updates: Dict[str, Any]) -> None:
        """
        Updates the global plant state with a new snapshot of data.
        """
        if "timestamp" in updates:
            self.plant_state["timestamp"] = updates["timestamp"]
            
        if "inverters" in updates:
            for inv_id, inv_data in updates["inverters"].items():
                self.plant_state["inverters"][inv_id] = inv_data
                
                # Update history buffer
                if inv_id not in self.plant_state["history"]:
                    self.plant_state["history"][inv_id] = []
                
                history_entry = {
                    "timestamp": updates.get("timestamp"),
                    "temperature": inv_data.get("temperature"),
                    "power": inv_data.get("power"),
                    "efficiency": inv_data.get("efficiency"),
                    "risk_score": inv_data.get("risk_score"),
                    "anomaly_score": inv_data.get("anomaly_score")
                }
                self.plant_state["history"][inv_id].append(history_entry)
                
                # Keep last 192 samples (48 hours)
                if len(self.plant_state["history"][inv_id]) > 192:
                    self.plant_state["history"][inv_id] = self.plant_state["history"][inv_id][-192:]
                
            self.plant_state["inverter_count"] = len(self.plant_state["inverters"])
            
        self.last_updated = datetime.now().timestamp()

    def get_state(self) -> Dict[str, Any]:
        return self.plant_state

    def get_inverter_history(self, inverter_id: str) -> List[Dict[str, Any]]:
        """Returns buffered historical telemetry for a specific inverter."""
        return self.plant_state["history"].get(inverter_id, [])

    def get_inverter_state(self, inverter_id: str) -> Optional[Dict[str, Any]]:
        return self.plant_state["inverters"].get(inverter_id)

    def update_inverter_state(self, inverter_id: str, updates: Dict[str, Any]) -> None:
        if inverter_id not in self.plant_state["inverters"]:
            self.plant_state["inverters"][inverter_id] = {}
        
        self.plant_state["inverters"][inverter_id].update(updates)
        self.plant_state["inverter_count"] = len(self.plant_state["inverters"])
        self.last_updated = datetime.now().timestamp()

# Global singleton instance
state_manager = PlantStateManager()

