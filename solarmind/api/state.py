"""
Centralized state manager for the SolarMind API.

This module provides the single source of truth for real-time telemetry,
replacing direct reads of Parquet files for API endpoints and WebSockets.
"""

from typing import Dict, Any, Optional
from datetime import datetime

class PlantStateManager:
    def __init__(self):
        self.plant_state: Dict[str, Any] = {
            "timestamp": None,
            "inverter_count": 0,
            "inverters": {}
        }
        self.last_updated: float = 0.0

    def update_state(self, updates: Dict[str, Any]) -> None:
        """
        Updates the global plant state with a new snapshot of data.
        Typically called by the replay engine or a live data ingestor.
        """
        # Ensure we always keep track of the timestamp
        if "timestamp" in updates:
            self.plant_state["timestamp"] = updates["timestamp"]
            
        if "inverters" in updates:
            # We overwrite the inverter data to ensure no stale state persists
            for inv_id, inv_data in updates["inverters"].items():
                self.plant_state["inverters"][inv_id] = inv_data
                
            self.plant_state["inverter_count"] = len(self.plant_state["inverters"])
            
        self.last_updated = datetime.now().timestamp()

    def get_state(self) -> Dict[str, Any]:
        """
        Returns the entire plant state.
        """
        return self.plant_state

    def get_inverter_state(self, inverter_id: str) -> Optional[Dict[str, Any]]:
        """
        Returns the state for a single inverter.
        """
        return self.plant_state["inverters"].get(inverter_id)

    def update_inverter_state(self, inverter_id: str, updates: Dict[str, Any]) -> None:
        """
        Updates specific fields for a single inverter.
        """
        if inverter_id not in self.plant_state["inverters"]:
            self.plant_state["inverters"][inverter_id] = {}
        
        self.plant_state["inverters"][inverter_id].update(updates)
        self.plant_state["inverter_count"] = len(self.plant_state["inverters"])
        self.last_updated = datetime.now().timestamp()

# Global singleton instance
state_manager = PlantStateManager()

