"""
Telemetry-Aware Context Builder.

Loads live telemetry from master_labelled.parquet, detects anomalies against
configured thresholds, and formats structured context for RAG prompt injection.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import structlog

import config

log = structlog.get_logger(__name__)


# ── Threshold bounds for anomaly detection ────────────────────────────
ANOMALY_THRESHOLDS = {
    "inverter_temperature": {"high": 75.0, "critical": 85.0, "unit": "°C"},
    "conversion_efficiency": {"low": 0.85, "critical_low": 0.70, "unit": "ratio"},
    "string_mismatch_cv": {"high": 0.15, "critical": 0.25, "unit": "CV"},
    "power_vs_24h_baseline": {"low": 0.80, "critical_low": 0.60, "unit": "ratio"},
    "meter_freq": {"low": 49.5, "high": 50.5, "unit": "Hz"},
}


def build_telemetry_context(
    inverter_id: Optional[str] = None,
    plant_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Load latest telemetry from PlantStateManager and build structured context for RAG prompts."""
    from api.state import state_manager
    state = state_manager.get_state()
    inverters = state.get("inverters", {})

    if not inverters:
        return {"inverters": [], "summary": "No matching telemetry found in PlantStateManager."}

    selected_inverters = []
    for inv_id, inv_data in inverters.items():
        if inverter_id and inv_id != inverter_id:
            continue
        if plant_id and inv_data.get("plant_id") != plant_id:
            continue
        
        selected_inverters.append({
            "inverter_id": inv_id,
            "plant_id": inv_data.get("plant_id", "UNKNOWN"),
            "temperature": inv_data.get("temperature"),
            "power": inv_data.get("power"),
            "efficiency": inv_data.get("efficiency"),
            "risk_score": inv_data.get("risk_score"),
            "anomaly_score": inv_data.get("anomaly_score"),
            "feature_drivers": inv_data.get("top_features", []),
            "label": inv_data.get("label", 0)
        })

    summary_parts = [f"Total inverters: {len(selected_inverters)}"]
    temps = [i["temperature"] for i in selected_inverters if i["temperature"] is not None]
    if temps:
        summary_parts.append(f"Avg temp: {sum(temps)/len(temps):.1f}°C, Max temp: {max(temps):.1f}°C")

    powers = [i["power"] for i in selected_inverters if i["power"] is not None]
    if powers:
        summary_parts.append(f"Total power: {sum(powers):.1f}W")

    return {
        "inverters": selected_inverters,
        "summary": " | ".join(summary_parts),
    }


def detect_anomalies(
    telemetry: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Flag sensor readings outside normal bounds."""
    anomalies = []

    for signal, bounds in ANOMALY_THRESHOLDS.items():
        value = telemetry.get(signal)
        if value is None or pd.isna(value):
            continue

        value = float(value)

        # High threshold checks
        if "critical" in bounds and value > bounds["critical"]:
            anomalies.append({
                "signal": signal,
                "value": value,
                "severity": "CRITICAL",
                "threshold": f"> {bounds['critical']}{bounds['unit']}",
                "description": f"{signal} at {value:.2f} exceeds critical threshold of {bounds['critical']}{bounds['unit']}",
            })
        elif "high" in bounds and value > bounds["high"]:
            anomalies.append({
                "signal": signal,
                "value": value,
                "severity": "WARNING",
                "threshold": f"> {bounds['high']}{bounds['unit']}",
                "description": f"{signal} at {value:.2f} exceeds warning threshold of {bounds['high']}{bounds['unit']}",
            })

        # Low threshold checks
        if "critical_low" in bounds and value < bounds["critical_low"]:
            anomalies.append({
                "signal": signal,
                "value": value,
                "severity": "CRITICAL",
                "threshold": f"< {bounds['critical_low']}{bounds['unit']}",
                "description": f"{signal} at {value:.2f} below critical threshold of {bounds['critical_low']}{bounds['unit']}",
            })
        elif "low" in bounds and value < bounds["low"]:
            anomalies.append({
                "signal": signal,
                "value": value,
                "severity": "WARNING",
                "threshold": f"< {bounds['low']}{bounds['unit']}",
                "description": f"{signal} at {value:.2f} below warning threshold of {bounds['low']}{bounds['unit']}",
            })

    # String mismatch check using CV of string currents
    string_currents = []
    for i in range(1, 25):
        val = telemetry.get(f"smu_string{i}")
        if val is not None and not pd.isna(val) and float(val) > 0:
            string_currents.append(float(val))

    if len(string_currents) >= 3:
        import numpy as np
        cv = float(np.std(string_currents) / np.mean(string_currents))
        if cv > 0.25:
            anomalies.append({
                "signal": "string_current_imbalance",
                "value": cv,
                "severity": "CRITICAL",
                "threshold": "CV > 0.25",
                "description": f"String current CV of {cv:.3f} indicates severe imbalance",
            })
        elif cv > 0.15:
            anomalies.append({
                "signal": "string_current_imbalance",
                "value": cv,
                "severity": "WARNING",
                "threshold": "CV > 0.15",
                "description": f"String current CV of {cv:.3f} indicates moderate imbalance",
            })

    return anomalies


def format_telemetry_for_prompt(
    inverter_id: Optional[str] = None,
    plant_id: Optional[str] = None,
) -> Tuple[str, List[Dict[str, Any]]]:
    """Build a complete telemetry context string for prompt injection."""
    context = build_telemetry_context(inverter_id, plant_id)

    if not context["inverters"]:
        return context["summary"], []

    all_anomalies = []
    parts = [f"## Live Telemetry Context\n{context['summary']}\n"]

    for inv in context["inverters"]:
        inv_id = inv.get("inverter_id", "UNKNOWN")
        
        # Map back to anomaly detection keys
        mapped_inv = {
            "inverter_temperature": inv.get("temperature"),
            "pv1_power": inv.get("power"),
            "conversion_efficiency": inv.get("efficiency")
        }
        anomalies = detect_anomalies(mapped_inv)
        all_anomalies.extend(anomalies)

        parts.append(f"### Inverter {inv_id}")
        
        import json
        telemetry_json = {
            "inverter_id": inv_id,
            "temperature": inv.get("temperature"),
            "efficiency": inv.get("efficiency"),
            "power": inv.get("power"),
            "risk_score": inv.get("risk_score"),
            "anomaly_score": inv.get("anomaly_score"),
            "feature_drivers": inv.get("feature_drivers")
        }
        parts.append("```json\n" + json.dumps(telemetry_json, indent=2) + "\n```")

        # Anomalies for this inverter
        if anomalies:
            parts.append(f"⚠️ ANOMALIES DETECTED ({len(anomalies)}):")
            for a in anomalies:
                parts.append(f"  - [{a['severity']}] {a['description']}")
                
        parts.append("\n")

    return "\n".join(parts), all_anomalies
