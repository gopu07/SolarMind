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
    """Load latest telemetry and build structured context for RAG prompts.

    Args:
        inverter_id: Optional specific inverter to focus on.
        plant_id: Optional plant filter.

    Returns:
        Dict with ``inverters`` list, each containing latest sensor readings.
    """
    master_path = config.PROCESSED_DIR / "master_labelled.parquet"
    if not master_path.exists():
        log.warning("master_labelled_missing", path=str(master_path))
        return {"inverters": [], "summary": "No telemetry data available."}

    try:
        df = pd.read_parquet(master_path)
    except Exception as e:
        log.error("telemetry_load_failed", error=str(e))
        return {"inverters": [], "summary": f"Failed to load telemetry: {e}"}

    # Apply filters
    if plant_id and "plant_id" in df.columns:
        df = df[df["plant_id"] == plant_id].copy()
    if inverter_id and "inverter_id" in df.columns:
        df = df[df["inverter_id"] == inverter_id].copy()

    if df.empty:
        return {"inverters": [], "summary": "No matching telemetry found."}

    # Get latest reading per inverter
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        latest = df.sort_values("timestamp").groupby("inverter_id").last().reset_index()
    else:
        latest = df.groupby("inverter_id").last().reset_index()

    # Extract key signals
    signal_cols = [
        "inverter_id", "plant_id",
        "pv1_current", "pv1_voltage", "pv1_power",
        "pv2_current", "pv2_voltage", "pv2_power",
        "inverter_temperature", "meter_active_power",
        "meter_pf", "meter_freq",
        "meter_v_r", "meter_v_y", "meter_v_b",
        "inverter_alarm_code", "inverter_op_state", "inverter_limit_percent",
    ]
    # Add string monitoring columns
    signal_cols += [f"smu_string{i}" for i in range(1, 25)]

    avail_cols = [c for c in signal_cols if c in latest.columns]
    inverters = latest[avail_cols].to_dict(orient="records")

    # Compute fleet summary
    summary_parts = []
    summary_parts.append(f"Total inverters: {len(inverters)}")

    if "inverter_temperature" in latest.columns:
        avg_temp = latest["inverter_temperature"].mean()
        max_temp = latest["inverter_temperature"].max()
        summary_parts.append(f"Avg temp: {avg_temp:.1f}°C, Max temp: {max_temp:.1f}°C")

    if "meter_active_power" in latest.columns:
        total_power = latest["meter_active_power"].sum()
        summary_parts.append(f"Total power: {total_power:.1f}W")

    return {
        "inverters": inverters,
        "summary": " | ".join(summary_parts),
    }


def detect_anomalies(
    telemetry: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Flag sensor readings outside normal bounds.

    Args:
        telemetry: Single inverter telemetry dict.

    Returns:
        List of anomaly dicts with ``signal``, ``value``, ``severity``, ``threshold``.
    """
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
    """Build a complete telemetry context string for prompt injection.

    Args:
        inverter_id: Optional inverter filter.
        plant_id: Optional plant filter.

    Returns:
        Tuple of (formatted_string, all_anomalies).
    """
    context = build_telemetry_context(inverter_id, plant_id)

    if not context["inverters"]:
        return context["summary"], []

    all_anomalies = []
    parts = [f"## Live Telemetry Context\n{context['summary']}\n"]

    for inv in context["inverters"]:
        inv_id = inv.get("inverter_id", "UNKNOWN")
        anomalies = detect_anomalies(inv)
        all_anomalies.extend(anomalies)

        parts.append(f"\n### Inverter {inv_id}")

        # Key signals
        key_signals = [
            ("Temperature", inv.get("inverter_temperature"), "°C"),
            ("PV1 Power", inv.get("pv1_power"), "W"),
            ("PV2 Power", inv.get("pv2_power"), "W"),
            ("Grid Power", inv.get("meter_active_power"), "W"),
            ("Efficiency", inv.get("conversion_efficiency"), ""),
            ("Op State", inv.get("inverter_op_state"), ""),
            ("Alarm Code", inv.get("inverter_alarm_code"), ""),
            ("Grid Freq", inv.get("meter_freq"), "Hz"),
            ("Limit %", inv.get("inverter_limit_percent"), "%"),
        ]
        for name, val, unit in key_signals:
            if val is not None and not pd.isna(val):
                parts.append(f"- {name}: {val}{unit}")

        # String currents (compact)
        string_vals = []
        for i in range(1, 25):
            sv = inv.get(f"smu_string{i}")
            if sv is not None and not pd.isna(sv):
                string_vals.append(f"S{i}={float(sv):.2f}A")
        if string_vals:
            parts.append(f"- String Currents: {', '.join(string_vals)}")

        # Anomalies for this inverter
        if anomalies:
            parts.append(f"\n⚠️ ANOMALIES DETECTED ({len(anomalies)}):")
            for a in anomalies:
                parts.append(f"  - [{a['severity']}] {a['description']}")

    return "\n".join(parts), all_anomalies
