"""
Layer 4 — GenAI Narrative Generator Guardrails.

Provides Pydantic models for validation, custom validation rules for
consistency, and fallback mechanisms for LLM failure recovery.
"""

from __future__ import annotations

import json
import logging
from enum import Enum
from typing import Any, Dict, List, Optional, Type

from pydantic import BaseModel, Field, field_validator, model_validator

# Setup basic logging for the validator
log = logging.getLogger(__name__)


# =====================================================================
# Enums
# =====================================================================
class RiskLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class ConfidenceLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class DataQuality(str, Enum):
    COMPLETE = "COMPLETE"
    PARTIAL = "PARTIAL"
    MISSING = "MISSING"


class Direction(str, Enum):
    UP = "UP"
    DOWN = "DOWN"


# =====================================================================
# Models
# =====================================================================
class CausalDriver(BaseModel):
    feature: str
    delta_shap: float = Field(ge=-1.0, le=1.0)
    direction: Direction


class InverterReport(BaseModel):
    inverter_id: str
    plant_id: str
    risk_score: float = Field(ge=0.0, le=1.0)
    risk_level: RiskLevel
    summary: str = Field(max_length=400)
    root_cause: str = Field(max_length=600)
    action: str = Field(max_length=400)
    confidence: ConfidenceLevel
    data_quality: DataQuality
    delta_shap_available: bool
    causal_drivers: Optional[List[CausalDriver]] = Field(default=None, max_length=5)
    data_gap_note: Optional[str] = None

    # =================================================================
    # Cross-field specific validation rules
    # =================================================================
    @model_validator(mode="after")
    def check_consistency(self) -> "InverterReport":
        """Apply cross-field business logic validation."""
        import config

        # Rule 1: risk_level must be consistent with risk_score thresholds
        expected_level = config.risk_level_from_score(self.risk_score)
        if self.risk_level != expected_level:
            raise ValueError(
                f"Inconsistent risk level: Score {self.risk_score} -> expected "
                f"{expected_level}, but got {self.risk_level}"
            )

        # Rule 2: If CRITICAL but confidence is LOW, override to HIGH and append warning
        if self.risk_level == RiskLevel.CRITICAL and self.confidence == ConfidenceLevel.LOW:
            self.risk_level = RiskLevel.HIGH
            warning_note = " [WARNING: Risk downgraded to HIGH due to LOW confidence]"
            if not self.summary.endswith(warning_note):
                self.summary = (self.summary + warning_note)[:400]

        # Rule 3: If delta_shap_available is False, ensure causal_drivers is omitted
        # and confidence is at most MEDIUM
        if not self.delta_shap_available:
            self.causal_drivers = None
            if self.confidence == ConfidenceLevel.HIGH:
                self.confidence = ConfidenceLevel.MEDIUM

        return self


# =====================================================================
# Retry and Fallback Logic
# =====================================================================
def get_fallback_report(
    inverter_id: str, plant_id: str, risk_score: float, risk_level: str,
    inv_state: Optional[Dict[str, Any]] = None
) -> InverterReport:
    """Generate a safe, rule-based fallback report.

    Used when the LLM fails to return valid JSON after retries.

    Args:
        inverter_id: Inverter ID.
        plant_id: Plant ID.
        risk_score: Base predicted risk score.
        risk_level: Base predicted risk level.
        inv_state: Dict with telemetry variables (optional).

    Returns:
        Valid InverterReport fallback object.
    """
    import config

    # Ensure risk level is strictly matched to score for the fallback
    valid_risk_level = config.risk_level_from_score(risk_score)

    if inv_state:
        temp = float(inv_state.get("temperature", 0.0))
        eff = float(inv_state.get("efficiency", 0.0))
        power = float(inv_state.get("power", 0.0))
        risk_pct = int(risk_score * 100)
        
        if valid_risk_level == "CRITICAL":
            summary = f"Inverter {inverter_id} shows critical risk ({risk_pct}%). Temperature is {temp:.1f}°C and efficiency is dropping to {eff*100:.1f}%. Predicted failure soon."
            root_cause = "Severe cooling degradation or hardware fault detected."
            action = "Dispatch technician immediately for cooling system inspection."
        elif valid_risk_level == "HIGH":
            summary = f"Inverter {inverter_id} is operating at high risk ({risk_pct}%). Elevated temperature ({temp:.1f}°C) and degraded efficiency ({eff*100:.1f}%) observed."
            root_cause = "Potential thermal stress or incipient component failure."
            action = "Schedule maintenance review and clean filters."
        else:
            summary = f"Inverter {inverter_id} is healthy. Operating efficiently at {eff*100:.1f}% with stable temperature ({temp:.1f}°C), delivering {power/1000:.1f}kW power."
            root_cause = "Normal operation parameters observed."
            action = "Continue routine monitoring."
    else:
        summary = "LLM unavailable — rule-based summary"
        root_cause = summary
        action = summary

    return InverterReport(
        inverter_id=inverter_id,
        plant_id=plant_id,
        risk_score=risk_score,
        risk_level=valid_risk_level,
        summary=summary,
        root_cause=root_cause,
        action=action,
        confidence=ConfidenceLevel.LOW,
        data_quality=DataQuality.PARTIAL,
        delta_shap_available=False,
        causal_drivers=None,
        data_gap_note=None,
    )


def parse_llm_response(
    response_text: str, context: Dict[str, Any]
) -> InverterReport:
    """Attempt to parse LLM string into an InverterReport.

    Raises:
        ValueError / pydantic.ValidationError if invalid.
    """
    # Try extra stripping for markdown blocks if LLM disobeyed
    cleaned = response_text.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    elif cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    cleaned = cleaned.strip()

    data = json.loads(cleaned)
    return InverterReport(**data)
