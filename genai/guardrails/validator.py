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
    inverter_id: str, plant_id: str, risk_score: float, risk_level: str
) -> InverterReport:
    """Generate a safe, rule-based fallback report.

    Used when the LLM fails to return valid JSON after retries.

    Args:
        inverter_id: Inverter ID.
        plant_id: Plant ID.
        risk_score: Base predicted risk score.
        risk_level: Base predicted risk level.

    Returns:
        Valid InverterReport fallback object.
    """
    import config

    # Ensure risk level is strictly matched to score for the fallback
    valid_risk_level = config.risk_level_from_score(risk_score)

    fallback_text = "LLM unavailable — rule-based summary"
    return InverterReport(
        inverter_id=inverter_id,
        plant_id=plant_id,
        risk_score=risk_score,
        risk_level=valid_risk_level,
        summary=fallback_text,
        root_cause=fallback_text,
        action=fallback_text,
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
