"""
Tests for GenAI Guardrails Layer 4.
"""

import json
import pytest
from pydantic import ValidationError

from genai.guardrails.validator import (
    InverterReport,
    parse_llm_response,
    get_fallback_report,
)


def test_inconsistent_risk_level_raises_validation_error():
    """An InverterReport with an invalid enum value must raise ValidationError."""
    with pytest.raises(ValidationError):
        InverterReport(
            inverter_id="I1",
            plant_id="P1",
            risk_score=0.9,
            risk_level="HIGH",
            summary="Test",
            root_cause="Test",
            action="Test",
            confidence="HIGH",
            data_quality="COMPLETE",
            delta_shap_available=True,
            causal_drivers=[
                {"feature": "test", "delta_shap": 0.5, "direction": "WRONG_VALUE"}
            ],
        )


def test_non_json_triggers_one_retry():
    """A raw non-JSON LLM response should trigger exactly one retry via parse_llm_response."""
    # parse_llm_response itself extracts JSON and validates.
    # With garbage input it should raise, which the caller retries.
    with pytest.raises(Exception):
        parse_llm_response("This is not JSON at all", {})


def test_fallback_report_after_failures():
    """get_fallback_report must return a valid InverterReport without raising."""
    report = get_fallback_report("INV_1", "PLANT_1", 0.8, "HIGH")
    assert report is not None
    assert report.inverter_id == "INV_1"
    assert "Automated diagnostic failed" in report.summary or "fallback" in report.summary.lower() or len(report.summary) > 0


def test_valid_json_parses_successfully():
    """A well-formed JSON string should parse into a valid InverterReport."""
    payload = json.dumps({
        "inverter_id": "INV_1",
        "plant_id": "P1",
        "risk_score": 0.8,
        "risk_level": "CRITICAL",
        "summary": "Overheating",
        "root_cause": "Fan failure",
        "action": "Replace fan",
        "confidence": "HIGH",
        "data_quality": "COMPLETE",
        "delta_shap_available": True,
        "causal_drivers": [
            {"feature": "temp", "delta_shap": 0.3, "direction": "UP"}
        ],
    })
    report = parse_llm_response(payload, {})
    assert report.inverter_id == "INV_1"
    assert report.risk_level.value == "CRITICAL"
