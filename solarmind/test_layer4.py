import json
from genai.guardrails.validator import parse_llm_response

llm_json = """```json
{
  "inverter_id": "INV_001",
  "plant_id": "PLANT_1",
  "risk_score": 0.88,
  "risk_level": "CRITICAL",
  "summary": "Inverter is likely to fail due to high thermal gradient.",
  "root_cause": "Thermal gradient exceeds normal operation bounds.",
  "action": "Dispatch technician to inspect cooling system.",
  "confidence": "LOW",
  "data_quality": "COMPLETE",
  "delta_shap_available": true,
  "causal_drivers": [
    {"feature": "thermal_gradient", "delta_shap": 0.45, "direction": "UP"}
  ]
}
```"""

print('Parsing mock response...')
report = parse_llm_response(llm_json, {})
print('Success! Validated object:')
print('Risk Level after validation:', report.risk_level.value)
print('Summary after validation:', report.summary)
