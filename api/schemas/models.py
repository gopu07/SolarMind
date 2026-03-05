"""
Layer 7 — API Schemas.

Defines all Pydantic models used by the FastAPI endpoints.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field

from genai.guardrails.validator import InverterReport


# =====================================================================
# Auth Schemas
# =====================================================================
class Token(BaseModel):
    access_token: str
    token_type: str


# =====================================================================
# Health Schemas
# =====================================================================
class HealthResponse(BaseModel):
    status: str = Field(description="'ok', 'degraded', or 'down'")
    checks: Dict[str, str] = Field(description="Status of individual components")


# =====================================================================
# Predict Schemas
# =====================================================================
class PredictRequest(BaseModel):
    inverter_id: str
    lookback_rows: int = Field(default=672, ge=96, le=2000)
    generate_narrative: bool = True
    include_delta_shap: bool = True


class BatchPredictRequest(BaseModel):
    plant_id: str
    generate_narrative: bool = False


class ShapFeature(BaseModel):
    feature: str
    shap_value: float


class DeltaShapFeature(BaseModel):
    feature: str
    delta_shap: float


class PredictResponse(BaseModel):
    inverter_id: str
    plant_id: str
    risk_score: float
    risk_level: str
    shap_top5: List[ShapFeature]
    delta_shap_top5: Optional[List[DeltaShapFeature]] = None
    report: Optional[InverterReport] = None
    latency_ms: float
    model_version: str = "1.0"
    timestamp: int


# =====================================================================
# Query (RAG) Schemas
# =====================================================================
class QueryRequest(BaseModel):
    question: str
    plant_id: Optional[str] = None
    top_k: int = Field(default=5, ge=1, le=20)


class Citation(BaseModel):
    inverter_id: str
    timestamp: int
    risk_level: str

    model_config = ConfigDict(extra="ignore")


class QueryResponse(BaseModel):
    answer: str
    citations: List[Citation]
    latency_ms: float
