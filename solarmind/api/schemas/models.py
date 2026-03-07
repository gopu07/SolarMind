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
    model_loaded: bool = False
    isolation_forest_loaded: bool = False
    inverter_count: int = 0
    websocket_clients: int = 0
    api_uptime_seconds: float = 0.0

class ModelMetricsResponse(BaseModel):
    macro_f1: float
    multiclass_roc_auc: float
    confusion_matrix: Optional[List[List[int]]] = None
    model_version: str


# =====================================================================
# State Schemas
# =====================================================================
class InverterState(BaseModel):
    inverter_id: str
    plant_id: str
    risk_score: float
    anomaly_score: float
    final_risk_score: float
    temperature: float
    power: float
    efficiency: float
    label: int
    top_features: List[Dict[str, float]] = Field(default_factory=list)
    predicted_failure_hours: Optional[int] = None

class PlantState(BaseModel):
    timestamp: str
    inverter_count: int
    inverters: Dict[str, InverterState]


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


class LimeFeature(BaseModel):
    feature: str
    lime_weight: float


class PredictionResult(BaseModel):
    inverter_id: str
    plant_id: str
    predicted_failure_type: str = "unknown"
    risk_score: float
    final_risk_score: Optional[float] = None
    anomaly_score: Optional[float] = None
    risk_level: str
    shap_top5: List[ShapFeature]
    lime_top5: Optional[List[LimeFeature]] = None
    delta_shap_top5: Optional[List[DeltaShapFeature]] = None
    report: Optional[InverterReport] = None
    latency_ms: float
    model_version: str = "1.0"
    timestamp: int
    predicted_failure_hours: Optional[int] = None


class NarrativeRequest(BaseModel):
    inverter_id: str
    risk_score: float
    plant_id: str = "PLANT_1"


# =====================================================================
# Query (RAG) Schemas
# =====================================================================
class QueryRequest(BaseModel):
    question: str
    plant_id: Optional[str] = None
    top_k: int = Field(default=5, ge=1, le=20)
    enable_multi_query: bool = True
    enable_reranking: bool = True


class Citation(BaseModel):
    inverter_id: str
    timestamp: int
    risk_level: str
    relevance_score: float = Field(default=0.0, ge=0.0, le=1.0)
    retrieval_method: str = Field(default="hybrid")

    model_config = ConfigDict(extra="ignore")


class SensorEvidence(BaseModel):
    signal: str
    value: str
    expected: str = ""
    assessment: str = ""


class RecommendedAction(BaseModel):
    action: str
    priority: str = Field(default="7d")  # immediate | 24h | 48h | 7d
    justification: str = ""


class SimilarPastEvent(BaseModel):
    event_id: str = ""
    similarity: str = ""


class ReasoningChain(BaseModel):
    step1_telemetry_analysis: str = ""
    step2_abnormal_signals: str = ""
    step3_fault_matching: str = ""
    step4_diagnosis: str = ""
    step5_actions: str = ""


class DiagnosticReport(BaseModel):
    diagnosis: str = Field(max_length=300)
    risk_level: str
    root_cause_hypothesis: str = Field(max_length=600)
    sensor_evidence: List[SensorEvidence] = Field(default_factory=list)
    recommended_actions: List[RecommendedAction] = Field(default_factory=list)
    similar_past_events: List[SimilarPastEvent] = Field(default_factory=list)
    reasoning_chain: Optional[ReasoningChain] = None
    confidence: str = "MEDIUM"
    data_quality: str = "PARTIAL"


class QueryResponse(BaseModel):
    answer: str
    citations: List[Citation]
    diagnostic_report: Optional[DiagnosticReport] = None
    retrieval_stats: Optional[Dict[str, Any]] = None
    latency_ms: float

# =====================================================================
# Alert & Ticket Schemas (Phase 2)
# =====================================================================
class Alert(BaseModel):
    id: str
    inverter_id: str
    plant_id: str
    risk_score: float
    level: str = Field(description="'warning' or 'critical'")
    message: str
    timestamp: str

class Ticket(BaseModel):
    id: str
    inverter_id: str
    plant_id: str
    risk_score: float
    suspected_issue: str
    recommended_action: str
    status: str = Field(default="open")
    created_at: str

# =====================================================================
# Memory / Session Schemas (Phase 2)
# =====================================================================
class SessionMemory(BaseModel):
    session_id: str
    last_inverter: Optional[str] = None
    last_intent: Optional[str] = None
    history: List[Dict[str, str]] = Field(default_factory=list)

# =====================================================================
# Phase 2.5 Timeline & Maintenance Schemas
# =====================================================================
class TimelineEvent(BaseModel):
    inverter_id: str
    predicted_failure_time: str
    predicted_failure_hours: int
    risk_score: float
    failure_type: str = "degradation"

class MaintenanceTask(BaseModel):
    maintenance_id: str
    inverter_id: str
    recommended_time: str
    priority: str = Field(description="'CRITICAL', 'HIGH', or 'MEDIUM'")
    recommended_action: str
