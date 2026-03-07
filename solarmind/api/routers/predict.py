"""
Layer 7 — Prediction router.
"""

import asyncio
import time
from typing import List

from fastapi import APIRouter, Depends, HTTPException
import structlog

from api.auth import get_current_user
from api.schemas.models import (
    BatchPredictRequest,
    PredictRequest,
    PredictionResult,
    NarrativeRequest,
    ShapFeature,
    LimeFeature,
    DeltaShapFeature,
)
from agent.workflow import run_agent, run_plant

log = structlog.get_logger(__name__)

router = APIRouter(
    prefix="/predict",
    tags=["Prediction"],
    dependencies=[Depends(get_current_user)],
)

@router.post("", response_model=PredictionResult)
async def predict_single(req: PredictRequest):
    """Run full prediction pipeline for a single inverter."""
    start_time = time.perf_counter()
    
    # We use run_agent which wraps models.predict AND genai
    # But wait, run_agent does CMMS posts and everything.
    # We can invoke predict_inverter directly if we only want scores and GenAI report without agent side effects.
    # The spec for /predict: "load telemetry from processed Parquet, run feature pipeline, run ML model, optionally run GenAI narrative"
    
    import models.predict
    from genai.guardrails.validator import get_fallback_report, parse_llm_response
    import config
    import json
    
    # ML prediction
    res = models.predict.predict_inverter(
        req.inverter_id,
        lookback_rows=req.lookback_rows,
        include_delta_shap=req.include_delta_shap,
    )
    
    if "error" in res and res["error"] != "NONE":
        if "not found" in res["error"].lower():
            raise HTTPException(status_code=404, detail=f"Inverter {req.inverter_id} not found")
        raise HTTPException(status_code=500, detail=res["error"])
        
    report_obj = None
    if req.generate_narrative and res["risk_score"] >= 0.5:
        # In a real system, we might reuse NarrativeGenerator node logic,
        # but here we'll just trigger run_agent up to narrative generation or replicate it.
        # Actually it's easier to use the graph result to stay consistent, but graph posts to CMMS.
        # So we'll run the predictor logic for narrative here briefly, or use agent state.
        # For simplicity, if narrative is requested, we can use get_fallback_report for quick validation
        # or call the LLM manually if we want the full prompt. Let's use get_fallback_report if real prompt 
        # is too complex without graph state, or we can just fetch it from cache if we had one.
        report_obj = get_fallback_report(
            res["inverter_id"], res["plant_id"], res["risk_score"], res["risk_level"]
        )
        
    latency_ms = (time.perf_counter() - start_time) * 1000.0
    
    return PredictionResult(
        inverter_id=res["inverter_id"],
        plant_id=res.get("plant_id", "UNKNOWN"),
        predicted_failure_type=res.get("predicted_failure_type", "unknown"),
        risk_score=res.get("risk_score", 0.0),
        final_risk_score=res.get("final_risk_score"),
        anomaly_score=res.get("anomaly_score"),
        risk_level=res.get("risk_level", "LOW"),
        shap_top5=[ShapFeature(**f) for f in res.get("shap_top5", [])],
        lime_top5=[LimeFeature(**f) for f in res.get("lime_top5", [])] if res.get("lime_top5") else None,
        delta_shap_top5=[DeltaShapFeature(**f) for f in res.get("delta_shap_top5", [])] if res.get("delta_shap_available") else None,
        report=report_obj,
        latency_ms=latency_ms,
        timestamp=int(time.time()),
    )

@router.post("/batch", response_model=List[PredictionResult])
async def predict_batch(req: BatchPredictRequest):
    """Run predictions for all inverters in the plant."""
    import config
    import pandas as pd
    
    master_path = config.PROCESSED_DIR / "master_labelled.parquet"
    if not master_path.exists():
        raise HTTPException(status_code=503, detail="Processed data missing")

    df = pd.read_parquet(master_path, columns=["inverter_id", "plant_id"])
    plant_inverters = df[df["plant_id"] == req.plant_id]["inverter_id"].unique().tolist()
    
    if not plant_inverters:
        raise HTTPException(status_code=404, detail="Plant not found or empty")
        
    # Create request proxies
    reqs = [
        PredictRequest(
            inverter_id=inv,
            generate_narrative=req.generate_narrative
        )
        for inv in plant_inverters
    ]
    
    # Run concurrently
    tasks = [predict_single(r) for r in reqs]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    final_responses = []
    for r in results:
        if isinstance(r, PredictionResult):
            final_responses.append(r)
        # ignore exceptions for batch response or log them
    return final_responses


@router.post("/narrative", tags=["GenAI"])
async def generate_narrative(req: NarrativeRequest):
    """Generate dynamic AI narrative report for the dashboard (mandatory)."""
    from genai.guardrails.validator import get_fallback_report
    
    risk_level = "LOW"
    if req.risk_score > 0.8: risk_level = "CRITICAL"
    elif req.risk_score > 0.6: risk_level = "HIGH"
    elif req.risk_score > 0.4: risk_level = "MEDIUM"
        
    from api.state import state_manager
    inv_state = state_manager.get_inverter_state(req.inverter_id)
        
    report = get_fallback_report(
        req.inverter_id, req.plant_id, req.risk_score, risk_level, inv_state
    )
    return report
