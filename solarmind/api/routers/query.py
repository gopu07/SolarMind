"""
Layer 7 — Query router.
"""

import time
from fastapi import APIRouter, Depends
import structlog

from api.auth import get_current_user
from api.schemas.models import QueryRequest, QueryResponse, Citation
from rag.retriever import query as rag_query
from rag.state import get_session
from rag.llm_service import llm_service
import config
import re

log = structlog.get_logger(__name__)

router = APIRouter(
    prefix="/query",
    tags=["RAG"],
    dependencies=[Depends(get_current_user)],
)

@router.post("", response_model=QueryResponse)
async def query_rag(req: QueryRequest):
    """Run a grounded RAG query against the vector store."""
    start_time = time.perf_counter()
    
    results = rag_query(req.question, req.plant_id, req.top_k)
    
    citations = []
    context_chunks = []
    for r in results:
        meta = r["metadata"]
        content = r["content"]
        citations.append(
            Citation(
                inverter_id=meta.get("inverter_id", "UNKNOWN"),
                timestamp=int(meta.get("timestamp", 0)),
                risk_level=meta.get("risk_level", "UNKNOWN")
            )
        )
        context_chunks.append(f"Source [{meta.get('inverter_id', 'UNKNOWN')} "
                              f"at {meta.get('timestamp', 0)}]:\n{content}")
        
    # Generate answer via LLM
    answer = "LLM unavailable or API key missing."
    if config.OPENAI_API_KEY:
        try:
            import openai
            import pandas as pd
            import json
            
            client_kwargs = {"api_key": config.OPENAI_API_KEY}
            if config.OPENAI_BASE_URL:
                client_kwargs["base_url"] = config.OPENAI_BASE_URL
            client = openai.OpenAI(**client_kwargs)
            
            # Retrieve latest telemetry for diagnostic reasoning
            telemetry_str = "No recent telemetry found."
            master_path = config.PROCESSED_DIR / "master_labelled.parquet"
            if master_path.exists():
                try:
                    df = pd.read_parquet(master_path)
                    if req.plant_id and req.plant_id in df["plant_id"].values:
                        df = df[df["plant_id"] == req.plant_id].copy()
                        
                    if not df.empty:
                        df["timestamp"] = pd.to_datetime(df["timestamp"])
                        latest = df.sort_values("timestamp").groupby("inverter_id").last().reset_index()
                        
                        signals = [
                            "inverter_id", "pv1_power", "pv2_power", "pv1_voltage", "pv1_current", 
                            "inverter_temperature", "inverter_op_state", "inverter_alarm_code", 
                            "meter_active_power", "meter_v_r", "meter_v_y", "meter_v_b"
                        ] + [f"smu_string{i}" for i in range(1, 15)]
                        
                        avail_cols = [c for c in signals if c in latest.columns]
                        telemetry_str = latest[avail_cols].to_json(orient="records")
                except Exception as e:
                    log.error("telemetry_load_failed_for_prompt", error=str(e))
            
            # Extract Inverter ID if present
            inverter_match = re.search(r'(INV[-_]\d+)', req.question.upper())
            inverter_id = inverter_match.group(1) if inverter_match else None
            
            # Determine Intent
            intent = "general"
            q_lower = req.question.lower()
            if any(term in q_lower for term in ["plant summary", "plant health", "overall status"]):
                intent = "plant_summary"
            elif inverter_id:
                intent = "inverter_diagnostics"
            elif any(term in q_lower for term in ["yes", "show details", "tell me more"]):
                intent = "follow_up"
            
            # Update Session State
            session = get_session() # Using default session for now
            if inverter_id:
                session.last_inverter = inverter_id
                session.last_intent = "inverter_diagnostics"
            elif intent == "plant_summary":
                session.last_intent = "plant_summary"
            
            # Handle Follow-up Context
            effective_inverter = inverter_id
            if intent == "follow_up" and session.last_inverter:
                effective_inverter = session.last_inverter
                intent = "inverter_diagnostics"
            
            # Tailor Prompt based on Intent
            intent_instruction = ""
            if intent == "plant_summary":
                intent_instruction = """The user wants a PLANT SUMMARY. 
Provide a high-level overview of the entire plant's health. 
List any inverters with high risk scores (>=0.6) and summarize active alerts."""
            elif intent == "inverter_diagnostics":
                intent_instruction = f"""The user wants INVERTER DIAGNOSTICS for {effective_inverter}.
Focus specifically on {effective_inverter}. 
Explain its risk score, identify the top SHAP drivers, analyze its temperature trend relative to plant average, and provide specific maintenance recommendations."""
            
            full_prompt = f"""Documents:
{"".join(context_chunks)}

Telemetry:
{telemetry_str}

Logic:
If fault diagnosis requested, apply fault detection steps...

Question: {req.question}
"""
            answer = llm_service.generate_response(
                prompt=full_prompt,
                system_prompt=f"You are a solar PV expert. {intent_instruction}"
            )
        except Exception as e:
            log.warning("llm_qa_failed", error=str(e))
            answer = f"Error: {str(e)}"
        except Exception as e:
            log.warning("llm_qa_failed", error=str(e))
            answer = f"Error calling LLM: {str(e)}"
            
    latency_ms = (time.perf_counter() - start_time) * 1000.0
    
    return QueryResponse(
        answer=answer,
        citations=citations,
        latency_ms=latency_ms
    )
