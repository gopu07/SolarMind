"""
Layer 6 — Agentic Workflow.

Implements a LangGraph StateGraph that orchestrates telemetry retrieval,
risk prediction, AI narrative generation, and ticketing/notification dispatch.
"""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any, Dict, List, Optional, TypedDict

import httpx
import pandas as pd
import structlog
from langgraph.graph import END, StateGraph

import config
from features.pipeline import FEATURE_COLUMNS
from genai.guardrails.validator import (
    ConfidenceLevel,
    DataQuality,
    InverterReport,
    get_fallback_report,
    parse_llm_response,
)
from models.predict import _assemble_feature_vector, _error_result, _load_recent_telemetry, predict_inverter

log = structlog.get_logger(__name__)


# =====================================================================
# State Definition
# =====================================================================
class AgentState(TypedDict):
    inverter_id: str
    telemetry_df: Optional[pd.DataFrame]
    features: Optional[Dict[str, Any]]
    risk_score: Optional[float]
    risk_level: Optional[str]
    shap_now: Optional[Dict[str, float]]
    delta_shap: Optional[Dict[str, float]]
    delta_shap_available: bool
    risk_score_24h: Optional[float]
    report: Optional[InverterReport]
    ticket_id: Optional[str]
    notification_sent: bool
    errors: List[str]
    skipped: bool
    skip_reason: Optional[str]


# =====================================================================
# Graph Nodes
# =====================================================================
def node_telemetry_retriever(state: AgentState) -> AgentState:
    """Load the last 672 rows from master_labelled.parquet."""
    inverter_id = state["inverter_id"]
    try:
        df = _load_recent_telemetry(inverter_id, lookback_rows=672)
        if df is None or len(df) < 96:
            state["skipped"] = True
            state["skip_reason"] = "insufficient_telemetry"
        else:
            state["telemetry_df"] = df
    except Exception as e:
        state["errors"].append(f"Telemetry missing: {str(e)}")
        state["skipped"] = True
        state["skip_reason"] = "telemetry_error"
    return state


def node_feature_engineer(state: AgentState) -> AgentState:
    """Call the feature pipeline in streaming mode."""
    if state.get("skipped"):
        return state

    inverter_id = state["inverter_id"]
    df = state["telemetry_df"]
    try:
        feat_df = _assemble_feature_vector(df, inverter_id)
        if feat_df is None or feat_df.empty:
            state["errors"].append("Feature computation yielded empty result")
        else:
            state["features"] = feat_df.iloc[-1].to_dict()
    except Exception as e:
        state["errors"].append(f"Feature computation failed: {str(e)}")
    return state


def node_risk_predictor(state: AgentState) -> AgentState:
    """Call models/predict.py to get scores and SHAP."""
    if state.get("skipped") or state.get("errors"):
        return state

    inverter_id = state["inverter_id"]
    try:
        # predict_inverter already handles telemetry and features centrally,
        # but the spec asks to call it to get score and shap.
        res = predict_inverter(inverter_id, include_delta_shap=True)
        if "error" in res and res["error"] != "NONE":
            state["errors"].append(f"Prediction failed: {res['error']}")
        else:
            state["risk_score"] = res["risk_score"]
            state["risk_level"] = res["risk_level"]
            state["shap_now"] = res["shap_now"]
            
            ds_avail = res.get("delta_shap_available", False)
            state["delta_shap_available"] = ds_avail
            if ds_avail and res.get("delta_shap_top5"):
                # Convert list of dicts back to simple dict for state
                ds_dict = {item["feature"]: item["delta_shap"] for item in res["delta_shap_top5"]}
                state["delta_shap"] = ds_dict
            else:
                state["delta_shap"] = None
                
            state["risk_score_24h"] = res.get("risk_score_24h")
    except Exception as e:
        state["errors"].append(f"Prediction node failed: {str(e)}")
    return state


def router_threshold(state: AgentState) -> str:
    """Determine if risk warrants narrative generation.
    
    Logic: risk > 0.75 AND increasing trend (risk > risk_24h).
    """
    if state.get("skipped") or state.get("errors"):
        return "END"

    risk = state.get("risk_score", 0)
    risk_24h = state.get("risk_score_24h", 0) or 0
    
    # User requirement: risk > 0.75 and increasing trend
    if risk > 0.75 and risk > risk_24h:
        return "NarrativeGenerator"
    
    state["skipped"] = True
    state["skip_reason"] = "below_threshold_or_stable"
    return "END"


def _call_llm_json(prompt: str) -> Optional[str]:
    """Call OpenAI, requesting JSON."""
    if not config.OPENAI_API_KEY:
        log.warning("openai_key_missing", fallback="True")
        return None
        
    try:
        import openai
        client_kwargs = {"api_key": config.OPENAI_API_KEY}
        if config.OPENAI_BASE_URL:
            client_kwargs["base_url"] = config.OPENAI_BASE_URL
        client = openai.OpenAI(**client_kwargs)
        response = client.chat.completions.create(
            model=config.LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.1,
            timeout=15.0
        )
        return response.choices[0].message.content
    except Exception as e:
        log.error("llm_call_failed", error=str(e))
        return None


def node_narrative_generator(state: AgentState) -> AgentState:
    """Generate narrative via GenAI."""
    inverter_id = state["inverter_id"]
    try:
        # Build prompt from v2 template
        prompt_path = config.PROJECT_ROOT / "genai" / "prompts" / "v2_production.txt"
        with open(prompt_path, "r") as f:
            template = f.read()

        df = state["telemetry_df"]
        plant_id = str(df["plant_id"].iloc[-1]) if df is not None and "plant_id" in df.columns else "UNKNOWN"
        block_id = str(df["block_id"].iloc[-1]) if df is not None and "block_id" in df.columns else "UNKNOWN"
        label_source = str(df["label_source"].iloc[-1]) if df is not None and "label_source" in df.columns else "negative"
        
        feat = state.get("features", {})
        ds_top5 = state.get("delta_shap")
        if ds_top5:
            ds_str = "\n".join([f"- {k}: {v:.4f}" for k, v in list(ds_top5.items())[:5]])
        else:
            ds_str = "Not available — telemetry gap at T-24h"

        shap_now = state.get("shap_now", {})
        shap_str = "\n".join([f"- {k}: {v:.4f}" for k, v in list(shap_now.items())[:5]])

        prompt = template.format(
            inverter_id=inverter_id,
            plant_id=plant_id,
            block_id=block_id,
            risk_score=round(state.get("risk_score", 0), 2),
            risk_level=state.get("risk_level", "LOW"),
            shap_top5=shap_str,
            delta_shap_top5=ds_str,
            delta_shap_available=str(state.get("delta_shap_available", False)),
            data_quality="COMPLETE" if state.get("delta_shap_available") else "PARTIAL",
            avg_temp=round(feat.get("temp_rolling_mean_24h", 0), 2),
            max_temp=round(feat.get("temp_rolling_max_24h", 0), 2),
            current_efficiency=round(feat.get("conversion_efficiency", 0), 3),
            efficiency_trend=round(feat.get("efficiency_7d_trend", 0), 5),
            string_mismatch_cv=round(feat.get("string_mismatch_cv", 0), 3),
            power_vs_24h_baseline=round(feat.get("power_vs_24h_baseline", 0), 3),
            label_source=label_source
        )

        resp_text = _call_llm_json(prompt)
        report = None
        
        if resp_text:
            try:
                report = parse_llm_response(resp_text, {})
            except Exception:
                # Retry once
                retry_prompt = prompt + "\n\nYour previous response was not valid JSON. Return only the JSON object with no other text."
                resp_text_2 = _call_llm_json(retry_prompt)
                if resp_text_2:
                    try:
                        report = parse_llm_response(resp_text_2, {})
                    except Exception:
                        pass
        
        if report is None:
            # Fallback
            report = get_fallback_report(
                inverter_id, plant_id, state.get("risk_score", 0), state.get("risk_level", "LOW")
            )
            
        state["report"] = report

    except Exception as e:
        log.error("narrative_generator_failed", error=str(e), inverter_id=inverter_id)
        df_safe = state.get("telemetry_df")
        pid = str(df_safe["plant_id"].iloc[-1]) if df_safe is not None and "plant_id" in df_safe.columns else "UNKNOWN"
        state["report"] = get_fallback_report(inverter_id, pid, state.get("risk_score", 0), state.get("risk_level", "LOW"))

    return state


async def _post_with_retry(url: str, json_data: dict) -> Optional[httpx.Response]:
    """POST with 1 retry and timeout."""
    import httpx
    timeout = httpx.Timeout(10.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        for attempt in range(2):
            try:
                resp = await client.post(url, json=json_data)
                return resp
            except Exception as e:
                log.warning("post_attempt_failed", url=url, attempt=attempt, error=str(e))
                if attempt == 0:
                    await asyncio.sleep(2)
    return None


def node_ticket_agent(state: AgentState) -> AgentState:
    """Simulate POST to CMMS."""
    report: InverterReport = state.get("report")
    if not report:
        return state

    sla = "7 days"
    if report.risk_level == "CRITICAL":
        sla = "24 hours"
    elif report.risk_level == "HIGH":
        sla = "48 hours"

    payload = {
        "inverter_id": report.inverter_id,
        "plant_id": report.plant_id,
        "risk_score": report.risk_score,
        "risk_level": report.risk_level.value,
        "action": report.action,
        "summary": report.summary,
        "causal_drivers": [d.model_dump() for d in report.causal_drivers] if report.causal_drivers else [],
        "sla_window": sla
    }

    # Run the async post synchronously for the graph node
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
    try:
        resp = loop.run_until_complete(_post_with_retry(config.CMMS_ENDPOINT, payload))
        if resp and resp.status_code in (200, 201):
            data = resp.json()
            state["ticket_id"] = data.get("ticket_id", f"MOCK-TK-{report.inverter_id}")
        else:
            state["ticket_id"] = f"MOCK-TK-{report.inverter_id}" # Default fallback for hackathon
            log.info("cmms_mock_ticket_created", ticket_id=state["ticket_id"])
    except Exception as e:
        log.error("cmms_post_failed", error=str(e))
        state["ticket_id"] = f"FAILED-TK-{report.inverter_id}"
        
    return state


def node_notification_agent(state: AgentState) -> AgentState:
    """Simulate POST to Slack."""
    if not config.SLACK_WEBHOOK_URL:
        # Skip silently if not configured as per spec
        state["notification_sent"] = False
        return state

    report: InverterReport = state.get("report")
    if not report:
        return state

    first_sentence = report.summary.split(".")[0] + "."
    payload = {
        "text": f"🚨 *{report.risk_level.value} Risk* detected on Inverter `{report.inverter_id}` (Plant {report.plant_id}).\n"
                f"> {first_sentence}\n"
                f"Ticket: {state.get('ticket_id', 'None')}"
    }

    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
    try:
        resp = loop.run_until_complete(_post_with_retry(config.SLACK_WEBHOOK_URL, payload))
        if resp and resp.status_code == 200:
            state["notification_sent"] = True
        else:
            state["notification_sent"] = False
    except Exception:
        state["notification_sent"] = False

    return state


# =====================================================================
# Graph Construction & Compilation
# =====================================================================
def build_graph():
    """Compile the LangGraph workflow."""
    workflow = StateGraph(AgentState)

    workflow.add_node("TelemetryRetriever", node_telemetry_retriever)
    workflow.add_node("FeatureEngineer", node_feature_engineer)
    workflow.add_node("RiskPredictor", node_risk_predictor)
    workflow.add_node("NarrativeGenerator", node_narrative_generator)
    workflow.add_node("TicketAgent", node_ticket_agent)
    workflow.add_node("NotificationAgent", node_notification_agent)

    workflow.set_entry_point("TelemetryRetriever")

    workflow.add_edge("TelemetryRetriever", "FeatureEngineer")
    workflow.add_edge("FeatureEngineer", "RiskPredictor")
    
    workflow.add_conditional_edges(
        "RiskPredictor",
        router_threshold,
        {
            "END": END,
            "NarrativeGenerator": "NarrativeGenerator"
        }
    )

    workflow.add_edge("NarrativeGenerator", "TicketAgent")
    workflow.add_edge("TicketAgent", "NotificationAgent")
    workflow.add_edge("NotificationAgent", END)

    return workflow.compile()


_agent_runner = build_graph()


def run_agent(inverter_id: str) -> AgentState:
    """Run the agent for a single inverter.
    
    Args:
        inverter_id: Target identifier.
        
    Returns:
        Final AgentState dict.
    """
    initial_state = AgentState(
        inverter_id=inverter_id,
        telemetry_df=None,
        features=None,
        risk_score=None,
        risk_level=None,
        shap_now=None,
        delta_shap=None,
        delta_shap_available=False,
        risk_score_24h=None,
        report=None,
        ticket_id=None,
        notification_sent=False,
        errors=[],
        skipped=False,
        skip_reason=None
    )
    
    log.info("starting_agent", inverter_id=inverter_id)
    final_state = _agent_runner.invoke(initial_state)
    log.info(
        "agent_finished", 
        inverter_id=inverter_id, 
        skipped=final_state.get("skipped"),
        ticket=final_state.get("ticket_id"),
        errors_count=len(final_state.get("errors", []))
    )
    return final_state


async def run_plant(plant_id: str):
    """Run the agent for all inverters in a specified plant concurrently."""
    # Read unique inverters for plant from master_labelled
    master_path = config.PROCESSED_DIR / "master_labelled.parquet"
    if not master_path.exists():
        log.error("master_labelled_missing", path=str(master_path))
        return []

    df = pd.read_parquet(master_path, columns=["inverter_id", "plant_id"])
    plant_inverters = df[df["plant_id"] == plant_id]["inverter_id"].unique().tolist()
    
    log.info("running_plant_batch", plant_id=plant_id, total_inverters=len(plant_inverters))
    
    # Run concurrently via asyncio wrappers
    loop = asyncio.get_event_loop()
    
    # Create wrapper to run blocking `run_agent` in executor
    import functools
    tasks = [
        loop.run_in_executor(None, functools.partial(run_agent, inv))
        for inv in plant_inverters
    ]
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    return results
