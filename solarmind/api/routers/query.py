"""
Layer 7 — Advanced Query Router.

Integrates the multi-stage hybrid RAG pipeline with telemetry-aware context
injection and chain-of-thought diagnostic reasoning.
"""

import json
import re
import time
from typing import Any, Dict, Optional

import structlog
from fastapi import APIRouter, Depends

import config
from api.auth import get_current_user
from api.schemas.models import (
    Citation,
    DiagnosticReport,
    QueryRequest,
    QueryResponse,
    ReasoningChain,
    RecommendedAction,
    SensorEvidence,
    SimilarPastEvent,
)
from rag.retriever import hybrid_query
from rag.state import get_session
from rag.llm_service import llm_service
from rag.telemetry_context import format_telemetry_for_prompt
from api.routers.timeline import get_timeline_events
from api.routers.maintenance import get_maintenance_schedule

log = structlog.get_logger(__name__)

router = APIRouter(
    prefix="/query",
    tags=["RAG"],
)


def _load_v3_prompt() -> str:
    """Load the v3 chain-of-thought diagnostic prompt template."""
    prompt_path = config.PROJECT_ROOT / "genai" / "prompts" / "v3_diagnostic.txt"
    try:
        with open(prompt_path, "r") as f:
            return f.read()
    except FileNotFoundError:
        log.warning("v3_prompt_missing", fallback="v2")
        return (
            "You are a solar PV expert. Analyze the following context and "
            "telemetry data to answer the question. Return JSON with keys: "
            "diagnosis, risk_level, root_cause_hypothesis, sensor_evidence, "
            "recommended_actions, similar_past_events, reasoning_chain, "
            "confidence, data_quality.\n\n{question}"
        )


def _parse_diagnostic_report(raw: str) -> Optional[DiagnosticReport]:
    """Attempt to parse LLM response into a DiagnosticReport."""
    cleaned = raw.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    elif cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    cleaned = cleaned.strip()

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        return None

    try:
        sensor_evidence = [
            SensorEvidence(**se) for se in data.get("sensor_evidence", [])
        ]
        recommended_actions = [
            RecommendedAction(**ra) for ra in data.get("recommended_actions", [])
        ]
        similar_past_events = [
            SimilarPastEvent(**spe) for spe in data.get("similar_past_events", [])
        ]
        reasoning_chain = None
        if data.get("reasoning_chain"):
            reasoning_chain = ReasoningChain(**data["reasoning_chain"])

        return DiagnosticReport(
            diagnosis=data.get("diagnosis", "Unable to determine diagnosis"),
            risk_level=data.get("risk_level", "MEDIUM"),
            root_cause_hypothesis=data.get("root_cause_hypothesis", "Insufficient data for root cause analysis"),
            sensor_evidence=sensor_evidence,
            recommended_actions=recommended_actions,
            similar_past_events=similar_past_events,
            reasoning_chain=reasoning_chain,
            confidence=data.get("confidence", "MEDIUM"),
            data_quality=data.get("data_quality", "PARTIAL"),
        )
    except Exception as e:
        log.warning("diagnostic_report_parse_failed", error=str(e))
        return None


@router.post("", response_model=QueryResponse)
async def query_rag(req: QueryRequest):
    """Run a grounded multi-stage RAG query with diagnostic reasoning."""
    start_time = time.perf_counter()

    # 1. Hybrid retrieval
    retrieval_result = hybrid_query(
        question=req.question,
        plant_id=req.plant_id,
        top_k=req.top_k,
        enable_multi_query=req.enable_multi_query,
        enable_reranking=req.enable_reranking,
    )

    results = retrieval_result["results"]
    history = retrieval_result["history"]
    stats = retrieval_result["stats"]

    # Build citations
    citations = []
    context_chunks = []
    for r in results:
        meta = r.get("metadata", {})
        content = r.get("content", "")
        citations.append(
            Citation(
                inverter_id=meta.get("inverter_id", "KNOWLEDGE_BASE"),
                timestamp=int(meta.get("timestamp", 0)),
                risk_level=meta.get("risk_level", meta.get("severity", "INFO")),
                relevance_score=round(r.get("rerank_score", r.get("combined_score", 0)), 3),
                retrieval_method=r.get("retrieval_method", "hybrid"),
            )
        )
        context_chunks.append(
            f"Source [{meta.get('inverter_id', meta.get('concept', 'UNKNOWN'))} "
            f"at {meta.get('timestamp', 0)} | "
            f"score={r.get('combined_score', 0):.3f}]:\n{content}"
        )

    # Build historical context
    history_parts = []
    for h in history:
        hmeta = h.get("metadata", {})
        history_parts.append(
            f"[{hmeta.get('event_type', 'unknown')} | "
            f"{hmeta.get('inverter_id', '?')} | "
            f"sim={h.get('similarity', 0):.3f}]: "
            f"{h.get('content', '')[:300]}"
        )

    # 2. Telemetry context injection
    inv_match = re.search(r"(INV[-_]\d+)", req.question.upper())
    inverter_id = inv_match.group(1).replace("-", "_") if inv_match else None

    telemetry_str, anomalies = format_telemetry_for_prompt(
        inverter_id=inverter_id,
        plant_id=req.plant_id,
    )

    anomaly_summary = "No anomalies detected."
    if anomalies:
        anomaly_lines = [f"- [{a['severity']}] {a['description']}" for a in anomalies]
        anomaly_summary = f"{len(anomalies)} anomalies detected:\n" + "\n".join(anomaly_lines)

    # Session state
    session = get_session(req.session_id)
    if inverter_id:
        session.last_inverter = inverter_id
        session.last_intent = "inverter_diagnostics"

    # Context header
    context_header = ""
    if inverter_id:
        context_header = f"Focus on Inverter {inverter_id}"
        if req.plant_id:
            context_header += f" in Plant {req.plant_id}"
    elif req.plant_id:
        context_header = f"Plant-wide analysis for {req.plant_id}"

    # 3. LLM generation with chain-of-thought
    answer = "LLM unavailable. Falling back to rule-based summary."
    diagnostic_report = None

    if config.GEMINI_API_KEY or config.OPENAI_API_KEY:
        try:
            template = _load_v3_prompt()

            timeline_events = await get_timeline_events()
            maintenance_tasks = await get_maintenance_schedule()
            
            timeline_context = ""
            if timeline_events:
                for ev in timeline_events:
                    timeline_context += f"- Inverter {ev.inverter_id}: {ev.failure_type} likely in {ev.predicted_failure_hours}h (Risk: {ev.risk_score})\n"
            else:
                timeline_context = "No upcoming failures predicted.\n"
                
            maintenance_context = ""
            if maintenance_tasks:
                for t in maintenance_tasks:
                    maintenance_context += f"- [{t.priority}] Inverter {t.inverter_id} by {t.recommended_time}: {t.recommended_action}\n"
            else:
                maintenance_context = "No scheduled maintenance tasks.\n"

            full_prompt = template.format(
                context_header=context_header,
                knowledge_context="\n\n".join(context_chunks),
                telemetry_context=telemetry_str,
                anomaly_summary=anomaly_summary,
                historical_context="\n".join(history_parts) if history_parts else "No similar historical events found.",
                rag_documents=f"{len(results)} documents retrieved via hybrid search.",
                conversation_history=json.dumps(session.history, indent=2) if session.history else "No previous conversation.",
                timeline_context=timeline_context,
                maintenance_context=maintenance_context,
                question=req.question,
            )

            raw_answer = llm_service.generate_response(full_prompt, system_prompt="Return JSON diagnostic report.")
            diagnostic_report = _parse_diagnostic_report(raw_answer)

            if diagnostic_report:
                answer = (
                    f"**Diagnosis:** {diagnostic_report.diagnosis}\n\n"
                    f"**Risk Level:** {diagnostic_report.risk_level}\n\n"
                    f"**Root Cause:** {diagnostic_report.root_cause_hypothesis}\n\n"
                )
                if diagnostic_report.recommended_actions:
                    answer += "**Recommended Actions:**\n"
                    for ra in diagnostic_report.recommended_actions:
                        answer += f"- [{ra.priority}] {ra.action}\n"
            else:
                answer = raw_answer

            session.history.append({"role": "user", "content": req.question})
            session.history.append({"role": "assistant", "content": answer})
            if len(session.history) > 10:
                session.history = session.history[-10:]

        except Exception as e:
            log.warning("llm_qa_failed", error=str(e))
            answer = f"Error calling LLM: {str(e)}"

    latency_ms = (time.perf_counter() - start_time) * 1000.0

    return QueryResponse(
        answer=answer,
        citations=citations,
        diagnostic_report=diagnostic_report,
        retrieval_stats=stats,
        latency_ms=latency_ms,
    )
