"""
Layer 5 — RAG Ingestion Pipeline.

Constructs document chunks containing structured telemetry and ML predictions
plus natural language summaries, then embeds and stores them in ChromaDB.
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional

import chromadb
import chromadb.utils.embedding_functions as embedding_functions
import structlog

import config
from genai.guardrails.validator import InverterReport

log = structlog.get_logger(__name__)


def _get_embedding_function():
    """Return embedding function depending on config availability."""
    if config.OPENAI_API_KEY:
        try:
            return embedding_functions.OpenAIEmbeddingFunction(
                api_key=config.OPENAI_API_KEY,
                model_name=config.EMBEDDING_MODEL,
            )
        except Exception as e:
            log.warning("openai_embedding_failed", error=str(e), fallback="default")
    return embedding_functions.DefaultEmbeddingFunction()


def get_chroma_client() -> chromadb.PersistentClient:
    """Return configured persistent ChromaDB client."""
    return chromadb.PersistentClient(path=str(config.CHROMA_PERSIST_DIR))


def ingest_inverter_status(
    inverter_id: str,
    plant_id: str,
    block_id: str,
    timestamp_unix: int,
    risk_score: float,
    risk_level: str,
    shap_top5: List[Dict[str, Any]],
) -> None:
    """Ingest a telemetry status snapshot into the RAG vector store.

    Args:
        inverter_id: Device ID.
        plant_id: Facility ID.
        block_id: Row/block ID.
        timestamp_unix: UNIX epoch timestamp of this prediction.
        risk_score: Float prediction probability.
        risk_level: String risk level.
        shap_top5: List of top driving features and their SHAP values.
    """
    client = get_chroma_client()
    ef = _get_embedding_function()

    collection = client.get_or_create_collection(
        name="inverter_status", embedding_function=ef
    )

    # 1. Construct natural language summary
    drivers = [f"{item['feature']} ({item['shap_value']:.4f})" for item in shap_top5]
    drivers_str = " and ".join(drivers[:2]) if drivers else "unknown factors"

    nl_summary = (
        f"Inverter {inverter_id} in Plant {plant_id} Block {block_id} "
        f"has {risk_level} risk ({risk_score:.2f}) driven by {drivers_str} "
        f"as of UNIX {timestamp_unix}."
    )

    # 2. Construct structured chunk
    structured_data = {
        "inverter_id": inverter_id,
        "plant_id": plant_id,
        "block_id": block_id,
        "timestamp_unix": timestamp_unix,
        "risk_score": risk_score,
        "risk_level": risk_level,
        "shap_top5": shap_top5,
    }

    # 3. Combine into single document body
    document_content = f"{nl_summary}\n\nSTRUCTURED_DATA:\n{json.dumps(structured_data, indent=2)}"

    # 4. Ingest
    doc_id = f"status_{inverter_id}_{timestamp_unix}"
    metadata = {
        "inverter_id": inverter_id,
        "plant_id": plant_id,
        "block_id": block_id,
        "risk_score": float(risk_score),
        "risk_level": risk_level,
        "timestamp": int(timestamp_unix),
        "doc_type": "status",
    }

    collection.upsert(
        documents=[document_content],
        metadatas=[metadata],
        ids=[doc_id],
    )
    log.info("ingested_status", doc_id=doc_id, inverter_id=inverter_id)


def ingest_inverter_report(report: InverterReport, timestamp_unix: int) -> None:
    """Ingest an AI diagnostic report into the RAG vector store.

    Args:
        report: Validated InverterReport object.
        timestamp_unix: UNIX epoch timestamp.
    """
    client = get_chroma_client()
    ef = _get_embedding_function()

    collection = client.get_or_create_collection(
        name="inverter_reports", embedding_function=ef
    )

    doc_id = f"report_{report.inverter_id}_{timestamp_unix}"
    document_content = report.model_dump_json(indent=2)

    metadata = {
        "inverter_id": report.inverter_id,
        "plant_id": report.plant_id,
        "risk_score": float(report.risk_score),
        "risk_level": report.risk_level.value,
        "timestamp": int(timestamp_unix),
        "doc_type": "report",
    }

    collection.upsert(
        documents=[document_content],
        metadatas=[metadata],  # type: ignore
        ids=[doc_id],
    )
    log.info("ingested_report", doc_id=doc_id, inverter_id=report.inverter_id)


def cleanup_old_documents() -> None:
    """Delete documents outside the 30-day/90-day rolling windows."""
    client = get_chroma_client()
    now = int(time.time())

    # 30 days = 30 * 24 * 3600 = 2592000 secs
    cutoff_30d = now - 2592000

    try:
        status_coll = client.get_collection("inverter_status")
        # Note: ChromaDB doesn't natively support range deletion via API easily in old versions,
        # but we can query then delete by IDs.
        results = status_coll.get(where={"timestamp": {"$lt": cutoff_30d}})
        if results and results["ids"]:
            status_coll.delete(ids=results["ids"])
            log.info("cleanup_status_done", deleted=len(results["ids"]))
    except Exception:
        pass  # Collection does not exist yet

    # 90 days = 90 * 24 * 3600 = 7776000 secs
    cutoff_90d = now - 7776000
    try:
        report_coll = client.get_collection("inverter_reports")
        results = report_coll.get(where={"timestamp": {"$lt": cutoff_90d}})
        if results and results["ids"]:
            report_coll.delete(ids=results["ids"])
            log.info("cleanup_reports_done", deleted=len(results["ids"]))
    except Exception:
        pass  # Collection does not exist yet
