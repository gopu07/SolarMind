"""
Layer 5 — RAG Ingestion Pipeline.

Constructs document chunks containing structured telemetry and ML predictions
plus natural language summaries, then embeds and stores them in ChromaDB.
Also handles knowledge base ingestion and maintenance history storage.
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
    try:
        return embedding_functions.DefaultEmbeddingFunction()
    except Exception as e:
        log.warning("default_embedding_init_failed", error=str(e), hint="Attempting SentenceTransformer fallback")
        try:
            return embedding_functions.SentenceTransformerEmbeddingFunction()
        except:
            log.error("all_embeddings_failed")
            # If everything fails, return a function that returns zeros to keep system alive
            class ZeroEmbedding:
                def __call__(self, texts): return [[0.0]*384 for _ in texts]
            return ZeroEmbedding()


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
    
    # Enrich the document content with a natural language header for better similarity search
    nl_header = (
        f"AI Diagnostic Report for Inverter {report.inverter_id} (Plant {report.plant_id}). "
        f"Detected {report.risk_level.value} risk of failure. "
        f"Summary: {report.summary} "
        f"Recommended Action: {report.action} "
        f"Root Cause Analysis: {report.root_cause}"
    )
    
    document_content = f"{nl_header}\n\nFULL_REPORT_JSON:\n{report.model_dump_json(indent=2)}"

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


# =====================================================================
# Knowledge Base Ingestion (Step 2)
# =====================================================================
def ingest_knowledge_base() -> int:
    """Parse structured YAML knowledge documents and ingest into ChromaDB.

    Reads all ``.yaml`` files from ``config.KNOWLEDGE_BASE_DIR``,
    converts each chunk into a combined natural-language document, and
    upserts into the ``knowledge_base`` collection.

    Returns:
        Total number of chunks ingested.
    """
    import yaml

    kb_dir = config.KNOWLEDGE_BASE_DIR
    if not kb_dir.exists():
        log.warning("knowledge_base_dir_missing", path=str(kb_dir))
        return 0

    client = get_chroma_client()
    ef = _get_embedding_function()
    collection = client.get_or_create_collection(
        name="knowledge_base", embedding_function=ef
    )

    total = 0
    for yaml_file in sorted(kb_dir.glob("*.yaml")):
        try:
            with open(yaml_file, "r", encoding="utf-8") as f:
                doc = yaml.safe_load(f)
        except Exception as e:
            log.warning("yaml_parse_failed", file=str(yaml_file), error=str(e))
            continue

        concept = doc.get("concept", yaml_file.stem)
        category = doc.get("category", "general")

        for chunk in doc.get("chunks", []):
            chunk_id = chunk.get("id", f"{concept}_{total}")
            title = chunk.get("title", "Untitled")

            # Build NL document from structured fields
            parts = [f"# {title}\n", f"Concept: {concept} | Category: {category}\n"]

            if chunk.get("symptoms"):
                parts.append("## Symptoms")
                for s in chunk["symptoms"]:
                    parts.append(f"- {s}")

            if chunk.get("sensor_signals"):
                parts.append("\n## Sensor Signals")
                for s in chunk["sensor_signals"]:
                    parts.append(f"- {s}")

            if chunk.get("probable_causes"):
                parts.append("\n## Probable Causes")
                for c in chunk["probable_causes"]:
                    parts.append(f"- {c}")

            if chunk.get("recommended_actions"):
                parts.append("\n## Recommended Actions")
                for a in chunk["recommended_actions"]:
                    parts.append(f"- {a}")

            document_content = "\n".join(parts)

            metadata = {
                "concept": concept,
                "category": category,
                "chunk_id": chunk_id,
                "title": title,
                "doc_type": "knowledge_base",
                "timestamp": int(time.time()),
            }

            collection.upsert(
                documents=[document_content],
                metadatas=[metadata],
                ids=[chunk_id],
            )
            total += 1

    log.info("knowledge_base_ingested", total_chunks=total)
    return total


# =====================================================================
# Maintenance History Ingestion (Step 6)
# =====================================================================
def ingest_maintenance_event(
    event_id: str,
    inverter_id: str,
    plant_id: str,
    event_type: str,
    description: str,
    resolution: str,
    timestamp_unix: int,
    root_cause: str = "",
    severity: str = "MEDIUM",
) -> None:
    """Ingest a historical maintenance / failure event into ChromaDB.

    Args:
        event_id: Unique identifier for this event.
        inverter_id: Related device ID.
        plant_id: Facility ID.
        event_type: Category (e.g., "fan_failure", "igbt_failure").
        description: Free-text description of the event.
        resolution: What was done to resolve it.
        timestamp_unix: When the event occurred.
        root_cause: Determined root cause, if available.
        severity: LOW / MEDIUM / HIGH / CRITICAL.
    """
    client = get_chroma_client()
    ef = _get_embedding_function()
    collection = client.get_or_create_collection(
        name="maintenance_history", embedding_function=ef
    )

    nl_summary = (
        f"Maintenance Event {event_id} for Inverter {inverter_id} "
        f"(Plant {plant_id}). Type: {event_type}. Severity: {severity}. "
        f"Description: {description} "
        f"Root Cause: {root_cause or 'undetermined'}. "
        f"Resolution: {resolution}"
    )

    metadata = {
        "event_id": event_id,
        "inverter_id": inverter_id,
        "plant_id": plant_id,
        "event_type": event_type,
        "severity": severity,
        "timestamp": int(timestamp_unix),
        "doc_type": "maintenance_history",
    }

    collection.upsert(
        documents=[nl_summary],
        metadatas=[metadata],
        ids=[event_id],
    )
    log.info("ingested_maintenance_event", event_id=event_id)


# =====================================================================
# BM25 Corpus Builder
# =====================================================================
def build_bm25_corpus(
    collection_names: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Retrieve all documents from specified ChromaDB collections.

    Used to build a BM25 index for keyword search alongside vector retrieval.

    Args:
        collection_names: Collections to include; defaults to all RAG collections.

    Returns:
        List of document dicts with ``content`` and ``metadata``.
    """
    if collection_names is None:
        collection_names = [
            "inverter_status",
            "inverter_reports",
            "knowledge_base",
            "maintenance_history",
        ]

    client = get_chroma_client()
    all_docs: List[Dict[str, Any]] = []

    for name in collection_names:
        try:
            coll = client.get_collection(name)
            result = coll.get(include=["documents", "metadatas"])
            if result and result["documents"]:
                for doc, meta in zip(
                    result["documents"],
                    result["metadatas"] or [{}] * len(result["documents"]),
                ):
                    all_docs.append({"content": doc, "metadata": meta})
        except Exception:
            pass  # collection does not exist yet

    return all_docs
