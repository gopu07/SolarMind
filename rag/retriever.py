"""
Layer 5 — RAG Retrieval Pipeline.

Hybrid retriever over ChromaDB using custom re-ranking heuristics based on
recency and risk severity.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import structlog

import config
from rag.ingest import _get_embedding_function, get_chroma_client

log = structlog.get_logger(__name__)


def calculate_recency_score(timestamp: int, now: int) -> float:
    """Calculate linear decay recency score from 1.0 (today) to 0.0 (7+ days)."""
    age_seconds = now - timestamp
    if age_seconds <= 0:
        return 1.0

    days_old = age_seconds / 86400.0
    if days_old <= 1.0:
        return 1.0
    if days_old >= 7.0:
        return 0.0

    # Linear decay from day 1 to day 7
    # y = 1 - (days - 1) / 6
    return max(0.0, 1.0 - ((days_old - 1.0) / 6.0))


def query(
    question: str, plant_id: Optional[str] = None, top_k: int = 5
) -> List[Dict[str, Any]]:
    """Hybrid RAG retrieval with custom re-ranking.

    Args:
        question: User query string.
        plant_id: Optional plant ID for metadata filtering.
        top_k: Number of final results to return after re-ranking.

    Returns:
        List of dicts containing ``content``, ``metadata``, and ``combined_score``.
    """
    client = get_chroma_client()
    ef = _get_embedding_function()

    # Try both collections
    collections = []
    try:
        collections.append(client.get_collection("inverter_status", embedding_function=ef))
    except Exception:
        pass
    try:
        collections.append(client.get_collection("inverter_reports", embedding_function=ef))
    except Exception:
        pass

    if not collections:
        return []

    # Prepare where filter
    where_filter: Dict[str, Any] = {}
    if plant_id:
        where_filter["plant_id"] = plant_id

    # Add implicit filters based on query keywords
    # If a specific inverter like INV_001 is inside the text, we could filter:
    import re
    inv_match = re.search(r'(INV_\d{3})', question, re.IGNORECASE)
    if inv_match:
        # Instead of replacing where_filter, use an AND block if plant_id is also set.
        # But Chroma 'where' syntax can only have one operator at root unless $and is used.
        inv_id = inv_match.group(1).upper()
        if plant_id:
            where_filter = {"$and": [{"plant_id": plant_id}, {"inverter_id": inv_id}]}
        else:
            where_filter = {"inverter_id": inv_id}

    query_params = {
        "query_texts": [question],
        "n_results": 10,
        "include": ["documents", "metadatas", "distances"]
    }
    if where_filter:
        query_params["where"] = where_filter

    all_results = []
    now = int(time.time())

    for coll in collections:
        try:
            res = coll.query(**query_params)
            
            if not res["documents"] or not res["documents"][0]:
                continue
                
            docs = res["documents"][0]
            metadatas = res["metadatas"][0] if res["metadatas"] else [{}] * len(docs)
            distances = res["distances"][0] if res["distances"] else [1.0] * len(docs)
            
            for doc, meta, dist in zip(docs, metadatas, distances):
                # distance is cosine distance (0 is exact, 2 is opposite)
                # similarity = 1 - (distance/2) to map [0,2] -> [1,0]
                similarity = max(0.0, 1.0 - (dist / 2.0))
                
                # Normalise risk score (it's already 0-1)
                risk_score = float(meta.get("risk_score", 0.0))
                
                # Recency score
                timestamp = int(meta.get("timestamp", 0))
                recency = calculate_recency_score(timestamp, now)
                
                # Re-rank formula
                # combined_score = 0.5 * vector_similarity + 0.3 * risk_score_normalised + 0.2 * recency_score
                combined = (0.5 * similarity) + (0.3 * risk_score) + (0.2 * recency)
                
                all_results.append({
                    "content": doc,
                    "metadata": meta,
                    "combined_score": combined,
                    "metrics": {
                        "similarity": similarity,
                        "risk": risk_score,
                        "recency": recency
                    }
                })
        except Exception as e:
            log.warning("query_collection_failed", collection=coll.name, error=str(e))

    # Sort and take top_k
    all_results.sort(key=lambda x: x["combined_score"], reverse=True)
    return all_results[:top_k]
