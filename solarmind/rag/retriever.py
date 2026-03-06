"""
Layer 5 — Advanced Multi-Stage RAG Retrieval Pipeline.

Implements hybrid retrieval (vector + BM25 + metadata), multi-query expansion,
cross-encoder reranking, and historical memory for expert-level diagnostics.
"""

from __future__ import annotations

import re
import time
from typing import Any, Dict, List, Optional

import structlog
from rank_bm25 import BM25Okapi

import config
from rag.ingest import _get_embedding_function, get_chroma_client

log = structlog.get_logger(__name__)


# =====================================================================
# Scoring helpers
# =====================================================================
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
    return max(0.0, 1.0 - ((days_old - 1.0) / 6.0))


def _tokenize(text: str) -> List[str]:
    """Simple whitespace + punctuation tokenizer for BM25."""
    return re.findall(r"\w+", text.lower())


# =====================================================================
# BM25 Keyword Search
# =====================================================================
def bm25_search(
    query: str, documents: List[Dict[str, Any]], top_k: int = 15
) -> List[Dict[str, Any]]:
    """Perform BM25 keyword search over a list of document dicts.

    Each doc should have a ``content`` key.  Returns scored results.
    """
    if not documents:
        return []

    corpus = [_tokenize(doc["content"]) for doc in documents]
    bm25 = BM25Okapi(corpus)

    query_tokens = _tokenize(query)
    scores = bm25.get_scores(query_tokens)

    max_score = max(scores) if max(scores) > 0 else 1.0

    results = []
    for doc, score in zip(documents, scores):
        results.append({
            **doc,
            "bm25_score": float(score / max_score),  # normalise to [0, 1]
        })

    results.sort(key=lambda x: x["bm25_score"], reverse=True)
    return results[:top_k]


# =====================================================================
# Multi-Query Expansion
# =====================================================================
def expand_queries(question: str, n: int = 3) -> List[str]:
    """Generate *n* semantic variations of the user question via LLM.

    Falls back to simple heuristic expansions if the LLM is unavailable.
    """
    queries = [question]

    # Try LLM expansion first
    if config.OPENAI_API_KEY:
        try:
            import openai

            client_kwargs: Dict[str, Any] = {"api_key": config.OPENAI_API_KEY}
            if config.OPENAI_BASE_URL:
                client_kwargs["base_url"] = config.OPENAI_BASE_URL
            client = openai.OpenAI(**client_kwargs)

            resp = client.chat.completions.create(
                model=config.LLM_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a solar plant expert. Generate exactly "
                            f"{n - 1} alternative search queries for the "
                            "following question. Each query should capture a "
                            "different aspect or use different technical "
                            "terminology. Return ONLY the queries, one per "
                            "line, with no numbering or bullets."
                        ),
                    },
                    {"role": "user", "content": question},
                ],
                temperature=0.7,
                max_tokens=300,
            )
            raw = resp.choices[0].message.content or ""
            for line in raw.strip().splitlines():
                line = line.strip().lstrip("0123456789.-) ")
                if line and line != question:
                    queries.append(line)
        except Exception as e:
            log.warning("multi_query_expansion_failed", error=str(e))

    # Heuristic fallback: append simple reformulations
    if len(queries) < n:
        # Add a diagnostic-focused variant
        queries.append(f"What are the possible causes and faults for: {question}")
        # Add a maintenance variant
        queries.append(f"Recommended maintenance actions for: {question}")

    return queries[:n]


# =====================================================================
# Cross-encoder Reranking (via LLM)
# =====================================================================
def rerank_results(
    question: str,
    candidates: List[Dict[str, Any]],
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    """Rerank candidates using LLM-based relevance scoring.

    For each candidate, asks the LLM to rate relevance 0–10.
    Falls back to existing ``combined_score`` if the LLM is unavailable.
    """
    if not candidates:
        return []
    if not config.OPENAI_API_KEY:
        return candidates[:top_k]

    try:
        import openai, json as _json

        client_kwargs: Dict[str, Any] = {"api_key": config.OPENAI_API_KEY}
        if config.OPENAI_BASE_URL:
            client_kwargs["base_url"] = config.OPENAI_BASE_URL
        client = openai.OpenAI(**client_kwargs)

        # Build compact summaries for the batch
        doc_summaries = []
        for i, cand in enumerate(candidates[:config.RAG_RERANK_TOP_K]):
            content = cand.get("content", "")[:400]
            doc_summaries.append(f"[{i}] {content}")

        batch_text = "\n\n".join(doc_summaries)

        resp = client.chat.completions.create(
            model=config.LLM_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a relevance judge for a solar plant diagnostic system. "
                        "Rate each document's relevance to the question on a scale of 0-10. "
                        "Return ONLY a JSON object mapping document index to score, e.g. "
                        '{"0": 8, "1": 3, "2": 9}.'
                    ),
                },
                {
                    "role": "user",
                    "content": f"Question: {question}\n\nDocuments:\n{batch_text}",
                },
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
            max_tokens=200,
        )

        raw_scores = _json.loads(resp.choices[0].message.content or "{}")

        for idx_str, score in raw_scores.items():
            try:
                idx = int(idx_str)
                if 0 <= idx < len(candidates):
                    candidates[idx]["rerank_score"] = float(score) / 10.0
            except (ValueError, TypeError):
                continue

        # Sort by rerank_score (fallback to combined_score)
        candidates.sort(
            key=lambda x: x.get("rerank_score", x.get("combined_score", 0)),
            reverse=True,
        )
    except Exception as e:
        log.warning("reranking_failed", error=str(e))

    return candidates[:top_k]


# =====================================================================
# Historical Memory Retrieval
# =====================================================================
def retrieve_similar_history(
    question: str, plant_id: Optional[str] = None, top_k: int = 3
) -> List[Dict[str, Any]]:
    """Retrieve similar past maintenance events / failure reports.

    Searches the ``maintenance_history`` and ``inverter_reports`` collections.
    """
    client = get_chroma_client()
    ef = _get_embedding_function()
    results: List[Dict[str, Any]] = []

    for coll_name in ("maintenance_history", "inverter_reports"):
        try:
            coll = client.get_collection(coll_name, embedding_function=ef)
        except Exception:
            continue

        query_params: Dict[str, Any] = {
            "query_texts": [question],
            "n_results": top_k,
            "include": ["documents", "metadatas", "distances"],
        }
        if plant_id:
            query_params["where"] = {"plant_id": plant_id}

        try:
            res = coll.query(**query_params)
            if not res["documents"] or not res["documents"][0]:
                continue

            docs = res["documents"][0]
            metadatas = res["metadatas"][0] if res["metadatas"] else [{}] * len(docs)
            distances = res["distances"][0] if res["distances"] else [1.0] * len(docs)

            for doc, meta, dist in zip(docs, metadatas, distances):
                similarity = max(0.0, 1.0 - (dist / 2.0))
                results.append({
                    "content": doc,
                    "metadata": meta,
                    "similarity": similarity,
                    "source_collection": coll_name,
                })
        except Exception as e:
            log.warning("history_retrieval_failed", collection=coll_name, error=str(e))

    results.sort(key=lambda x: x["similarity"], reverse=True)
    return results[:top_k]


# =====================================================================
# Primary Hybrid Query (public API)
# =====================================================================
def hybrid_query(
    question: str,
    plant_id: Optional[str] = None,
    top_k: int = 5,
    enable_multi_query: bool = True,
    enable_reranking: bool = True,
) -> Dict[str, Any]:
    """Full multi-stage hybrid retrieval pipeline.

    Pipeline stages:
      1. Multi-query expansion
      2. Vector retrieval from ChromaDB
      3. BM25 keyword retrieval on same corpus
      4. Hybrid score fusion
      5. Cross-encoder reranking
      6. Historical memory attachment

    Returns:
        Dict with ``results``, ``history``, ``queries_used``, and ``stats``.
    """
    weights = config.RAG_FUSION_WEIGHTS
    now = int(time.time())

    # ── Stage 1: Multi-query expansion ────────────────────────────────
    if enable_multi_query:
        queries = expand_queries(question, n=config.RAG_MULTI_QUERY_COUNT)
    else:
        queries = [question]

    # ── Stage 2: Vector retrieval from ChromaDB ───────────────────────
    client = get_chroma_client()
    ef = _get_embedding_function()

    collections = []
    for name in ("inverter_status", "inverter_reports", "knowledge_base"):
        try:
            collections.append(client.get_collection(name, embedding_function=ef))
        except Exception:
            pass

    if not collections:
        return {"results": [], "history": [], "queries_used": queries, "stats": {}}

    # Build metadata filter
    where_filter: Dict[str, Any] = {}
    if plant_id:
        where_filter["plant_id"] = plant_id

    inv_match = re.search(r"(INV_\d{3})", question, re.IGNORECASE)
    if inv_match:
        inv_id = inv_match.group(1).upper()
        if plant_id:
            where_filter = {"$and": [{"plant_id": plant_id}, {"inverter_id": inv_id}]}
        else:
            where_filter = {"inverter_id": inv_id}

    # Collect all vector results across queries and collections
    all_vector_docs: Dict[str, Dict[str, Any]] = {}  # keyed by content hash

    for q in queries:
        query_params: Dict[str, Any] = {
            "query_texts": [q],
            "n_results": config.RAG_RERANK_TOP_K,
            "include": ["documents", "metadatas", "distances"],
        }
        if where_filter:
            query_params["where"] = where_filter

        for coll in collections:
            try:
                res = coll.query(**query_params)
                if not res["documents"] or not res["documents"][0]:
                    continue

                docs = res["documents"][0]
                metadatas = res["metadatas"][0] if res["metadatas"] else [{}] * len(docs)
                distances = res["distances"][0] if res["distances"] else [1.0] * len(docs)

                for doc, meta, dist in zip(docs, metadatas, distances):
                    doc_key = hash(doc)
                    similarity = max(0.0, 1.0 - (dist / 2.0))

                    if doc_key in all_vector_docs:
                        # Keep max similarity across queries
                        if similarity > all_vector_docs[doc_key]["vector_score"]:
                            all_vector_docs[doc_key]["vector_score"] = similarity
                    else:
                        all_vector_docs[doc_key] = {
                            "content": doc,
                            "metadata": meta,
                            "vector_score": similarity,
                            "bm25_score": 0.0,
                            "source_collection": coll.name,
                        }
            except Exception as e:
                log.warning("vector_query_failed", collection=coll.name, error=str(e))

    candidates = list(all_vector_docs.values())

    # ── Stage 3: BM25 keyword retrieval ───────────────────────────────
    if candidates:
        bm25_results = bm25_search(question, candidates, top_k=len(candidates))
        bm25_map = {hash(r["content"]): r["bm25_score"] for r in bm25_results}
        for cand in candidates:
            cand["bm25_score"] = bm25_map.get(hash(cand["content"]), 0.0)

    # ── Stage 4: Hybrid score fusion ──────────────────────────────────
    for cand in candidates:
        meta = cand.get("metadata", {})
        risk_score = float(meta.get("risk_score", 0.0))
        timestamp = int(meta.get("timestamp", 0))
        recency = calculate_recency_score(timestamp, now)

        cand["combined_score"] = (
            weights["vector"] * cand["vector_score"]
            + weights["bm25"] * cand["bm25_score"]
            + weights["recency"] * recency
            + weights["risk"] * risk_score
        )
        cand["metrics"] = {
            "vector": cand["vector_score"],
            "bm25": cand["bm25_score"],
            "recency": recency,
            "risk": risk_score,
        }
        cand["retrieval_method"] = "hybrid"

    candidates.sort(key=lambda x: x["combined_score"], reverse=True)

    # ── Stage 5: Cross-encoder reranking ──────────────────────────────
    if enable_reranking and candidates:
        candidates = rerank_results(
            question, candidates, top_k=config.RAG_FINAL_TOP_K
        )
    else:
        candidates = candidates[:top_k]

    # ── Stage 6: Historical memory ────────────────────────────────────
    history = retrieve_similar_history(question, plant_id, top_k=3)

    stats = {
        "queries_expanded": len(queries),
        "total_vector_candidates": len(all_vector_docs),
        "final_results": len(candidates),
        "history_matches": len(history),
        "collections_searched": len(collections),
    }

    return {
        "results": candidates,
        "history": history,
        "queries_used": queries,
        "stats": stats,
    }


# =====================================================================
# Legacy API (backwards-compatible wrapper)
# =====================================================================
def query(
    question: str, plant_id: Optional[str] = None, top_k: int = 5
) -> List[Dict[str, Any]]:
    """Backwards-compatible query API.

    Wraps :func:`hybrid_query` and returns just the results list with
    ``content``, ``metadata``, and ``combined_score`` keys.
    """
    result = hybrid_query(
        question, plant_id, top_k,
        enable_multi_query=True,
        enable_reranking=True,
    )
    return result["results"]
