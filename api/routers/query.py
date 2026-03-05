"""
Layer 7 — Query router.
"""

import time
from fastapi import APIRouter, Depends
import structlog

from api.auth import get_current_user
from api.schemas.models import QueryRequest, QueryResponse, Citation
from rag.retriever import query as rag_query
import config

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
            client = openai.OpenAI(api_key=config.OPENAI_API_KEY)
            
            grounding_instruction = (
                "Answer only using the retrieved documents provided below. "
                "Cite the `inverter_id` and `timestamp` of every claim. "
                "If the documents do not contain sufficient information to answer the question, say so explicitly rather than speculating.\n\n"
            )
            
            prompt = grounding_instruction + "\n\n".join(context_chunks) + f"\n\nQuestion: {req.question}"
            
            resp = client.chat.completions.create(
                model=config.LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0
            )
            answer = resp.choices[0].message.content
        except Exception as e:
            log.warning("llm_qa_failed", error=str(e))
            answer = f"Error calling LLM: {str(e)}"
            
    latency_ms = (time.perf_counter() - start_time) * 1000.0
    
    return QueryResponse(
        answer=answer,
        citations=citations,
        latency_ms=latency_ms
    )
