"""
Layer 7 — Health router.
"""

from fastapi import APIRouter, Response, status
import structlog
from typing import Dict

from api.schemas.models import HealthResponse
import config
from rag.ingest import get_chroma_client

log = structlog.get_logger(__name__)

router = APIRouter(
    prefix="/health",
    tags=["Health"],
)

@router.get("", response_model=HealthResponse)
async def check_health(response: Response):
    """Deep health check of all required subsystems."""
    checks: Dict[str, str] = {}
    
    # Check 1: Model artifacts
    model_path = config.ARTIFACTS_DIR / "model.pkl"
    if model_path.exists():
        checks["models"] = "ok"
    else:
        checks["models"] = "missing"
        
    # Check 2: Processed Parquet data
    parquet_path = config.PROCESSED_DIR / "master_labelled.parquet"
    if parquet_path.exists():
        checks["data"] = "ok"
    else:
        checks["data"] = "missing"
        
    # Check 3: ChromaDB
    try:
        client = get_chroma_client()
        client.heartbeat()
        checks["chromadb"] = "ok"
    except Exception:
        checks["chromadb"] = "down"
        
    # Check 4: LLM API Configuration
    if config.OPENAI_API_KEY:
        checks["llm"] = "configured"
    else:
        checks["llm"] = "unconfigured"
        
    # Aggregate status
    is_down = checks["models"] == "missing" or checks["data"] == "missing"
    is_degraded = checks["chromadb"] == "down" or checks["llm"] == "unconfigured"
    
    if is_down:
        overall_status = "down"
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    elif is_degraded:
        overall_status = "degraded"
        response.status_code = status.HTTP_207_MULTI_STATUS
    else:
        overall_status = "ok"
        response.status_code = status.HTTP_200_OK
        
    log.info("health_check", status=overall_status, checks=checks)
    
    return HealthResponse(
        status=overall_status,
        checks=checks
    )
