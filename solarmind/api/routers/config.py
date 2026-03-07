from fastapi import APIRouter
from app_config.settings import get_masked_settings

router = APIRouter(prefix="/config", tags=["Configuration"])

@router.get("/status")
async def get_config_status():
    """Returnmasked environment configuration for health monitoring."""
    return get_masked_settings()
