"""
Layer 0 — Configuration and Environment.

Single source of truth for all paths, constants, thresholds,
and environment variables used across the SolarMind platform.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict

from dotenv import load_dotenv

# ── Load .env if present ──────────────────────────────────────────────
load_dotenv()

# ── Project root ──────────────────────────────────────────────────────
PROJECT_ROOT: Path = Path(__file__).resolve().parent

# ── Data directories ──────────────────────────────────────────────────
DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
PROCESSED_DIR: Path = DATA_DIR / "processed"
MAPPINGS_DIR: Path = DATA_DIR / "mappings"

# ── Key data files ────────────────────────────────────────────────────
MAC_MAPPING_FILE: Path = MAPPINGS_DIR / "mac_mapping.csv"
EVENTS_FILE: Path = MAPPINGS_DIR / "events.csv"

# ── Model artifacts ───────────────────────────────────────────────────
ARTIFACTS_DIR: Path = PROJECT_ROOT / "models" / "artifacts"

# ── Labelling & prediction constants ─────────────────────────────────
LABEL_WINDOW_DAYS: int = 10
PREDICTION_HORIZON_DAYS: int = 7
TELEMETRY_INTERVAL_MINUTES: int = 15
SAMPLES_PER_DAY: int = 96  # 24h * 60min / 15min

# ── Risk thresholds ──────────────────────────────────────────────────
RISK_THRESHOLDS: Dict[str, float] = {
    "LOW": 0.3,
    "MEDIUM": 0.6,
    "HIGH": 0.8,
}

# ── LLM / GenAI ──────────────────────────────────────────────────────
LLM_MODEL: str = os.getenv("LLM_MODEL", "openai/gpt-4o-mini")
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL: str = os.getenv("OPENAI_BASE_URL", "")

# ── RAG / ChromaDB ───────────────────────────────────────────────────
CHROMA_PERSIST_DIR: Path = PROJECT_ROOT / "rag" / "chroma_store"
KNOWLEDGE_BASE_DIR: Path = PROJECT_ROOT / "rag" / "knowledge_base"

# ── RAG Advanced Pipeline ────────────────────────────────────────────
RAG_FUSION_WEIGHTS: Dict[str, float] = {
    "vector": 0.40,
    "bm25": 0.25,
    "recency": 0.15,
    "risk": 0.20,
}
RAG_MULTI_QUERY_COUNT: int = 3          # semantic query expansions
RAG_RERANK_TOP_K: int = 15             # candidates fed to reranker
RAG_FINAL_TOP_K: int = 5              # results returned after reranking

# ── API / Auth ───────────────────────────────────────────────────────
JWT_SECRET: str = os.getenv("JWT_SECRET", "change-me-in-production")
JWT_ALGORITHM: str = "HS256"
JWT_EXPIRE_MINUTES: int = 60
DEV_USERNAME: str = os.getenv("DEV_USERNAME", "admin")
DEV_PASSWORD: str = os.getenv("DEV_PASSWORD", "admin")

# ── External services ────────────────────────────────────────────────
CMMS_ENDPOINT: str = os.getenv("CMMS_ENDPOINT", "http://localhost:8001/tickets")
SLACK_WEBHOOK_URL: str = os.getenv("SLACK_WEBHOOK_URL", "")
REDIS_URL: str = os.getenv("REDIS_URL", "")

# ── Inference heuristic thresholds (for hybrid labelling) ────────────
EFFICIENCY_DROP_THRESHOLD: float = 0.15          # 15% below 30-day median
TEMPERATURE_PERCENTILE: float = 0.99             # 30-day 99th percentile
STRING_MISMATCH_CV_THRESHOLD: float = 0.25
CONSECUTIVE_HOURS_THRESHOLD: int = 4             # Reduced to 4 consecutive hours to get more labels
POWER_DROP_THRESHOLD: float = 0.20               # 20% sustained drop
NIGHTTIME_POWER_THRESHOLD: float = 10.0          # watts — below this is night

# ── New Feature Engineering Column Setup ─────────────────────────────
# This replaces the hardcoded CANONICAL_COLUMNS in ingest_raw.py
CANONICAL_COLUMNS: list[str] = [
    "timestamp",
    # PV 1
    "pv1_current", "pv1_voltage", "pv1_power",
    # PV 2 (Dual MPPT)
    "pv2_current", "pv2_voltage", "pv2_power",
    # Inverter & Grid
    "inverter_temperature", "meter_active_power",
    "meter_pf", "meter_freq", 
    "meter_v_r", "meter_v_y", "meter_v_b",
    # Internal Status
    "inverter_alarm_code", "inverter_op_state", "inverter_limit_percent",
] + [f"smu_string{i}" for i in range(1, 25)] + [f"inv_string{i}" for i in range(1, 25)]

# ── Logging ──────────────────────────────────────────────────────────
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")


def risk_level_from_score(score: float) -> str:
    """Derive a risk level string from a numeric risk score.

    Args:
        score: Calibrated probability in [0, 1].

    Returns:
        One of ``"LOW"``, ``"MEDIUM"``, ``"HIGH"``, or ``"CRITICAL"``.
    """
    if score < RISK_THRESHOLDS["LOW"]:
        return "LOW"
    if score < RISK_THRESHOLDS["MEDIUM"]:
        return "MEDIUM"
    if score < RISK_THRESHOLDS["HIGH"]:
        return "HIGH"
    return "CRITICAL"
