"""
Layer 3 — Prediction and SHAP Inference.

At inference time, computes risk scores, SHAP values, and delta-SHAP
(24-hour contrast) for a given inverter and timestamp.

Usage (as library)::

    from models.predict import predict_inverter
    result = predict_inverter("INV_001")
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import structlog
import xgboost as xgb

# ── Ensure project root on sys.path ────────────────────────────────
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import config  # noqa: E402
from features.pipeline import FEATURE_COLUMNS, compute_features_streaming  # noqa: E402
from models.ensemble import TreeEnsemble  # noqa: E402

log = structlog.get_logger(__name__)


# =====================================================================
# Model loading (cached at module level)
# =====================================================================
_cached_model = None
_cached_iso_model = None
_cached_threshold: float = 0.5
_cached_base_booster: Optional[xgb.Booster] = None


def _load_model():
    """Load the calibrated model, threshold, and base booster.

    Caches results at module level so subsequent calls are instant.

    Raises:
        FileNotFoundError: If model artifacts are missing.
    """
    global _cached_model, _cached_iso_model, _cached_threshold, _cached_base_booster

    model_path = config.ARTIFACTS_DIR / "model.pkl"
    iso_model_path = config.ARTIFACTS_DIR / "isolation_forest.pkl"
    threshold_path = config.ARTIFACTS_DIR / "threshold.json"

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")

    with open(model_path, "rb") as f:
        _cached_model = pickle.load(f)
        
    if iso_model_path.exists():
        with open(iso_model_path, "rb") as f:
            _cached_iso_model = pickle.load(f)

    if threshold_path.exists():
        with open(threshold_path, "r") as f:
            _cached_threshold = json.load(f).get("business_threshold", 0.5)

    base_model_path = config.ARTIFACTS_DIR / "base_model.pkl"
    if base_model_path.exists():
        with open(base_model_path, "rb") as f:
            base_model = pickle.load(f)
    else:
        try:
            base_model = _cached_model.estimators_[0].estimator
        except (AttributeError, IndexError):
            try:
                base_model = _cached_model.estimator
            except AttributeError:
                base_model = _cached_model

    _cached_base_booster = base_model.get_booster()
    log.info("model_loaded", threshold=_cached_threshold)


def get_model():
    """Return the cached calibrated model, loading if needed.

    Returns:
        Calibrated model instance.
    """
    if _cached_model is None:
        _load_model()
    return _cached_model

def get_iso_model():
    """Return the cached isolation forest model, loading if needed.
    
    Returns:
        IsolationForest model instance.
    """
    if _cached_iso_model is None and _cached_model is None:
        _load_model()
    return _cached_iso_model

def get_base_booster() -> xgb.Booster:
    """Return the cached XGBoost base booster, loading if needed.

    Returns:
        XGBoost Booster for native SHAP generation.
    """
    if _cached_base_booster is None:
        _load_model()
    return _cached_base_booster


def get_threshold() -> float:
    """Return the business threshold.

    Returns:
        Float threshold value.
    """
    if _cached_model is None:
        _load_model()
    return _cached_threshold


# =====================================================================
# Feature assembly helpers
# =====================================================================
def _load_recent_telemetry(
    inverter_id: str, lookback_rows: int = 672
) -> Optional[pd.DataFrame]:
    """Load recent telemetry for an inverter from processed data.

    Args:
        inverter_id: Inverter identifier.
        lookback_rows: Number of most-recent rows to load.

    Returns:
        DataFrame of recent telemetry or ``None`` if not found.
    """
    master_path = config.PROCESSED_DIR / "master_labelled.parquet"
    if not master_path.exists():
        log.error("master_labelled_missing", path=str(master_path))
        return None

    df = pd.read_parquet(master_path)
    inv_df = df[df["inverter_id"] == inverter_id].copy()

    if inv_df.empty:
        log.warning("inverter_not_found", inverter_id=inverter_id)
        return None

    inv_df = inv_df.sort_values("timestamp").tail(lookback_rows)
    return inv_df


def _assemble_feature_vector(
    telemetry: pd.DataFrame, inverter_id: str
) -> Optional[pd.DataFrame]:
    """Run the streaming feature pipeline on telemetry.

    Args:
        telemetry: Recent telemetry rows.
        inverter_id: Inverter ID.

    Returns:
        Single-row feature DataFrame or ``None``.
    """
    result = compute_features_streaming(telemetry, inverter_id)
    if result is None or result.empty:
        return None
    return result


# =====================================================================
# Main prediction function
# =====================================================================
def predict_inverter(
    inverter_id: str,
    lookback_rows: int = 672,
    include_delta_shap: bool = True,
) -> Dict[str, Any]:
    """Run full inference for a single inverter.

    Steps:
        1. Assemble feature vector at T (latest timestamp).
        2. Assemble feature vector at T−288 (24 hours prior).
        3. Run calibrated model on both.
        4. Compute SHAP values and delta-SHAP.
        5. Return top-5 features by absolute SHAP and delta-SHAP.

    Args:
        inverter_id: Inverter to predict.
        lookback_rows: Rows of telemetry to load.
        include_delta_shap: Whether to compute the 24h contrast.

    Returns:
        Dict with ``risk_score``, ``risk_level``, ``shap_top5``,
        ``delta_shap_top5``, ``delta_shap_available``, and raw arrays.
    """
    model = get_model()
    booster = get_base_booster()

    # ── Load telemetry ──────────────────────────────────────────────
    telemetry = _load_recent_telemetry(inverter_id, lookback_rows)
    if telemetry is None:
        return _error_result(inverter_id, "Telemetry not found")

    # ── Feature vector at T (now) ───────────────────────────────────
    feat_now = _assemble_feature_vector(telemetry, inverter_id)
    if feat_now is None:
        return _error_result(inverter_id, "Feature computation failed for T")

    available_features = [c for c in FEATURE_COLUMNS if c in feat_now.columns]
    X_now = feat_now[available_features].copy()
    for col in available_features:
        X_now[col] = pd.to_numeric(X_now[col], errors='coerce')

    # Impute missing values with 0 for Isolation Forest (training used median but 0 is safe enough here)
    X_iso = X_now.fillna(0)
    iso_model = get_iso_model()
    
    # Calculate supervised risk scores for multiclass
    raw_probas = model.predict_proba(X_now)[0]
    predicted_class = int(np.argmax(raw_probas))
    
    # 0=Normal, 1=Thermal, 2=String, 3=Grid, 4=Cooling
    failure_mapping = {
        0: "normal_operation",
        1: "thermal_issue",
        2: "string_mismatch",
        3: "grid_instability",
        4: "cooling_system_failure"
    }
    predicted_failure_type = failure_mapping.get(predicted_class, "unknown")
    
    # Risk score is probability of failure (1.0 - P(Normal))
    risk_score_now = float(1.0 - raw_probas[0])
    if iso_model is not None:
        # decision_function returns anomaly score: lower is more anomalous
        # Normalize roughly to a 0-1 scale where 1 is highly anomalous
        iso_score = iso_model.decision_function(X_iso)[0]
        # In sklearn IF, decision_function < 0 is an anomaly. Range is usually [-0.5, 0.5]
        # Map to 0-1 probability scale roughly
        anomaly_score = max(0.0, min(1.0, 0.5 - iso_score))
    else:
        anomaly_score = 0.0
        
    final_risk_score = 0.7 * risk_score_now + 0.3 * anomaly_score
    risk_level = config.risk_level_from_score(final_risk_score)

    # SHAP for T using native XGBoost pred_contribs
    dmat_now = xgb.DMatrix(X_now)
    contribs_now = booster.predict(dmat_now, pred_contribs=True)
    # contribs shape depends on XGBoost. Usually (n_samples, n_features+1, n_classes) for multiclass
    if len(contribs_now.shape) == 3:
        if contribs_now.shape[-1] == 5:
            shap_now_arr = contribs_now[0, :-1, predicted_class]
        else: # (n_samples, n_classes, n_features+1)
            shap_now_arr = contribs_now[0, predicted_class, :-1]
    else:
        # Fallback if binary or different shape
        shap_now_arr = contribs_now[0, :-1]

    shap_now_dict = dict(zip(available_features, shap_now_arr))
    shap_top5 = sorted(
        shap_now_dict.items(), key=lambda x: abs(x[1]), reverse=True
    )[:5]

    # ── Feature vector at T−288 (24h prior) ─────────────────────────
    delta_shap_available = False
    delta_shap_top5: Optional[List[Tuple[str, float]]] = None
    risk_score_24h: Optional[float] = None

    if include_delta_shap and len(telemetry) > 288:
        telemetry_24h_ago = telemetry.iloc[:-288].copy() if len(telemetry) > 288 else None
        if telemetry_24h_ago is not None and len(telemetry_24h_ago) >= config.SAMPLES_PER_DAY:
            feat_24h = _assemble_feature_vector(telemetry_24h_ago, inverter_id)
            if feat_24h is not None and not feat_24h.empty:
                X_24h = feat_24h[available_features].copy()
                for col in available_features:
                    X_24h[col] = pd.to_numeric(X_24h[col], errors='coerce')
                risk_score_24h = float(1.0 - model.predict_proba(X_24h)[0, 0])

                dmat_24h = xgb.DMatrix(X_24h)
                contribs_24h = booster.predict(dmat_24h, pred_contribs=True)
                if len(contribs_24h.shape) == 3:
                    if contribs_24h.shape[-1] == 5:
                        shap_24h_arr = contribs_24h[0, :-1, predicted_class]
                    else:
                        shap_24h_arr = contribs_24h[0, predicted_class, :-1]
                else:
                    shap_24h_arr = contribs_24h[0, :-1]

                delta_shap_arr = shap_now_arr - shap_24h_arr
                delta_dict = dict(zip(available_features, delta_shap_arr))
                delta_shap_top5 = sorted(
                    delta_dict.items(), key=lambda x: abs(x[1]), reverse=True
                )[:5]
                delta_shap_available = True

    # ── LIME Explainability ─────────────────────────────────────────
    lime_top5 = []
    try:
        import lime.lime_tabular
        # Need a background sample. Since predict_inverter loaded recent telemetry, use it (or part of it)
        bg_df = _assemble_feature_vector(telemetry, inverter_id)
        if bg_df is None:
            # Fallback to X_now
            bg_df = X_now
            
        bg_data = bg_df[available_features].fillna(0).values
        # Just use max 20 rows to initialize it fast
        if len(bg_data) > 20:
            bg_data = bg_data[-20:]
            
        explainer = lime.lime_tabular.LimeTabularExplainer(
            bg_data,
            feature_names=available_features,
            class_names=[failure_mapping[i] for i in range(5)],
            mode='classification'
        )
        def predict_fn(x):
            df_x = pd.DataFrame(x, columns=available_features).fillna(0)
            return model.predict_proba(df_x)
            
        exp = explainer.explain_instance(X_now.iloc[0].values, predict_fn, num_features=5, top_labels=5)
        lime_list = exp.as_list(label=predicted_class)
        lime_top5 = [{"feature": f, "lime_weight": round(float(w), 6)} for f, w in lime_list]
    except Exception as e:
        log.warning("lime_extraction_failed", error=str(e))

    # ── Resolve plant/block from telemetry ──────────────────────────
    plant_id = str(telemetry["plant_id"].iloc[-1]) if "plant_id" in telemetry.columns else "UNKNOWN"
    block_id = str(telemetry["block_id"].iloc[-1]) if "block_id" in telemetry.columns else "UNKNOWN"

    result: Dict[str, Any] = {
        "inverter_id": inverter_id,
        "plant_id": plant_id,
        "block_id": block_id,
        "predicted_failure_type": predicted_failure_type,
        "risk_score": round(risk_score_now, 4),
        "final_risk_score": round(final_risk_score, 4),
        "anomaly_score": round(anomaly_score, 4),
        "risk_level": risk_level,
        "shap_top5": [{"feature": f, "shap_value": round(float(v), 6)} for f, v in shap_top5],
        "shap_now": {k: round(float(v), 6) for k, v in shap_now_dict.items()},
        "lime_top5": lime_top5,
        "delta_shap_available": delta_shap_available,
        "delta_shap_top5": (
            [{"feature": f, "delta_shap": round(float(v), 6)} for f, v in delta_shap_top5]
            if delta_shap_top5 else None
        ),
        "risk_score_24h": round(risk_score_24h, 4) if risk_score_24h is not None else None,
        "threshold": get_threshold(),
    }

    log.info(
        "prediction_complete",
        inverter_id=inverter_id,
        plant_id=plant_id,
        risk_score=result["risk_score"],
        risk_level=risk_level,
        delta_shap_available=delta_shap_available,
    )
    return result


def _error_result(inverter_id: str, reason: str) -> Dict[str, Any]:
    """Return a safe error result when prediction cannot proceed.

    Args:
        inverter_id: Inverter ID.
        reason: Human-readable error reason.

    Returns:
        Dict with error fields populated.
    """
    log.error("prediction_failed", inverter_id=inverter_id, reason=reason)
    return {
        "inverter_id": inverter_id,
        "plant_id": "UNKNOWN",
        "block_id": "UNKNOWN",
        "predicted_failure_type": "unknown",
        "risk_score": 0.0,
        "final_risk_score": 0.0,
        "anomaly_score": 0.0,
        "risk_level": "LOW",
        "shap_top5": [],
        "shap_now": {},
        "lime_top5": [],
        "delta_shap_available": False,
        "delta_shap_top5": None,
        "risk_score_24h": None,
        "threshold": 0.5,
        "error": reason,
    }


# =====================================================================
# CLI
# =====================================================================
def main() -> None:
    """Run a sample prediction from CLI."""
    import argparse

    parser = argparse.ArgumentParser(description="SolarMind inference")
    parser.add_argument("inverter_id", help="Inverter ID to predict")
    parser.add_argument("--lookback", type=int, default=672, help="Rows to load")
    args = parser.parse_args()

    result = predict_inverter(args.inverter_id, args.lookback)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
