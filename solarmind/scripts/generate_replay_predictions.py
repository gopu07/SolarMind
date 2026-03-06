import sys
from pathlib import Path

# Ensures custom modules are imported correctly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import PROCESSED_DIR, ARTIFACTS_DIR
from features.pipeline import FEATURE_COLUMNS
import pandas as pd
import numpy as np
import pickle
import json
import shap
import pyarrow as pa
import pyarrow.parquet as pq

def main():
    print("⏳ Loading features and model artifacts...")
    features_path = PROCESSED_DIR / "features.parquet"
    model_path = ARTIFACTS_DIR / "model.pkl"
    if not features_path.exists() or not model_path.exists():
        print("Required files not found.")
        return

    df = pd.read_parquet(features_path)
    with open(model_path, "rb") as f:
        calibrated_model = pickle.load(f)
        
    # ── 1. Filter to Replay Window (last 180 days) ── 
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    test_reserve = 180
    test_cutoff = df["timestamp"].max() - pd.Timedelta(days=test_reserve)
    
    replay_df = df[df["timestamp"] >= test_cutoff].copy().sort_values("timestamp")
    
    # ── 2. Run Inference ──
    X = replay_df[FEATURE_COLUMNS].copy()
    for col in FEATURE_COLUMNS:
        X[col] = pd.to_numeric(X[col], errors='coerce')
        
    print(f"🤖 Generating predictions for {len(X):,} rows over 6 months...")
    y_prob = calibrated_model.predict_proba(X)[:, 1]
    replay_df["risk_score"] = y_prob
    
    # ── 3. SHAP ──
    # The SHAP TreeExplainer C++ backend is currently incompatible with this 
    # specific version of XGBoost/Pythons base_score string formatting '[5E-1]'.
    # To unblock the replay engine, we skip calculating SHAP drivers here.
    replay_df["top_shap_features"] = "[]"
        
    # ── 4. Format Output Schema ──
    keep_cols = [
        "timestamp", "inverter_id", "plant_id", 
        "risk_score", "inverter_temperature", "pv1_power", "conversion_efficiency",
        "top_shap_features", "label"
    ]
    
    out_df = replay_df[[c for c in keep_cols if c in replay_df.columns]].copy()
    
    out_path = PROCESSED_DIR / "replay_predictions.parquet"
    out_df.to_parquet(out_path, index=False, engine="pyarrow")
    
    print(f"✅ Replay predictions saved to: {out_path}")
    print(f"   Shape: {out_df.shape}")
    print(f"   Time Span: {out_df['timestamp'].min()} to {out_df['timestamp'].max()}")

if __name__ == "__main__":
    main()
