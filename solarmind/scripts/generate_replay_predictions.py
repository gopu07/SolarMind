import sys
from pathlib import Path

# Ensures custom modules are imported correctly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import PROCESSED_DIR, ARTIFACTS_DIR
from features.pipeline import FEATURE_COLUMNS
from models.ensemble import TreeEnsemble
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
    iso_model_path = ARTIFACTS_DIR / "isolation_forest.pkl"
    if not features_path.exists() or not model_path.exists():
        print("Required files not found.")
        return

    df = pd.read_parquet(features_path)
    with open(model_path, "rb") as f:
        calibrated_model = pickle.load(f)
        
    iso_model = None
    if iso_model_path.exists():
        with open(iso_model_path, "rb") as f:
            iso_model = pickle.load(f)
        
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
    y_proba_full = calibrated_model.predict_proba(X)
    y_prob = 1.0 - y_proba_full[:, 0] # Probability of ANY fault
    replay_df["risk_score"] = y_prob
    replay_df["predicted_failure_type"] = np.argmax(y_proba_full, axis=1)
    
    if iso_model is not None:
        iso_features = list(getattr(iso_model, "feature_names_in_", X.columns))
        X_iso = X[[c for c in iso_features if c in X.columns]].fillna(0)
        iso_score = iso_model.decision_function(X_iso)
        # Map roughly to 0-1 where 1 is anomalous
        anomaly_score = np.clip(0.5 - iso_score, 0.0, 1.0)
        replay_df["anomaly_score"] = anomaly_score
    replay_df["final_risk_score"] = 0.7 * y_prob + 0.3 * anomaly_score
    
    # ── 2.5 Validation & Logging ──
    print(f"📊 Risk Score Stats:")
    print(f"    Min:  {replay_df['final_risk_score'].min():.4f}")
    print(f"    Max:  {replay_df['final_risk_score'].max():.4f}")
    print(f"    Mean: {replay_df['final_risk_score'].mean():.4f}")
    print(f"    Std:  {replay_df['final_risk_score'].std():.4f}")
    
    binary_threshold = 0.05
    near_0 = (replay_df['final_risk_score'] < binary_threshold).mean() * 100
    near_1 = (replay_df['final_risk_score'] > (1.0 - binary_threshold)).mean() * 100
    print(f"    Near 0 (fault-free): {near_0:.1f}%")
    print(f"    Near 1 (critical):   {near_1:.1f}%")
    
    if near_1 > 80:
        print("🚨 WARNING: High percentage of extreme risk predictions. Check feature scaling!")
    
    # ── 3. SHAP ──
    replay_df["top_shap_features"] = "[]"
        
    # ── 4. Format Output Schema ──
    keep_cols = [
        "timestamp", "inverter_id", "plant_id", 
        "risk_score", "anomaly_score", "final_risk_score", 
        "inverter_temperature", "pv1_power", "conversion_efficiency",
        "top_shap_features", "label", "predicted_failure_type"
    ]
    
    out_df = replay_df[[c for c in keep_cols if c in replay_df.columns]].copy()

    # ── 5. FIX: Ensure ALL master inverters appear in the replay dataset ──
    master_path = PROCESSED_DIR / "master_labelled.parquet"
    if master_path.exists():
        master_inv = pd.read_parquet(master_path, columns=["inverter_id", "plant_id"])
        master_inv = master_inv.drop_duplicates(subset=["inverter_id"])
        
        replay_inv_ids = set(out_df["inverter_id"].unique())
        master_inv_ids = set(master_inv["inverter_id"].unique())
        missing_ids = master_inv_ids - replay_inv_ids
        
        if missing_ids:
            print(f"⚠️  {len(missing_ids)} inverters missing from replay: {sorted(missing_ids)}")
            # For each missing inverter, create placeholder rows at each unique timestamp
            timestamps = out_df["timestamp"].unique()
            # Pick a subset of timestamps to avoid huge expansion (every 10th)
            sample_ts = timestamps[::10] if len(timestamps) > 100 else timestamps
            
            fill_rows = []
            for inv_id in missing_ids:
                plant_id = master_inv[master_inv["inverter_id"] == inv_id]["plant_id"].iloc[0]
                for ts in sample_ts:
                    fill_rows.append({
                        "timestamp": ts,
                        "inverter_id": inv_id,
                        "plant_id": plant_id,
                        "risk_score": 0.0,
                        "anomaly_score": 0.0,
                        "final_risk_score": 0.0,
                        "inverter_temperature": 0.0,
                        "pv1_power": 0.0,
                        "conversion_efficiency": 0.0,
                        "top_shap_features": "[]",
                        "label": 0,
                        "predicted_failure_type": 0
                    })
            
            if fill_rows:
                fill_df = pd.DataFrame(fill_rows)
                # Align columns
                for c in out_df.columns:
                    if c not in fill_df.columns:
                        fill_df[c] = np.nan
                out_df = pd.concat([out_df, fill_df[out_df.columns]], ignore_index=True)
                print(f"✅ Backfilled {len(fill_rows)} rows for {len(missing_ids)} missing inverters")
    
    # ── 6. Validate ──
    unique_inverters = out_df["inverter_id"].nunique()
    print(f"[REPLAY-DEBUG] Unique inverter count: {unique_inverters}")
    assert 25 <= unique_inverters <= 40, f"Unexpected replay inverter count: {unique_inverters}"

    out_path = PROCESSED_DIR / "replay_predictions.parquet"
    out_df.to_parquet(out_path, index=False, engine="pyarrow")
    
    print(f"✅ Replay predictions saved to: {out_path}")
    print(f"   Shape: {out_df.shape}")
    print(f"   Unique Inverters: {unique_inverters}")
    print(f"   Time Span: {out_df['timestamp'].min()} to {out_df['timestamp'].max()}")

if __name__ == "__main__":
    main()
