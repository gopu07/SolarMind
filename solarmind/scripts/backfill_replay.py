"""Backfill missing inverters in replay_predictions.parquet."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np
from config import PROCESSED_DIR

replay_path = PROCESSED_DIR / "replay_predictions.parquet"
master_path = PROCESSED_DIR / "master_labelled.parquet"

replay = pd.read_parquet(replay_path)
master = pd.read_parquet(master_path, columns=["inverter_id", "plant_id"])
master = master.drop_duplicates(subset=["inverter_id"])

replay_ids = set(replay["inverter_id"].unique())
master_ids = set(master["inverter_id"].unique())
missing = master_ids - replay_ids

print(f"Replay has {len(replay_ids)} inverters, master has {len(master_ids)}")
print(f"Missing: {sorted(missing)}")

if missing:
    timestamps = replay["timestamp"].unique()
    sample_ts = timestamps[::10] if len(timestamps) > 100 else timestamps
    
    fill_rows = []
    for inv_id in missing:
        plant_id = master[master["inverter_id"] == inv_id]["plant_id"].iloc[0]
        for ts in sample_ts:
            fill_rows.append({
                "timestamp": ts,
                "inverter_id": inv_id,
                "plant_id": plant_id,
                "risk_score": 0.0,
                "inverter_temperature": 0.0,
                "pv1_power": 0.0,
                "conversion_efficiency": 0.0,
                "top_shap_features": "[]",
                "label": 0
            })
    
    fill_df = pd.DataFrame(fill_rows)
    for c in replay.columns:
        if c not in fill_df.columns:
            fill_df[c] = np.nan
    
    patched = pd.concat([replay, fill_df[replay.columns]], ignore_index=True)
    patched.to_parquet(replay_path, index=False, engine="pyarrow")
    
    final_count = patched["inverter_id"].nunique()
    print(f"Backfilled {len(fill_rows)} rows for {len(missing)} inverters")
    print(f"Final inverter count: {final_count}")
    assert 25 <= final_count <= 40
else:
    print("No backfill needed.")
