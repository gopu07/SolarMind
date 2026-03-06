import pandas as pd
from pathlib import Path
import sys

# ── Ensure project root is on sys.path ──────────────────────────────
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import config

def main():
    p = config.PROCESSED_DIR / "master_labelled.parquet"
    if not p.exists():
        print(f"File not found: {p}")
        return

    df = pd.read_parquet(p)
    print(f"Total rows: {len(df)}")
    print(f"Unique inverters: {df['inverter_id'].nunique()}")
    print(f"Inverter IDs: {df['inverter_id'].unique().tolist()}")
    print("\nColumns:")
    print(df.columns.tolist())
    print("\nSample Data:")
    print(df.head())
    
    if "label" in df.columns:
        print("\nLabel statistics:")
        print(df["label"].value_counts())

if __name__ == "__main__":
    main()
