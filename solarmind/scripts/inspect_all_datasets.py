import pandas as pd
import pathlib

processed_dir = pathlib.Path(r"c:\Users\pranshu\Desktop\solar-mind\SolarMind\solarmind\data\processed")
files = ["features.parquet", "master_telemetry.parquet", "master_labelled.parquet"]

for f in files:
    path = processed_dir / f
    if path.exists():
        print(f"\n--- {f} ---")
        df = pd.read_parquet(path)
        print("Rows:", len(df))
        eff_cols = [c for c in df.columns if 'eff' in c.lower()]
        print("Efficiency columns:", eff_cols)
        if eff_cols:
            print(df[eff_cols].describe())
    else:
        print(f"\n{f} not found")
