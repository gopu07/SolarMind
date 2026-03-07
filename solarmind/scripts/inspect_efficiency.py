import pandas as pd
import pathlib

replay_file = pathlib.Path(r"c:\Users\pranshu\Desktop\solar-mind\SolarMind\solarmind\data\processed\replay_predictions.parquet")
if replay_file.exists():
    df = pd.read_parquet(replay_file)
    print("Efficiency Describe:")
    if 'conversion_efficiency' in df.columns:
        print(df['conversion_efficiency'].describe())
    else:
        print("conversion_efficiency column not found")
        cols = [c for c in df.columns if 'eff' in c.lower()]
        print("Found matching columns:", cols)
        if cols:
            print(df[cols].describe())
else:
    print(f"File not found: {replay_file}")
