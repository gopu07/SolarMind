import pandas as pd
import pathlib

replay_file = pathlib.Path(r"c:\Users\pranshu\Desktop\solar-mind\SolarMind\solarmind\data\processed\replay_predictions.parquet")
if replay_file.exists():
    df = pd.read_parquet(replay_file)
    print("Columns:", df.columns.tolist())
    print("\nSample values for power and temperature:")
    power_cols = [c for c in df.columns if 'power' in c.lower()]
    temp_cols = [c for c in df.columns if 'temp' in c.lower()]
    print("Power columns:", power_cols)
    print("Temp columns:", temp_cols)
    if power_cols or temp_cols:
        print(df[power_cols + temp_cols].head())
else:
    print(f"File not found: {replay_file}")
