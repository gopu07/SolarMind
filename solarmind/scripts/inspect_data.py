import pandas as pd
import pathlib

replay_file = pathlib.Path(r"c:\Users\pranshu\Desktop\solar-mind\SolarMind\solarmind\data\processed\replay_predictions.parquet")
if replay_file.exists():
    df = pd.read_parquet(replay_file)
    print("Replay Predictions Head:")
    print(df.head())
    print("\nColumns:", df.columns.tolist())
    print("\nDescribe risk_score:")
    if 'risk_score' in df.columns:
        print(df['risk_score'].describe())
    else:
        print("risk_score column not found")
else:
    print(f"File not found: {replay_file}")
