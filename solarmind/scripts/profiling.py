import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import PROCESSED_DIR
import pandas as pd

def main():
    file_path = PROCESSED_DIR / "master_labelled.parquet"
    if not file_path.exists():
        print("Data not found.")
        return

    print("Loading data...")
    df = pd.read_parquet(file_path)
    
    print("\n" + "="*50)
    print("DATA PROFILING REPORT")
    print("="*50)
    
    print(f"Total Rows: {len(df):,}")
    print(f"Total Columns: {len(df.columns)}")
    print(f"Total Inverters: {df['inverter_id'].nunique()}")
    print("\nInverter List:", sorted(df['inverter_id'].unique().tolist())[:10], "... (truncated)")
    
    # Time span
    print(f"\nTime Span: {df['timestamp'].min()} to {df['timestamp'].max()}")
    
    # Target stats
    positives = df['label'].sum()
    print(f"\nLabel 'will_fail_24h' Total Positives: {positives:,} ({positives/len(df)*100:.2f}%)")
    
    # Missing value stats for key columns
    print("\nMissing Values (Key Columns):")
    key_cols = ['pv1_power', 'pv1_current', 'pv1_voltage', 'inverter_temperature', 'meter_active_power']
    for col in key_cols:
        if col in df.columns:
            missing = df[col].isna().sum()
            print(f"  {col}: {missing:,} ({missing/len(df)*100:.2f}%)")
            
    print("\nSensor Summaries:")
    for col in key_cols:
        if col in df.columns:
            desc = df[col].describe(percentiles=[0.01, 0.5, 0.99])
            print(f"  {col:20} Min: {desc['min']:>8.2f} | 50%: {desc['50%']:>8.2f} | 99%: {desc['99%']:>8.2f} | Max: {desc['max']:>8.2f}")

    print("\n" + "="*50)

if __name__ == "__main__":
    main()
