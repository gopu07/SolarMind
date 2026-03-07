import pandas as pd
import numpy as np
from pathlib import Path
import config
import structlog

log = structlog.get_logger(__name__)

class DriftMonitor:
    def __init__(self):
        self.baseline_stats = {}
        self.baseline_file = config.ARTIFACTS_DIR / "drift_baseline.json"
        self._load_or_calc_baseline()

    def _load_or_calc_baseline(self):
        """Load baseline stats from artifact or calculate from historical features."""
        import json
        if self.baseline_file.exists():
            with open(self.baseline_file, "r") as f:
                self.baseline_stats = json.load(f)
                return

        # Calculate from features.parquet (training period)
        features_path = config.PROCESSED_DIR / "features.parquet"
        if not features_path.exists():
            log.warning("features_parquet_not_found_for_drift_baseline")
            return

        try:
            df = pd.read_parquet(features_path)
            # Use same split as train.py (last 180 days is test)
            timestamps = pd.to_datetime(df["timestamp"])
            cutoff = timestamps.max() - pd.Timedelta(days=180)
            train_df = df[timestamps < cutoff]

            signals = ["inverter_temperature", "pv1_power", "conversion_efficiency"]
            stats = {}
            for s in signals:
                if s in train_df.columns:
                    vals = train_df[s].dropna()
                    stats[s] = {
                        "mean": float(vals.mean()),
                        "std": float(vals.std()),
                        "count": int(len(vals))
                    }
            
            self.baseline_stats = stats
            config.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
            with open(self.baseline_file, "w") as f:
                json.dump(stats, f)
            log.info("drift_baseline_calculated", signals=list(stats.keys()))
        except Exception as e:
            log.error("drift_baseline_calc_failed", error=str(e))

    def calculate_drift(self, current_telemetry: list[dict]) -> dict:
        """Calculate drift (Z-score) for a window of current telemetry."""
        if not self.baseline_stats:
            return {"status": "error", "message": "Baseline not available"}

        current_df = pd.DataFrame(current_telemetry)
        results = {}
        overall_drift_score = 0
        signal_count = 0

        for signal, baseline in self.baseline_stats.items():
            if signal in current_df.columns:
                curr_vals = current_df[signal].dropna()
                if len(curr_vals) == 0:
                    continue
                
                curr_mean = float(curr_vals.mean())
                # Z-score = (Current Mean - Baseline Mean) / Baseline Std
                z_score = abs(curr_mean - baseline["mean"]) / (baseline["std"] + 1e-9)
                
                results[signal] = {
                    "baseline_mean": baseline["mean"],
                    "current_mean": curr_mean,
                    "z_score": z_score,
                    "status": "drifted" if z_score > 2.0 else "stable"
                }
                overall_drift_score += z_score
                signal_count += 1
        
        avg_drift = overall_drift_score / signal_count if signal_count > 0 else 0
        return {
            "status": "ok",
            "overall_drift_score": avg_drift,
            "drift_detected": avg_drift > 1.5,
            "signals": results,
            "timestamp": pd.Timestamp.now().isoformat()
        }

drift_monitor = DriftMonitor()
