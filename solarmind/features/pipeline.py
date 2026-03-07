"""
Layer 2 — Feature Engineering Pipeline.

Stateless, reproducible feature computation module. Works in two modes:

* **Batch mode** — operates on the full master DataFrame (for training).
* **Streaming mode** — operates on a single inverter's recent window
  (for real-time inference).

Both modes produce identical feature columns defined in
:data:`FEATURE_COLUMNS`.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import structlog

# ── Ensure project root on sys.path ────────────────────────────────
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import config  # noqa: E402

log = structlog.get_logger(__name__)


# =====================================================================
# Canonical feature list  — imported by every downstream module
# =====================================================================
FEATURE_COLUMNS: List[str] = [
    # Cyclical time
    "hour_sin", "hour_cos", "month_sin", "month_cos",
    "day_of_year_sin", "day_of_year_cos",
    "daylight_hours_indicator",
    
    # Physics & Thermal
    "conversion_efficiency",
    "thermal_gradient",
    "temp_power_interaction",
    "temp_efficiency_divergence",
    "temperature_mean_6h",
    "temp_rolling_std_12h",
    "temp_rolling_mean_24h",
    "temp_rolling_min_24h",
    "temp_rolling_max_24h",
    "temp_vs_30d_percentile",
    
    # Grid Stability & MPPT
    "mppt_imbalance_ratio",
    "mppt_power_ratio_24h",
    "grid_voltage_disturbances_24h",
    "frequency_deviation",
    "power_efficiency_ratio",
    "efficiency_volatility_24h",
    "efficiency_drop_rate",
    
    # String anomaly
    "string_mismatch_std",
    "string_mismatch_cv",
    "string_max_deviation",
    "string_current_variance",
    
    # Plant Context (Relative Benchmarking)
    "relative_power",
    "rel_power_to_plant",
    "rel_temp_to_plant",
    "power_rank_within_plant",
    "temperature_rank_within_plant",
    "efficiency_vs_plant_avg",
    
    # Lags & Baselines
    "pv1_power_lag_1", "pv1_power_lag_3", "pv1_power_lag_288",
    "power_vs_24h_baseline",
    "power_trend_12h",
    "power_trend_24h",
    "efficiency_7d_trend",
    "efficiency_trend_24h",
    "power_ramp_rate",
    "temperature_gradient_6h",
    
    # Raw sensors kept as features
    "pv1_current", "pv1_voltage", "pv1_power",
    "pv2_current", "pv2_voltage", "pv2_power",
    "meter_active_power", "meter_pf", "meter_freq",
    "meter_v_r", "meter_v_y", "meter_v_b",
    "inverter_temperature",
]

_STRING_COLS: List[str] = [f"smu_string{i}" for i in range(1, 25)] + [f"inv_string{i}" for i in range(1, 25)]
_SAMPLES_24H: int = config.SAMPLES_PER_DAY            # 96
_SAMPLES_12H: int = config.SAMPLES_PER_DAY // 2       # 48
_SAMPLES_6H: int  = config.SAMPLES_PER_DAY // 4       # 24
_SAMPLES_7D: int = config.SAMPLES_PER_DAY * 7         # 672
_SAMPLES_30D: int = config.SAMPLES_PER_DAY * 30       # 2880
_INTERVAL_HOURS: float = config.TELEMETRY_INTERVAL_MINUTES / 60.0  # 0.25 h


# =====================================================================
# Individual feature helpers
# =====================================================================

def _add_cyclical_time(df: pd.DataFrame) -> pd.DataFrame:
    """Add cyclical sine/cosine encodings for hour, month, and day-of-year."""
    ts = pd.to_datetime(df["timestamp"])
    hour = ts.dt.hour + ts.dt.minute / 60.0
    month = ts.dt.month
    day_of_year = ts.dt.dayofyear

    df["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    df["month_sin"] = np.sin(2 * np.pi * month / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * month / 12.0)
    df["day_of_year_sin"] = np.sin(2 * np.pi * day_of_year / 365.25)
    df["day_of_year_cos"] = np.cos(2 * np.pi * day_of_year / 365.25)
    
    pv1_pwr = df.get("pv1_power", pd.Series(0, index=df.index)).fillna(0)
    df["daylight_hours_indicator"] = (pv1_pwr > 50.0).astype(float)
    return df


def _add_physics_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute conversion efficiency and thermal gradient."""
    # Sensor clipping
    if "inverter_temperature" in df.columns:
        df["inverter_temperature"] = df["inverter_temperature"].clip(lower=-20, upper=120)
    if "pv1_power" in df.columns:
        df["pv1_power"] = df["pv1_power"].clip(lower=0)
    if "pv2_power" in df.columns:
        df["pv2_power"] = df["pv2_power"].clip(lower=0)
    if "meter_active_power" in df.columns:
        df["meter_active_power"] = df["meter_active_power"].clip(lower=0)

    pv1 = df.get("pv1_power", pd.Series(0, index=df.index)).astype(float)
    pv2 = df.get("pv2_power", pd.Series(0, index=df.index)).astype(float).fillna(0)
    total_dc = pv1 + pv2
    
    # Mask nighttime readings
    total_dc_safe = total_dc.where(total_dc >= config.NIGHTTIME_POWER_THRESHOLD, other=np.nan)
    meter_ac = df.get("meter_active_power", pd.Series(0, index=df.index)).astype(float)
    
    df["conversion_efficiency"] = meter_ac / total_dc_safe
    df["power_efficiency_ratio"] = (meter_ac / (pv1.replace(0, np.nan))).clip(lower=0, upper=2)
    
    inv_temp = df.get("inverter_temperature", pd.Series(np.nan, index=df.index)).astype(float)
    df["thermal_gradient"] = inv_temp.diff()
    df["temp_power_interaction"] = inv_temp * pv1
    
    # mppt imbalance
    df["mppt_imbalance_ratio"] = (pv1 - pv2).abs() / (total_dc_safe + 1e-6)
    
    # grid stability
    freq = df.get("meter_freq", pd.Series(50.0, index=df.index)).astype(float)
    df["frequency_deviation"] = (freq - 50.0).abs()
    
    return df


def _add_string_anomaly(df: pd.DataFrame) -> pd.DataFrame:
    """Compute string mismatch std and coefficient of variation."""
    present = [c for c in _STRING_COLS if c in df.columns]
    if not present:
        df["string_mismatch_std"] = np.nan
        df["string_mismatch_cv"] = np.nan
        df["string_max_deviation"] = np.nan
        return df

    string_vals = df[present].astype(float)
    df["string_mismatch_std"] = string_vals.std(axis=1, skipna=True)
    df["string_current_variance"] = string_vals.var(axis=1, skipna=True)
    row_mean = string_vals.mean(axis=1, skipna=True).replace(0, np.nan)
    df["string_mismatch_cv"] = df["string_mismatch_std"] / row_mean
    
    max_val = string_vals.max(axis=1, skipna=True)
    min_val = string_vals.min(axis=1, skipna=True)
    
    diff_max = (max_val - row_mean).abs()
    diff_min = (min_val - row_mean).abs()
    
    df["string_max_deviation"] = diff_max.where(diff_max > diff_min, diff_min) / row_mean
    return df


def _add_plant_context_features(master: pd.DataFrame) -> pd.DataFrame:
    """Benchmarking: Compare each inverter against the plant average at each timestamp."""
    log.info("adding_plant_context_features")
    
    # Compute plant averages per timestamp
    plant_avgs = master.groupby(["plant_id", "timestamp"])[["pv1_power", "inverter_temperature", "conversion_efficiency"]].transform("mean")
    
    master["rel_power_to_plant"] = master["pv1_power"] / (plant_avgs["pv1_power"] + 1e-6)
    master["relative_power"] = master["rel_power_to_plant"] # Alias for the requested feature name
    master["rel_temp_to_plant"] = master["inverter_temperature"] - plant_avgs["inverter_temperature"]
    master["efficiency_vs_plant_avg"] = master["conversion_efficiency"] - plant_avgs["conversion_efficiency"]
    
    # Ranks
    master["power_rank_within_plant"] = master.groupby(["plant_id", "timestamp"])["pv1_power"].rank(pct=True, ascending=False)
    master["temperature_rank_within_plant"] = master.groupby(["plant_id", "timestamp"])["inverter_temperature"].rank(pct=True, ascending=False)
    
    return master


def _add_lag_features(grp: pd.DataFrame) -> pd.DataFrame:
    """Compute lagged features within a single inverter group."""
    if "pv1_power" in grp.columns:
        pv1 = grp["pv1_power"].astype(float)
        grp["pv1_power_lag_1"] = pv1.shift(1)
        grp["pv1_power_lag_3"] = pv1.shift(3)
        grp["pv1_power_lag_288"] = pv1.shift(288)
        
        # Power ramp rate (diff over 3 samples)
        grp["power_ramp_rate"] = pv1.diff(periods=3)
    else:
        grp["pv1_power_lag_1"] = np.nan
        grp["pv1_power_lag_3"] = np.nan
        grp["pv1_power_lag_288"] = np.nan
        grp["power_ramp_rate"] = np.nan
    return grp


def _rolling_slope(series: pd.Series, window: int, min_valid: int = 50) -> pd.Series:
    def _slope(arr: np.ndarray) -> float:
        valid = arr[~np.isnan(arr)]
        if len(valid) < min_valid:
            return np.nan
        x = np.arange(len(valid), dtype=np.float64)
        try:
            coeffs = np.polyfit(x, valid, 1)
            return float(coeffs[0])
        except (np.linalg.LinAlgError, ValueError):
            return np.nan

    return series.rolling(window=window, min_periods=min_valid).apply(_slope, raw=True)


def _add_rolling_features(grp: pd.DataFrame) -> pd.DataFrame:
    """Compute rolling aggregations within a single inverter group."""
    if "inverter_temperature" in grp.columns:
        temp = grp["inverter_temperature"].astype(float)
        grp["temperature_mean_6h"] = temp.rolling(window=_SAMPLES_6H, min_periods=_SAMPLES_6H // 2).mean()
        grp["temp_rolling_std_12h"] = temp.rolling(window=_SAMPLES_12H, min_periods=_SAMPLES_12H // 2).std()
        grp["temp_rolling_mean_24h"] = temp.rolling(window=_SAMPLES_24H, min_periods=_SAMPLES_24H // 2).mean()
        grp["temp_rolling_min_24h"] = temp.rolling(window=_SAMPLES_24H, min_periods=_SAMPLES_24H // 2).min()
        grp["temp_rolling_max_24h"] = temp.rolling(window=_SAMPLES_24H, min_periods=_SAMPLES_24H // 2).max()
        grp["temperature_gradient_6h"] = temp.diff(periods=_SAMPLES_6H) / 6.0
    else:
        grp["temperature_mean_6h"] = np.nan
        grp["temp_rolling_std_12h"] = np.nan
        grp["temp_rolling_mean_24h"] = np.nan
        grp["temp_rolling_min_24h"] = np.nan
        grp["temp_rolling_max_24h"] = np.nan
        grp["temperature_gradient_6h"] = np.nan

    if "conversion_efficiency" in grp.columns:
        eff = grp["conversion_efficiency"].astype(float)
        grp["efficiency_7d_trend"] = _rolling_slope(eff, window=_SAMPLES_7D, min_valid=200)
        grp["efficiency_trend_24h"] = _rolling_slope(eff, window=_SAMPLES_24H, min_valid=50)
        grp["efficiency_volatility_24h"] = eff.rolling(window=_SAMPLES_24H, min_periods=10).std()
        # Drop rate (rate of change over 24 samples)
        grp["efficiency_drop_rate"] = eff.diff(_SAMPLES_6H) / _SAMPLES_6H
        
        # Temp rises but efficiency drops
        if "thermal_gradient" in grp.columns:
            eff_diff = eff.diff()
            grp["temp_efficiency_divergence"] = (grp["thermal_gradient"].astype(float) > 2) & (eff_diff < -0.05)
            grp["temp_efficiency_divergence"] = grp["temp_efficiency_divergence"].astype(float)
    else:
        grp["efficiency_7d_trend"] = np.nan
        grp["efficiency_trend_24h"] = np.nan
        grp["efficiency_volatility_24h"] = np.nan
        grp["efficiency_drop_rate"] = np.nan
        grp["temp_efficiency_divergence"] = np.nan
        
    if "pv1_power" in grp.columns:
        pv1 = grp["pv1_power"].astype(float)
        grp["power_trend_12h"] = _rolling_slope(pv1, window=_SAMPLES_12H, min_valid=24)
        grp["power_trend_24h"] = _rolling_slope(pv1, window=_SAMPLES_24H, min_valid=48)
    else:
        grp["power_trend_12h"] = np.nan
        grp["power_trend_24h"] = np.nan
        
    if "mppt_imbalance_ratio" in grp.columns:
        grp["mppt_power_ratio_24h"] = grp["mppt_imbalance_ratio"].astype(float).rolling(window=_SAMPLES_24H, min_periods=10).mean()
    else:
        grp["mppt_power_ratio_24h"] = np.nan
        
    if "string_mismatch_std" in grp.columns:
        # Keep original string_current_variance_24h if it was used anywhere, else we removed it from FEATURE_COLUMNS
        pass
        
    # Grid voltage disturbances
    v_cols = [c for c in ["meter_v_r", "meter_v_y", "meter_v_b"] if c in grp.columns]
    if v_cols:
        v_data = grp[v_cols].astype(float)
        disturbances = (v_data > 250.0).any(axis=1) | (v_data < 200.0).any(axis=1)
        grp["grid_voltage_disturbances_24h"] = disturbances.rolling(window=_SAMPLES_24H, min_periods=1).sum()
    else:
        grp["grid_voltage_disturbances_24h"] = np.nan
        
    return grp


def _add_baseline_deviations(grp: pd.DataFrame) -> pd.DataFrame:
    """Compute baseline deviation features within a single inverter group."""
    if "pv1_power" in grp.columns:
        pv1 = grp["pv1_power"].astype(float)
        lag_288 = pv1.shift(288)
        grp["power_vs_24h_baseline"] = (pv1 - lag_288) / (lag_288 + 1e-6)
    else:
        grp["power_vs_24h_baseline"] = np.nan

    if "inverter_temperature" in grp.columns:
        temp = grp["inverter_temperature"].astype(float)
        temp_30d_99p = temp.rolling(
            window=_SAMPLES_30D, min_periods=config.SAMPLES_PER_DAY
        ).quantile(0.99)
        grp["temp_vs_30d_percentile"] = temp - temp_30d_99p
    else:
        grp["temp_vs_30d_percentile"] = np.nan

    return grp


# =====================================================================
# Missing-value handling
# =====================================================================

def _handle_missing(grp: pd.DataFrame) -> pd.DataFrame:
    """Forward-fill short gaps, but DO NOT aggressively drop rows."""
    feature_cols = [c for c in FEATURE_COLUMNS if c in grp.columns]
    
    # We forward fill up to 8 intervals (2 hours) so we don't lose data during transient logger drops
    grp[feature_cols] = grp[feature_cols].ffill(limit=8)

    # We only drop rows if they have literally no telemetry (e.g. 100% missing key sensors)
    # The XGBoost model handles NaNs natively, so we preserve the sparse data structure.
    vital_cols = ["inverter_temperature", "meter_active_power", "pv1_power", "meter_v_r"]
    present_vitals = [c for c in vital_cols if c in grp.columns]
    
    if present_vitals:
        nan_frac = grp[present_vitals].isna().mean(axis=1)
        before = len(grp)
        # 1.0 means all vital cols are NaN. If at least one vital col exists, we keep the row.
        grp = grp[nan_frac < 1.0].copy()
        dropped = before - len(grp)
        if dropped > 0:
            inv = grp["inverter_id"].iloc[0] if len(grp) > 0 else "UNKNOWN"
            log.info("rows_dropped_no_vital_telemetry", inverter_id=inv, count=dropped, before=before)
    
    return grp


# =====================================================================
# Public API — batch and streaming modes
# =====================================================================

def compute_features_batch(master: pd.DataFrame) -> pd.DataFrame:
    assert "inverter_id" in master.columns, "inverter_id column is required"

    log.info("feature_pipeline_batch_start", rows=len(master))
    master = master.sort_values(["inverter_id", "timestamp"]).copy()

    master = _add_cyclical_time(master)
    master = _add_physics_features(master)
    master = _add_string_anomaly(master)

    # 1. Intra-inverter features (lags, rolling)
    groups: List[pd.DataFrame] = []
    for inv_id, grp in master.groupby("inverter_id", sort=False):
        grp = grp.sort_values("timestamp").copy()
        grp = _add_lag_features(grp)
        grp = _add_rolling_features(grp)
        grp = _add_baseline_deviations(grp)
        grp = _handle_missing(grp)
        groups.append(grp)
        log.info("inverter_features_computed", inverter_id=inv_id, rows=len(grp))

    result = pd.concat(groups, ignore_index=True)

    # 2. Inter-inverter features (Plant Context)
    result = _add_plant_context_features(result)

    for col in FEATURE_COLUMNS:
        if col not in result.columns:
            result[col] = np.nan
            log.warning("feature_column_missing_added_nan", column=col)

    id_cols = ["inverter_id", "plant_id", "block_id", "timestamp"]
    label_cols = ["label", "label_source"]
    telemetry_cols = ["inverter_temperature", "pv1_power", "conversion_efficiency"]
    keep = id_cols + [c for c in label_cols if c in result.columns] + telemetry_cols + FEATURE_COLUMNS
    result = result[[c for c in keep if c in result.columns]].copy()

    out_path = config.PROCESSED_DIR / "features.parquet"
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    result.to_parquet(out_path, index=False, engine="pyarrow")
    log.info("features_saved", path=str(out_path), rows=len(result))

    return result


def compute_features_streaming(recent_df: pd.DataFrame, inverter_id: str) -> Optional[pd.DataFrame]:
    if len(recent_df) < _SAMPLES_24H:
        log.warning("streaming_input_too_short", inverter_id=inverter_id, rows=len(recent_df))
        return None

    assert "inverter_id" in recent_df.columns, "inverter_id column is required"

    df = recent_df.sort_values("timestamp").copy()

    df = _add_cyclical_time(df)
    df = _add_physics_features(df)
    df = _add_string_anomaly(df)
    df = _add_lag_features(df)
    df = _add_rolling_features(df)
    df = _add_baseline_deviations(df)
    df = _handle_missing(df)

    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan

    return df.tail(1)


def main() -> None:
    master_path = config.PROCESSED_DIR / "master_labelled.parquet"
    if not master_path.exists():
        log.error("master_labelled_not_found", path=str(master_path))
        sys.exit(1)

    master = pd.read_parquet(master_path)
    result = compute_features_batch(master)

    print(f"\nFeature pipeline complete.")
    print(f"  Rows:            {len(result):,}")
    print(f"  Feature columns: {len(FEATURE_COLUMNS)}")

if __name__ == "__main__":
    main()
