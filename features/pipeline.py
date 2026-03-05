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
    "hour_sin",
    "hour_cos",
    "month_sin",
    "month_cos",
    # Physics-derived
    "conversion_efficiency",
    "thermal_gradient",
    # String anomaly
    "string_mismatch_std",
    "string_mismatch_cv",
    # Lags
    "pv1_power_lag_1",
    "pv1_power_lag_3",
    "pv1_power_lag_288",
    # Rolling aggregations
    "temp_rolling_mean_24h",
    "temp_rolling_max_24h",
    "efficiency_7d_trend",
    # Baseline deviations
    "power_vs_24h_baseline",
    "temp_vs_30d_percentile",
    # Raw sensors kept as features
    "pv1_current",
    "pv1_voltage",
    "pv1_power",
    "meter_active_power",
    "inverter_temperature",
]

_STRING_COLS: List[str] = [f"string{i}" for i in range(1, 11)]
_SAMPLES_24H: int = config.SAMPLES_PER_DAY            # 96
_SAMPLES_7D: int = config.SAMPLES_PER_DAY * 7         # 672
_SAMPLES_30D: int = config.SAMPLES_PER_DAY * 30       # 2880
_INTERVAL_HOURS: float = config.TELEMETRY_INTERVAL_MINUTES / 60.0  # 0.25 h


# =====================================================================
# Individual feature helpers
# =====================================================================

def _add_cyclical_time(df: pd.DataFrame) -> pd.DataFrame:
    """Add cyclical sine/cosine encodings for hour-of-day and month-of-year.

    Args:
        df: DataFrame with a ``timestamp`` column.

    Returns:
        DataFrame with four new columns added in-place.
    """
    ts = pd.to_datetime(df["timestamp"])
    hour = ts.dt.hour + ts.dt.minute / 60.0
    month = ts.dt.month

    df["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    df["month_sin"] = np.sin(2 * np.pi * month / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * month / 12.0)
    return df


def _add_physics_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute conversion efficiency and thermal gradient.

    Args:
        df: DataFrame with ``meter_active_power``, ``pv1_power``,
            ``inverter_temperature``.

    Returns:
        DataFrame with ``conversion_efficiency`` and
        ``thermal_gradient`` columns.
    """
    pv_power = df["pv1_power"].copy()
    # Mask nighttime readings (< 50 W) to avoid spurious efficiency
    pv_power_safe = pv_power.where(
        pv_power >= config.NIGHTTIME_POWER_THRESHOLD, other=np.nan
    )
    df["conversion_efficiency"] = df["meter_active_power"] / pv_power_safe

    # Thermal gradient  (°C / hr) = Δtemp / interval_hours
    df["thermal_gradient"] = (
        df["inverter_temperature"].diff() / _INTERVAL_HOURS
    )
    return df


def _add_string_anomaly(df: pd.DataFrame) -> pd.DataFrame:
    """Compute string mismatch std and coefficient of variation.

    Args:
        df: DataFrame with ``string1`` … ``string10`` columns (may be
            partially NaN).

    Returns:
        DataFrame with ``string_mismatch_std`` and ``string_mismatch_cv``.
    """
    present = [c for c in _STRING_COLS if c in df.columns]
    if not present:
        df["string_mismatch_std"] = np.nan
        df["string_mismatch_cv"] = np.nan
        return df

    string_vals = df[present]
    df["string_mismatch_std"] = string_vals.std(axis=1, skipna=True)
    row_mean = string_vals.mean(axis=1, skipna=True).replace(0, np.nan)
    df["string_mismatch_cv"] = df["string_mismatch_std"] / row_mean
    return df


def _add_lag_features(grp: pd.DataFrame) -> pd.DataFrame:
    """Compute lagged features within a single inverter group.

    Args:
        grp: DataFrame for one ``inverter_id``, sorted by ``timestamp``.

    Returns:
        DataFrame with lag columns added.
    """
    grp["pv1_power_lag_1"] = grp["pv1_power"].shift(1)
    grp["pv1_power_lag_3"] = grp["pv1_power"].shift(3)
    grp["pv1_power_lag_288"] = grp["pv1_power"].shift(288)
    return grp


def _rolling_efficiency_slope(series: pd.Series, window: int = _SAMPLES_7D,
                               min_valid: int = 200) -> pd.Series:
    """Rolling linear-regression slope of conversion efficiency.

    Uses a custom rolling apply with numpy polyfit.

    Args:
        series: Efficiency values.
        window: Rolling window size in samples.
        min_valid: Minimum non-NaN values required.

    Returns:
        Series of slope values.
    """
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

    return series.rolling(window=window, min_periods=min_valid).apply(
        _slope, raw=True
    )


def _add_rolling_features(grp: pd.DataFrame) -> pd.DataFrame:
    """Compute rolling aggregations within a single inverter group.

    Args:
        grp: DataFrame for one ``inverter_id``, sorted by ``timestamp``.

    Returns:
        DataFrame with rolling columns added.
    """
    temp = grp["inverter_temperature"]
    grp["temp_rolling_mean_24h"] = temp.rolling(
        window=_SAMPLES_24H, min_periods=_SAMPLES_24H // 2
    ).mean()
    grp["temp_rolling_max_24h"] = temp.rolling(
        window=_SAMPLES_24H, min_periods=_SAMPLES_24H // 2
    ).max()

    grp["efficiency_7d_trend"] = _rolling_efficiency_slope(
        grp["conversion_efficiency"]
    )
    return grp


def _add_baseline_deviations(grp: pd.DataFrame) -> pd.DataFrame:
    """Compute baseline deviation features within a single inverter group.

    Args:
        grp: DataFrame for one ``inverter_id``, sorted by ``timestamp``.

    Returns:
        DataFrame with deviation columns.
    """
    lag_288 = grp["pv1_power"].shift(288)
    grp["power_vs_24h_baseline"] = (
        (grp["pv1_power"] - lag_288) / (lag_288 + 1e-6)
    )

    temp_30d_99p = grp["inverter_temperature"].rolling(
        window=_SAMPLES_30D, min_periods=config.SAMPLES_PER_DAY
    ).quantile(0.99)
    grp["temp_vs_30d_percentile"] = grp["inverter_temperature"] - temp_30d_99p

    return grp


# =====================================================================
# Missing-value handling
# =====================================================================

def _handle_missing(grp: pd.DataFrame) -> pd.DataFrame:
    """Forward-fill, backward-fill, then drop rows > 40% NaN.

    Fills within each inverter group with a max gap of 4 consecutive
    intervals (1 hour).

    Args:
        grp: DataFrame for one ``inverter_id``.

    Returns:
        Cleaned DataFrame (may have fewer rows).
    """
    feature_cols = [c for c in FEATURE_COLUMNS if c in grp.columns]
    grp[feature_cols] = grp[feature_cols].ffill(limit=4).bfill(limit=4)

    # Drop rows where > 40% of features are NaN
    nan_frac = grp[feature_cols].isna().mean(axis=1)
    before = len(grp)
    grp = grp[nan_frac <= 0.4].copy()
    dropped = before - len(grp)
    if dropped > 0:
        inv = grp["inverter_id"].iloc[0] if len(grp) > 0 else "UNKNOWN"
        log.info("rows_dropped_high_nan", inverter_id=inv, count=dropped)
    return grp


# =====================================================================
# Public API — batch and streaming modes
# =====================================================================

def compute_features_batch(master: pd.DataFrame) -> pd.DataFrame:
    """Compute all features on the full master DataFrame (batch mode).

    Args:
        master: Labelled master telemetry DataFrame.

    Returns:
        Feature DataFrame saved to ``data/processed/features.parquet``.
    """
    assert "inverter_id" in master.columns, "inverter_id column is required"

    log.info("feature_pipeline_batch_start", rows=len(master))
    master = master.sort_values(["inverter_id", "timestamp"]).copy()

    # ── Global features (no groupby needed) ─────────────────────────
    master = _add_cyclical_time(master)
    master = _add_physics_features(master)
    master = _add_string_anomaly(master)

    # ── Per-inverter features ───────────────────────────────────────
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

    # ── Ensure all FEATURE_COLUMNS are present ──────────────────────
    for col in FEATURE_COLUMNS:
        if col not in result.columns:
            result[col] = np.nan
            log.warning("feature_column_missing_added_nan", column=col)

    # ── Select output columns ───────────────────────────────────────
    id_cols = ["inverter_id", "plant_id", "block_id", "timestamp"]
    label_cols = ["label", "label_source"]
    keep = id_cols + [c for c in label_cols if c in result.columns] + FEATURE_COLUMNS
    result = result[[c for c in keep if c in result.columns]].copy()

    # ── Save ────────────────────────────────────────────────────────
    out_path = config.PROCESSED_DIR / "features.parquet"
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    result.to_parquet(out_path, index=False, engine="pyarrow")
    log.info("features_saved", path=str(out_path), rows=len(result))

    return result


def compute_features_streaming(
    recent_df: pd.DataFrame,
    inverter_id: str,
) -> Optional[pd.DataFrame]:
    """Compute features for a single inverter's recent telemetry window.

    Produces the same output columns as batch mode so downstream models
    receive identically structured inputs.

    Args:
        recent_df: Recent telemetry rows for one inverter, sorted by
                   ``timestamp`` ascending. Should contain at least 96
                   rows (1 day) for rolling features to be meaningful.
        inverter_id: ID of the inverter.

    Returns:
        Feature DataFrame (single or few rows at the tail), or ``None``
        if the input is too short.
    """
    if len(recent_df) < _SAMPLES_24H:
        log.warning(
            "streaming_input_too_short",
            inverter_id=inverter_id,
            rows=len(recent_df),
        )
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

    # Ensure all feature columns present
    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan

    # Return only the last row (current inference point)
    return df.tail(1)


# =====================================================================
# CLI entry point
# =====================================================================

def main() -> None:
    """Run the batch feature pipeline on the labelled master telemetry."""
    master_path = config.PROCESSED_DIR / "master_labelled.parquet"
    if not master_path.exists():
        log.error(
            "master_labelled_not_found",
            path=str(master_path),
            hint="Run scripts/ingest_raw.py first",
        )
        sys.exit(1)

    master = pd.read_parquet(master_path)
    result = compute_features_batch(master)

    print(f"\nFeature pipeline complete.")
    print(f"  Rows:            {len(result):,}")
    print(f"  Feature columns: {len(FEATURE_COLUMNS)}")
    print(f"  Columns list:    {FEATURE_COLUMNS}")


if __name__ == "__main__":
    main()
