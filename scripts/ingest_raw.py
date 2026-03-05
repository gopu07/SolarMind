"""
Layer 1 — Data Ingestion and Merging.

Entry-point script that loads raw inverter CSVs, resolves MAC-to-inverter
mappings, standardises schemas, applies hybrid labelling, and produces
``data/processed/master_labelled.parquet``.

Usage::

    python -m scripts.ingest_raw
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import structlog

# ── Ensure project root is on sys.path ──────────────────────────────
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import config  # noqa: E402

log = structlog.get_logger(__name__)

# ── Canonical column schema ─────────────────────────────────────────
CANONICAL_COLUMNS: List[str] = [
    "timestamp",
    "pv1_current",
    "pv1_voltage",
    "pv1_power",
    "meter_active_power",
    "inverter_temperature",
] + [f"string{i}" for i in range(1, 11)]

TIMESTAMP_FORMATS: List[str] = [
    "%Y-%m-%d %H:%M:%S",
    "%d/%m/%Y %H:%M",
    "%m/%d/%Y %H:%M:%S",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%d %H:%M",
]


# =====================================================================
# Step 0 — Auto-generate MAC mapping if missing
# =====================================================================
def _auto_generate_mac_mapping() -> pd.DataFrame:
    """Scan ``data/raw/`` and generate a MAC mapping from CSV filenames.

    Creates sequential inverter IDs (``INV_001``, ``INV_002``, …) with
    default ``plant_id="PLANT_1"`` and ``block_id="BLOCK_A"``.

    Returns:
        Generated mapping DataFrame.

    Side effects:
        Writes ``data/mappings/mac_mapping.csv``.
    """
    raw_dir: Path = config.RAW_DATA_DIR
    csv_files = sorted(raw_dir.rglob("*.csv")) if raw_dir.exists() else []

    if not csv_files:
        log.error(
            "no_raw_csvs_for_mapping",
            path=str(raw_dir),
            hint="Place at least one inverter CSV in data/raw/",
        )
        sys.exit(1)

    rows = []
    for idx, csv_path in enumerate(csv_files, start=1):
        mac = csv_path.stem.strip().upper()
        rows.append(
            {
                "mac_address": mac,
                "inverter_id": f"INV_{idx:03d}",
                "plant_id": "PLANT_1",
                "block_id": "BLOCK_A",
            }
        )
        log.info(
            "auto_mapped_inverter",
            filename=csv_path.name,
            inverter_id=f"INV_{idx:03d}",
            mac_address=mac,
        )

    mapping_df = pd.DataFrame(rows)
    config.MAPPINGS_DIR.mkdir(parents=True, exist_ok=True)
    mapping_df.to_csv(config.MAC_MAPPING_FILE, index=False)
    log.info(
        "auto_generated_mac_mapping",
        path=str(config.MAC_MAPPING_FILE),
        inverters=len(mapping_df),
    )
    return mapping_df


# =====================================================================
# Step 1 — Load the MAC mapping file
# =====================================================================
def load_mac_mapping() -> pd.DataFrame:
    """Load the MAC-to-inverter mapping CSV, auto-generating it if absent.

    If ``data/mappings/mac_mapping.csv`` does not exist, the function
    scans ``data/raw/`` and creates a mapping automatically so the
    pipeline never halts due to a missing mapping file.

    Returns:
        DataFrame with columns ``mac_address``, ``inverter_id``,
        ``plant_id``, ``block_id``.
    """
    mapping_path: Path = config.MAC_MAPPING_FILE

    if not mapping_path.exists():
        log.warning(
            "mac_mapping_not_found_auto_generating",
            path=str(mapping_path),
        )
        return _auto_generate_mac_mapping()

    df = pd.read_csv(mapping_path, dtype=str)
    required = {"mac_address", "inverter_id", "plant_id", "block_id"}
    missing = required - set(df.columns)
    if missing:
        log.warning(
            "mac_mapping_columns_missing_regenerating",
            missing=list(missing),
        )
        return _auto_generate_mac_mapping()

    df["mac_address"] = df["mac_address"].str.strip().str.upper()
    log.info("mac_mapping_loaded", rows=len(df))
    return df


# =====================================================================
# Step 2 — Discover and load all inverter CSVs
# =====================================================================
def discover_csvs(mapping: pd.DataFrame) -> List[dict]:
    """Scan ``data/raw/`` for CSVs and resolve each file's MAC address.

    Args:
        mapping: MAC-to-inverter mapping from :func:`load_mac_mapping`.

    Returns:
        List of dicts, each with ``path``, ``mac``, ``inverter_id``,
        ``plant_id``, ``block_id``.
    """
    raw_dir: Path = config.RAW_DATA_DIR
    if not raw_dir.exists():
        log.error("raw_data_dir_missing", path=str(raw_dir))
        sys.exit(1)

    csv_files = list(raw_dir.rglob("*.csv"))
    if not csv_files:
        log.warning("no_csv_files_found", path=str(raw_dir))
        return []

    mac_lookup = mapping.set_index("mac_address")
    resolved: List[dict] = []

    for csv_path in csv_files:
        mac = csv_path.stem.strip().upper()
        if mac not in mac_lookup.index:
            log.warning(
                "unmapped_mac_address",
                filename=csv_path.name,
                mac=mac,
            )
            continue
        row = mac_lookup.loc[mac]
        inv_id = row["inverter_id"]
        p_id = row["plant_id"]
        resolved.append(
            {
                "path": csv_path,
                "mac": mac,
                "inverter_id": inv_id,
                "plant_id": p_id,
                "block_id": row["block_id"],
            }
        )
        log.info(
            "inverter_csv_loaded",
            inverter_id=inv_id,
            plant_id=p_id,
            filename=csv_path.name,
        )

    log.info(
        "csvs_discovered",
        total_files=len(csv_files),
        mapped=len(resolved),
        skipped=len(csv_files) - len(resolved),
    )
    return resolved


# =====================================================================
# Step 3 — Standardise each CSV
# =====================================================================
def _try_parse_timestamps(series: pd.Series) -> Optional[pd.Series]:
    """Attempt to parse a timestamp series using several known formats.

    Args:
        series: Raw timestamp strings.

    Returns:
        UTC-aware DatetimeIndex on success, ``None`` on failure.
    """
    for fmt in TIMESTAMP_FORMATS:
        try:
            parsed = pd.to_datetime(series, format=fmt)
            if parsed.notna().sum() > 0:
                return parsed.dt.tz_localize("UTC") if parsed.dt.tz is None else parsed.dt.tz_convert("UTC")
        except (ValueError, TypeError):
            continue
    # Fallback: infer
    try:
        parsed = pd.to_datetime(series, infer_datetime_format=True)
        return parsed.dt.tz_localize("UTC") if parsed.dt.tz is None else parsed.dt.tz_convert("UTC")
    except Exception:
        return None


def standardise_csv(entry: dict) -> Optional[pd.DataFrame]:
    """Load a single inverter CSV, normalise columns, and reindex to 15-min.

    Args:
        entry: Dict from :func:`discover_csvs` with ``path`` and identity cols.

    Returns:
        Cleaned DataFrame or ``None`` if critical parsing fails.
    """
    path: Path = entry["path"]
    inverter_id = entry["inverter_id"]
    plant_id = entry["plant_id"]
    block_id = entry["block_id"]

    log.info("standardising_csv", file=path.name, inverter_id=inverter_id)
    df = pd.read_csv(path)

    # ── Identify timestamp column ────────────────────────────────────
    ts_col = None
    for col in df.columns:
        if "time" in col.lower() or "date" in col.lower():
            ts_col = col
            break
    if ts_col is None:
        ts_col = df.columns[0]

    parsed = _try_parse_timestamps(df[ts_col])
    if parsed is None:
        log.error(
            "timestamp_parse_failed",
            file=path.name,
            first_value=str(df[ts_col].iloc[0]) if len(df) > 0 else "EMPTY",
        )
        return None

    df["timestamp"] = parsed

    # ── Map column names to canonical form (best-effort) ─────────────
    col_map = {}
    lower_cols = {c.lower().replace(" ", "_").replace("-", "_"): c for c in df.columns}
    for canon in CANONICAL_COLUMNS:
        if canon == "timestamp":
            continue
        if canon in lower_cols:
            col_map[lower_cols[canon]] = canon
        else:
            found = False
            for lc, orig in lower_cols.items():
                if canon.replace("_", "") in lc.replace("_", ""):
                    col_map[orig] = canon
                    found = True
                    break
            if not found:
                df[canon] = np.nan
                log.warning(
                    "missing_column_filled_nan",
                    file=path.name,
                    column=canon,
                    inverter_id=inverter_id,
                )
    if col_map:
        df.rename(columns=col_map, inplace=True)

    # ── Keep only canonical columns ──────────────────────────────────
    keep = ["timestamp"] + [c for c in CANONICAL_COLUMNS if c != "timestamp"]
    for c in keep:
        if c not in df.columns:
            df[c] = np.nan
    df = df[keep].copy()

    # ── Add identity columns ─────────────────────────────────────────
    df["inverter_id"] = inverter_id
    df["plant_id"] = plant_id
    df["block_id"] = block_id

    # ── Sort and reindex to 15-min ───────────────────────────────────
    df.sort_values("timestamp", inplace=True)
    df.drop_duplicates(subset=["timestamp"], keep="first", inplace=True)
    df.set_index("timestamp", inplace=True)

    full_idx = pd.date_range(
        start=df.index.min(),
        end=df.index.max(),
        freq=f"{config.TELEMETRY_INTERVAL_MINUTES}min",
        tz="UTC",
    )
    df = df.reindex(full_idx)
    df.index.name = "timestamp"

    # Re-fill identity columns after reindex
    df["inverter_id"] = inverter_id
    df["plant_id"] = plant_id
    df["block_id"] = block_id

    df.reset_index(inplace=True)
    log.info(
        "csv_standardised",
        inverter_id=inverter_id,
        rows=len(df),
    )
    return df


# =====================================================================
# Step 4 — Concatenate all inverters
# =====================================================================
def merge_all(entries: List[dict]) -> pd.DataFrame:
    """Standardise and concatenate all inverter CSVs into a master DataFrame.

    Args:
        entries: List of resolved CSV entries.

    Returns:
        Master DataFrame saved to ``data/processed/master_telemetry.parquet``.
    """
    frames: List[pd.DataFrame] = []
    for entry in entries:
        df = standardise_csv(entry)
        if df is not None:
            frames.append(df)

    if not frames:
        log.error("no_valid_dataframes_produced")
        sys.exit(1)

    master = pd.concat(frames, ignore_index=True)

    # Assert uniqueness of (inverter_id, timestamp)
    dupes = master.duplicated(subset=["inverter_id", "timestamp"], keep="first")
    if dupes.any():
        log.warning("duplicate_rows_found", count=int(dupes.sum()))
        master = master[~dupes].copy()

    # Save
    out_path = config.PROCESSED_DIR / "master_telemetry.parquet"
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    master.to_parquet(out_path, index=False, engine="pyarrow")
    log.info("master_telemetry_saved", path=str(out_path), rows=len(master))
    return master


# =====================================================================
# Step 5 — Hybrid labelling
# =====================================================================
def _apply_hard_labels(
    master: pd.DataFrame, events: pd.DataFrame
) -> pd.DataFrame:
    """Apply hard labels from the events file using a lookback window.

    Args:
        master: Master telemetry DataFrame.
        events: Events DataFrame with ``inverter_id``, ``event_timestamp``,
                ``event_type``.

    Returns:
        Master DataFrame with ``label`` and ``label_source`` partially filled.
    """
    master["label"] = 0
    master["label_source"] = "negative"
    master["timestamp"] = pd.to_datetime(master["timestamp"], utc=True)

    for _, event in events.iterrows():
        inv_id = event["inverter_id"]
        ev_ts = event["event_timestamp"]
        window_start = ev_ts - pd.Timedelta(days=config.LABEL_WINDOW_DAYS)

        mask = (
            (master["inverter_id"] == inv_id)
            & (master["timestamp"] >= window_start)
            & (master["timestamp"] <= ev_ts)
        )
        master.loc[mask, "label"] = 1
        master.loc[mask, "label_source"] = "events_file"

    return master


def _apply_inferred_labels(master: pd.DataFrame) -> pd.DataFrame:
    """Apply inferred labels using telemetry heuristics for unlabelled rows.

    Assigns ``y = 1`` if anomalous conditions persist for >= 6 consecutive
    hours AND are followed within 10 days by a sustained power drop.

    Args:
        master: Master DataFrame (already has hard labels applied).

    Returns:
        Master DataFrame with inferred labels added where applicable.
    """
    for inv_id, grp in master.groupby("inverter_id"):
        grp = grp.sort_values("timestamp").copy()
        idx = grp.index

        # Compute rolling baselines
        eff = grp["meter_active_power"] / grp["pv1_power"].replace(0, np.nan)
        eff_30d_median = eff.rolling(
            window=config.SAMPLES_PER_DAY * 30, min_periods=config.SAMPLES_PER_DAY
        ).median()

        temp_99p = grp["inverter_temperature"].rolling(
            window=config.SAMPLES_PER_DAY * 30, min_periods=config.SAMPLES_PER_DAY
        ).quantile(0.99)

        # Anomaly flags
        eff_anomaly = eff < (eff_30d_median * (1 - config.EFFICIENCY_DROP_THRESHOLD))
        temp_anomaly = grp["inverter_temperature"] > temp_99p

        string_cols = [c for c in grp.columns if c.startswith("string")]
        if string_cols:
            string_vals = grp[string_cols]
            string_mean = string_vals.mean(axis=1)
            string_std = string_vals.std(axis=1)
            string_cv = string_std / string_mean.replace(0, np.nan)
            string_anomaly = string_cv > config.STRING_MISMATCH_CV_THRESHOLD
        else:
            string_anomaly = pd.Series(False, index=idx)

        any_anomaly = eff_anomaly | temp_anomaly | string_anomaly

        # Mark consecutive runs of >= 6 hours (24 intervals of 15 min)
        min_consecutive = config.CONSECUTIVE_HOURS_THRESHOLD * 4  # 15-min intervals
        run_lengths = any_anomaly.astype(int)
        cumsum = run_lengths.cumsum()
        block_id_series = cumsum - cumsum.where(~any_anomaly).ffill().fillna(0)
        sustained_anomaly = block_id_series >= min_consecutive

        # Check for power drop within 10 days after anomaly
        power_7d_median = grp["meter_active_power"].rolling(
            window=config.SAMPLES_PER_DAY * 7, min_periods=config.SAMPLES_PER_DAY
        ).median()
        power_drop = (
            grp["meter_active_power"]
            < power_7d_median * (1 - config.POWER_DROP_THRESHOLD)
        )

        # For each row with sustained anomaly, check if power drop follows
        # within LABEL_WINDOW_DAYS
        lookahead = config.LABEL_WINDOW_DAYS * config.SAMPLES_PER_DAY
        power_drop_forward = power_drop.rolling(
            window=lookahead, min_periods=1
        ).max().shift(-lookahead).fillna(0).astype(bool)

        inferred_mask = sustained_anomaly & power_drop_forward
        # Only apply where no hard label exists
        already_labelled = master.loc[idx, "label_source"] == "events_file"
        apply_mask = inferred_mask & ~already_labelled

        master.loc[idx[apply_mask], "label"] = 1
        master.loc[idx[apply_mask], "label_source"] = "inferred"

    return master


def load_events() -> pd.DataFrame:
    """Load and standardise the events file.

    If the events file is missing, returns an empty DataFrame with the
    correct schema so the pipeline can proceed using inferred labels only.

    Returns:
        Events DataFrame (may be empty if no events file exists).
    """
    _empty_events = pd.DataFrame(
        columns=["inverter_id", "event_timestamp", "event_type"]
    )

    events_path: Path = config.EVENTS_FILE
    if not events_path.exists():
        # Try Excel variant
        xlsx_variant = events_path.with_suffix(".xlsx")
        if xlsx_variant.exists():
            events_path = xlsx_variant
        else:
            log.warning(
                "no_events_file_detected",
                path=str(events_path),
                action="Proceeding using inferred telemetry labels only.",
            )
            return _empty_events

    if events_path.suffix == ".xlsx":
        df = pd.read_excel(events_path)
    else:
        df = pd.read_csv(events_path)

    # Standardise column names
    col_map = {}
    lower_cols = {c.lower().replace(" ", "_").replace("-", "_"): c for c in df.columns}
    for target in ["inverter_id", "event_timestamp", "event_type"]:
        if target in lower_cols:
            col_map[lower_cols[target]] = target
    if col_map:
        df.rename(columns=col_map, inplace=True)

    if "event_timestamp" in df.columns:
        df["event_timestamp"] = pd.to_datetime(df["event_timestamp"], utc=True)

    log.info("events_loaded", rows=len(df))
    return df


def apply_labels(master: pd.DataFrame) -> pd.DataFrame:
    """Construct labels using the hybrid strategy.

    Args:
        master: Master telemetry DataFrame.

    Returns:
        Labelled master DataFrame.
    """
    events = load_events()

    if len(events) > 0:
        master = _apply_hard_labels(master, events)
    else:
        master["label"] = 0
        master["label_source"] = "negative"
        log.info(
            "no_events_available",
            action="Labels will be inferred from telemetry heuristics only.",
        )

    master = _apply_inferred_labels(master)
    return master


def print_label_statistics(df: pd.DataFrame) -> None:
    """Print a full label statistics report to stdout.

    Args:
        df: Labelled master DataFrame.
    """
    total = len(df)
    positives = int(df["label"].sum())
    rate = positives / total if total > 0 else 0.0

    print("\n" + "=" * 60)
    print("LABEL STATISTICS REPORT")
    print("=" * 60)
    print(f"Total rows:           {total:,}")
    print(f"Total positives:      {positives:,}")
    print(f"Overall positive rate: {rate:.4%}")
    print()

    # Breakdown by source
    print("Breakdown by label_source:")
    for source, count in df["label_source"].value_counts().items():
        print(f"  {source:20s}: {count:>10,}")
    print()

    # Per-plant breakdown
    print("Positive rate per plant:")
    for plant_id, grp in df.groupby("plant_id"):
        p_rate = grp["label"].mean()
        flag = ""
        if p_rate < 0.005:
            flag = " ⚠️  WARNING: < 0.5%"
        elif p_rate > 0.15:
            flag = " ⚠️  WARNING: > 15%"
        print(f"  Plant {plant_id}: {p_rate:.4%}{flag}")

    print("=" * 60 + "\n")


# =====================================================================
# Main entry point
# =====================================================================
def main() -> None:
    """Run the full ingestion pipeline."""
    log.info("ingestion_started")

    # Step 1: Load mapping
    mapping = load_mac_mapping()

    # Step 2: Discover CSVs
    entries = discover_csvs(mapping)
    if not entries:
        log.error("no_csvs_resolved_halting")
        sys.exit(1)

    # Steps 3 & 4: Standardise and merge
    master = merge_all(entries)

    # Step 5: Label
    master = apply_labels(master)

    # Save labelled master
    out_path = config.PROCESSED_DIR / "master_labelled.parquet"
    master.to_parquet(out_path, index=False, engine="pyarrow")
    log.info("master_labelled_saved", path=str(out_path), rows=len(master))

    # Print stats
    print_label_statistics(master)

    log.info("ingestion_completed")


if __name__ == "__main__":
    main()
