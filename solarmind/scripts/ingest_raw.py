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
CANONICAL_COLUMNS: List[str] = config.CANONICAL_COLUMNS

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
    # ── Check for Unix Epoch Milliseconds ────────────────────────────
    try:
        numeric_series = pd.to_numeric(series, errors='raise')
        # If values are very large (ms epoch for year 2000+ is > 9e11)
        if numeric_series.min() > 1e11:
            parsed = pd.to_datetime(numeric_series, unit="ms", utc=True)
            return parsed
    except (ValueError, TypeError):
        pass

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


def standardise_csv(entry: dict) -> pd.DataFrame:
    """Load a single logger CSV, unpivot inverter arrays, and reindex to 15-min.

    Args:
        entry: Dict from :func:`discover_csvs` with ``path`` and identity cols.

    Returns:
        DataFrame in long format (one row per timestamp per extracted inverter).
    """
    path: Path = entry["path"]
    logger_id = entry["inverter_id"] # the auto-mapper assigned this as INV_XXX, we'll treat as LOGGER_ID
    plant_id = entry["plant_id"]
    block_id = entry["block_id"]

    log.info("parsing_datalogger_csv", file=path.name, logger_id=logger_id)
    df = pd.read_csv(path)

    # 1. Identify timestamp column
    ts_col = None
    for col in df.columns:
        if "time" in col.lower() or "date" in col.lower():
            ts_col = col
            break
    if ts_col is None:
        ts_col = df.columns[0]

    parsed = _try_parse_timestamps(df[ts_col])
    if parsed is None:
        log.error("timestamp_parse_failed", file=path.name)
        return pd.DataFrame()

    df["timestamp"] = parsed
    
    # 2. Extract Common/Meter Fields
    meter_mappings = {
        "meter_active_power": ["meter_active_power"],
        "meter_pf": ["meter_pf", "meters[0].pf"],
        "meter_freq": ["meter_freq", "meters[0].freq"],
        "meter_v_r": ["meters[0].v_r"],
        "meter_v_y": ["meters[0].v_y"],
        "meter_v_b": ["meters[0].v_b"],
    }
    
    common_data = {"timestamp": df["timestamp"]}
    for canon, patterns in meter_mappings.items():
        common_data[canon] = np.nan
        for p in patterns:
            for c in df.columns:
                if p.lower() in c.lower():
                    common_data[canon] = df[c]
                    break
    
    # 3. Discover Inverter Indices
    import re
    inv_indices = set()
    for col in df.columns:
        match = re.search(r'inverters\[(\d+)\]', col, re.IGNORECASE)
        if match:
            inv_indices.add(int(match.group(1)))
            
    if not inv_indices:
        # Fallback to single inverter if no arrays
        inv_indices = {0}
            
    # 4. Extract per-inverter data
    inv_frames = []
    
    # The mapper rules for each inverter
    inv_rules = {
        "pv1_current": ["pv1_current", "pv1_curr"],
        "pv1_voltage": ["pv1_voltage", "pv1_volt"],
        "pv1_power": ["pv1_power"],
        "pv2_current": ["pv2_current", "pv2_curr"],
        "pv2_voltage": ["pv2_voltage", "pv2_volt"],
        "pv2_power": ["pv2_power"],
        "inverter_temperature": ["temp", "temperature"],
        "inverter_alarm_code": ["alarm_code"],
        "inverter_op_state": ["op_state"],
        "inverter_limit_percent": ["limit_percent"],
    }
    
    # For strings, we'll just map up to 12
    string_types = ["pv", "string", "smu_string"]
    for i in range(1, 13):
        # We'll map inv_stringX
        inv_rules[f"inv_string{i}"] = [f"{st}{i}" for st in string_types]
        
    for idx in sorted(list(inv_indices)):
        # Base dataframe with common meter data
        idf = pd.DataFrame(common_data).copy()
        
        # Inverter ID: logger_id + _00
        # Wait, the prompt said: LOGGER_A + _00 -> INV_A_00
        # If logger_id is INV_001, we want INV_001_00
        actual_inv_id = f"{logger_id}_{idx:02d}"
        idf["inverter_id"] = actual_inv_id
        idf["plant_id"] = plant_id
        idf["block_id"] = block_id
        
        prefix = f"inverters[{idx}]."
        
        # Populate specific inverter features
        for canon, patterns in inv_rules.items():
            idf[canon] = np.nan
            for p in patterns:
                # 1. Try exact exact prefix match
                target_col = f"{prefix}{p}"
                
                # Check case sensitive mostly, or loop
                found = None
                for c in df.columns:
                    if c.lower() == target_col.lower():
                        found = c
                        break
                        
                # If not found with prefix, and this is idx 0, maybe it's globally defined
                if not found and idx == 0:
                     for c in df.columns:
                         if c.lower() == p.lower():
                             found = c
                             break
                             
                if found:
                    idf[canon] = df[found]
                    break
                    
        # Check if this inverter actually has any data (don't add empty ghost inverters)
        # If all critical fields are null across all rows, skip
        vital_cols = ["pv1_power", "inverter_temperature", "pv1_current"]
        has_data = False
        for vc in vital_cols:
            if vc in idf.columns and idf[vc].notna().sum() > 0:
                has_data = True
                break
                
        if has_data:
            inv_frames.append(idf)
            
    if not inv_frames:
        log.warning("no_inverters_extracted", file=path.name)
        return pd.DataFrame()
        
    master_inv_df = pd.concat(inv_frames, ignore_index=True)

    # 5. Add any missing canonical columns
    for canon in CANONICAL_COLUMNS:
        if canon not in master_inv_df.columns:
            master_inv_df[canon] = np.nan
            
    keep = ["timestamp", "inverter_id", "plant_id", "block_id"] + [c for c in CANONICAL_COLUMNS if c != "timestamp"]
    master_inv_df = master_inv_df[[c for c in keep if c in master_inv_df.columns]].copy()

    # 6. Sort and reindex to 15-min per inverter
    master_inv_df.sort_values(["inverter_id", "timestamp"], inplace=True)
    
    reindexed_frames = []
    for inv_id, grp in master_inv_df.groupby("inverter_id"):
        grp = grp.drop_duplicates(subset=["timestamp"], keep="first")
        grp.set_index("timestamp", inplace=True)
        
        full_idx = pd.date_range(
            start=grp.index.min(), 
            end=grp.index.max(), 
            freq=f"{config.TELEMETRY_INTERVAL_MINUTES}min", 
            tz="UTC"
        )
        
        grp = grp.reindex(full_idx)
        grp.index.name = "timestamp"
        grp["inverter_id"] = inv_id
        # Forward fill plant and block
        grp["plant_id"] = grp["plant_id"].ffill().bfill()
        grp["block_id"] = grp["block_id"].ffill().bfill()
        grp.reset_index(inplace=True)
        reindexed_frames.append(grp)

    final_df = pd.concat(reindexed_frames, ignore_index=True)
    
    log.info(
        "datalogger_standardised",
        logger_id=logger_id,
        inverters_extracted=len(reindexed_frames),
        rows=len(final_df),
    )
    return final_df


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

    # Coerce all data columns to numeric to avoid pyarrow object type errors
    # caused by dirty sensor data containing strings like "50.12590-"
    for col in config.CANONICAL_COLUMNS:
        if col in master.columns and col != "timestamp":
            master[col] = pd.to_numeric(master[col], errors='coerce')

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
# Step 5 — Predictive Labelling
# =====================================================================

def _identify_fault_class(df: pd.DataFrame) -> pd.Series:
    """Identify the multiclass fault label for each row heuristically."""
    # 0: Normal, 1: Thermal, 2: String Mismatch, 3: Grid Instability, 4: Cooling Failure
    
    # Extract needed columns, filling missing with safe defaults
    temp = df.get("inverter_temperature", pd.Series(0, index=df.index)).astype(float)
    power = df.get("pv1_power", pd.Series(0, index=df.index)).astype(float)
    freq = df.get("meter_freq", pd.Series(50.0, index=df.index)).astype(float)
    
    # We need string variance if calculated early, otherwise we use raw string logic
    # In ingest_raw, string_mismatch_std is NOT calculated yet. Features are computed LATER (in Layer 2).
    # Wait, apply_predictive_labels runs ON ingest, BEFORE feature engineering pipeline!
    # I need to calculate basic heuristics for the labels directly or use an alarm code.
    # The user asked for `if cooling_failure_condition: label = 4...`
    # Let's define simple thresholds for the raw sensor data available in master:
    # Cooling failure: Extremely high temp (e.g. > 85) while power is low
    cooling_fail = (temp > 85) & (power < 1000)
    # Thermal issue: High temp (e.g. > 70)
    thermal = (temp > 70)
    # String Mismatch: if we have string columns, check max - min
    string_cols = [c for c in df.columns if "string" in c.lower() or "pv" in c.lower() and "power" not in c.lower()]
    string_mismatch = pd.Series(False, index=df.index)
    if string_cols:
        string_data = df[string_cols].astype(float)
        string_mismatch = (string_data.max(axis=1) - string_data.min(axis=1)) > 5.0 # arbitrary threshold
        
    # Grid instability: Frequency deviation > 0.5 Hz or Voltage > 250 or < 200
    grid_instability = (freq > 50.5) | (freq < 49.5)
    v_cols = [c for c in ["meter_v_r", "meter_v_y", "meter_v_b"] if c in df.columns]
    if v_cols:
        v_data = df[v_cols].astype(float)
        grid_instability = grid_instability | (v_data > 250.0).any(axis=1) | (v_data < 200.0).any(axis=1)
        
    # Also include the original alarm code as a fallback cooling/thermal/general fault
    has_alarm = df["inverter_alarm_code"].fillna(0) != 0

    # Apply deterministic priority
    labels = pd.Series(0, index=df.index)
    labels.mask(grid_instability, 3, inplace=True)
    labels.mask(string_mismatch, 2, inplace=True)
    labels.mask(thermal, 1, inplace=True)
    labels.mask(cooling_fail, 4, inplace=True)
    # If there's an alarm but none of the above caught it, default to thermal (1) or something else?
    # Let's map alarm_code to 1 if label is still 0
    labels.mask((labels == 0) & has_alarm, 1, inplace=True)
    
    return labels

def apply_predictive_labels(master: pd.DataFrame) -> pd.DataFrame:
    """Implement multiclass predictive labeling.
    
    Label represents the highest priority fault occurring within the next 24 hours.
    """
    master["timestamp"] = pd.to_datetime(master["timestamp"], utc=True)
    master = master.sort_values(["inverter_id", "timestamp"]).copy()
    
    processed_groups = []
    
    for inv_id, grp in master.groupby("inverter_id"):
        grp = grp.sort_values("timestamp").copy()
        
        # 1. Identify active fault classes per row
        current_class = _identify_fault_class(grp)
        is_faulty = current_class > 0
        
        # 2. Define prediction target: what is the worst fault in next 24h?
        # Look ahead window: 24h / 15min = 96 samples
        lookahead = 24 * 4 # samples
        
        # For multiclass, we roll and take the MAX class because:
        # Cooling (4) > Grid (3) > String (2) > Thermal (1)
        # However, the requested priority is: 4 > 1 > 2 > 3 > 0
        # This means simple maximum won't respect the exact priority if we just use max().
        # Let's map to a priority order, take max rolling, and map back.
        # Original: 0=N, 1=T, 2=S, 3=G, 4=C
        # Priority: 4 > 1 > 2 > 3 > 0
        priority_map = {0: 0, 3: 1, 2: 2, 1: 3, 4: 4}
        reverse_map = {0: 0, 1: 3, 2: 2, 3: 1, 4: 4}
        
        priority_series = current_class.map(priority_map).fillna(0)
        future_priority = priority_series.rolling(window=lookahead, min_periods=1).max().shift(-lookahead).fillna(0)
        future_fault_class = future_priority.map(reverse_map).fillna(0).astype(int)
        
        grp["will_fail_24h"] = future_fault_class
        grp["is_currently_faulty"] = is_faulty
        
        # 3. Filter out rows where ANY fault is already active
        grp = grp[~is_faulty].copy()
        
        processed_groups.append(grp)
        log.info("labels_applied_to_inverter", inverter_id=inv_id, rows_remaining=len(grp))

    if not processed_groups:
        return master

    labelled_master = pd.concat(processed_groups, ignore_index=True)
    
    # Standardize label column name
    labelled_master["label"] = labelled_master["will_fail_24h"]
    labelled_master["label_source"] = "predictive_24h"
    
    return labelled_master



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
    master = apply_predictive_labels(master)

    # Save labelled master
    out_path = config.PROCESSED_DIR / "master_labelled.parquet"
    master.to_parquet(out_path, index=False, engine="pyarrow")
    log.info("master_labelled_saved", path=str(out_path), rows=len(master))

    # Print stats
    print_label_statistics(master)

    log.info("ingestion_completed")


if __name__ == "__main__":
    main()
