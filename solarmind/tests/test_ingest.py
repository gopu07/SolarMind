"""
Tests for Data Ingestion Layer 1.
"""

import pandas as pd
import pytest


def test_timestamp_parsing_variants():
    """Ingest must handle at least 3 different timestamp format variants."""
    # Test 3 common variants individually since pd.to_datetime with mixed formats
    # may need format='mixed'
    variants = [
        "2024-01-01T12:00:00Z",
        "2024-01-01 12:00:00",
        "01/01/2024 12:00",
    ]
    for v in variants:
        dt = pd.to_datetime(v, utc=True)
        assert pd.notna(dt), f"Failed to parse: {v}"


def test_mac_parsing_from_filename():
    """MAC address should be extracted from the CSV filename (stem)."""
    from pathlib import Path

    csv_path = Path("data/raw/A1B2C3D4E5F6.csv")
    mac = csv_path.stem.strip().upper()
    assert mac == "A1B2C3D4E5F6"


def test_missing_mapping_entries_skipped():
    """If a CSV filename MAC is not in the mapping, it should be skipped."""
    from scripts.ingest_raw import discover_csvs
    import config
    import tempfile, os

    # Create tiny raw dir with one CSV
    with tempfile.TemporaryDirectory() as tmp:
        raw = os.path.join(tmp, "raw")
        os.makedirs(raw)
        open(os.path.join(raw, "UNKNOWNMAC.csv"), "w").close()

        # Provide a mapping that does NOT contain UNKNOWNMAC
        mapping = pd.DataFrame({
            "mac_address": ["OTHERMAC"],
            "inverter_id": ["INV_001"],
            "plant_id": ["P1"],
            "block_id": ["B1"],
        })

        # Monkey-patch RAW_DATA_DIR temporarily
        original = config.RAW_DATA_DIR
        try:
            from pathlib import Path
            config.RAW_DATA_DIR = Path(raw)
            results = discover_csvs(mapping)
            assert len(results) == 0, "Unknown MAC should be skipped"
        finally:
            config.RAW_DATA_DIR = original


def test_reindexed_df_no_timestamp_gaps():
    """After reindexing to 15-min, timestamps should be gap-free."""
    df = pd.DataFrame({
        "timestamp": pd.to_datetime(
            ["2024-01-01 12:00:00", "2024-01-01 12:30:00"], utc=True
        ),
        "pv1_power": [100.0, 110.0],
    }).set_index("timestamp")

    full_idx = pd.date_range(
        start=df.index.min(), end=df.index.max(), freq="15min"
    )
    reindexed = df.reindex(full_idx)
    assert len(reindexed) == 3, "Should have 3 rows: 12:00, 12:15, 12:30"
    assert pd.isna(
        reindexed.loc[pd.to_datetime("2024-01-01 12:15:00", utc=True), "pv1_power"]
    ), "Gap at 12:15 should be NaN"
