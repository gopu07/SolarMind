"""
Tests for Labelling (Layer 1 — hybrid labelling sub-system).
"""

import pandas as pd
import numpy as np
import pytest

import config
from scripts.ingest_raw import apply_labels, _apply_hard_labels


def _make_master(n_rows=20):
    """Helper: create a minimal master telemetry DataFrame."""
    return pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=n_rows, freq="15min", tz="UTC"),
        "inverter_id": "INV_1",
        "plant_id": "P1",
        "block_id": "B1",
        "pv1_power": [1000.0] * n_rows,
        "pv2_power": [1000.0] * n_rows,
        "internal_temp": [30.0] * n_rows,
        "ambient_temp": [25.0] * n_rows,
        "inverter_temperature": [30.0] * n_rows,
        "meter_active_power": [950.0] * n_rows,
    })


def test_hard_labels_override_inferred():
    """Hard labels from events file must set label=1 using the lookback window."""
    master = _make_master()
    # The event falls within the window of all 20 rows (they are close in time)
    events = pd.DataFrame({
        "inverter_id": ["INV_1"],
        "event_timestamp": pd.to_datetime(["2024-01-01 05:00:00"], utc=True),
        "event_type": ["failure"],
    })

    labelled = _apply_hard_labels(master, events)
    assert (labelled["label"] == 1).any(), "At least some rows should have label=1 from hard label"


def test_label_window_is_10_days():
    """LABEL_WINDOW_DAYS config constant must be exactly 10."""
    assert config.LABEL_WINDOW_DAYS == 10


def test_label_source_always_populated():
    """label_source column must never contain null values after apply_labels."""
    master = _make_master()
    labelled = apply_labels(master)
    assert "label_source" in labelled.columns
    assert labelled["label_source"].notna().all(), "label_source must never be null"
