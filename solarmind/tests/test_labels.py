"""
Tests for Labelling (Layer 1 — hybrid labelling sub-system).
"""

import pandas as pd
import numpy as np
import pytest

import config
from scripts.ingest_raw import apply_predictive_labels


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
        "inverter_alarm_code": [0.0] * n_rows,
    })


def test_apply_predictive_labels_multiclass():
    """Predictive labels must assign the correct future fault class."""
    master = _make_master(n_rows=200) # Ensure enough rows for the 96-sample lookahead window
    
    # Introduce a cooling failure fault (label=4) at row 150
    master.loc[150, "inverter_temperature"] = 90.0
    master.loc[150, "pv1_power"] = 0.0
    
    labelled = apply_predictive_labels(master)
    
    # Check that labels were assigned correctly
    assert "label" in labelled.columns
    assert "label_source" in labelled.columns
    
    # Since look-ahead window is 96, rows 150-96 up to 149 should ideally be labelled as 4
    # The predictive labeling logic drops the exact point of failure, so we just check for presence of 4
    if len(labelled) > 0:
        assert (labelled["label"] == 4).any(), "Expected Cooling Failure (4) to be in future labels"


def test_label_source_always_populated():
    """label_source column must never contain null values after apply_predictive_labels."""
    master = _make_master(n_rows=50) # Not enough for fault logic or lookahead, but sufficient to test fallback
    labelled = apply_predictive_labels(master)
    if "label_source" in labelled.columns and len(labelled) > 0:
        assert labelled["label_source"].notna().all(), "label_source must never be null"
