"""The temporal target audit treats a missing label as missing, not as a value.

The suite builds the audit frame from numpy arrays, so a missing label arrives as a float NaN, which polars keeps
distinct from null: ``NaN > 0`` counted a missing binary label as positive, a NaN regression value made the bin mean
NaN, and every target's bin size counted the rows it has no label for.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import polars as pl

from mlframe.training.targets._target_temporal_audit_aggregate import _aggregate_by_time_polars_multi


def test_rates_and_bin_sizes_use_labelled_rows_only():
    """Per-period rates and bin sizes count labelled rows only, so missing labels neither dilute nor inflate them."""
    day = [dt.datetime(2026, 1, 1)] * 4 + [dt.datetime(2026, 1, 2)] * 4
    hired = np.array([1, 0, np.nan, np.nan, 1, 1, 0, 0], dtype=float)
    charge = np.array([10, 20, np.nan, np.nan, 5, 5, 5, 5], dtype=float)
    df = pl.DataFrame({"ts": day, "hired": hired, "charge": charge})
    agg = _aggregate_by_time_polars_multi(df, "ts", [("hired", "binary_classification", "r_h"), ("charge", "regression", "r_c")], "day")
    first = agg.iloc[0]
    assert first["r_h"] == 0.5, "two labelled rows, one positive; the NaN rows are neither positive nor counted"
    assert first["r_c"] == 15.0
    assert first["n_obs__r_h"] == 2 and first["n_obs__r_c"] == 2 and first["n_obs"] == 4
