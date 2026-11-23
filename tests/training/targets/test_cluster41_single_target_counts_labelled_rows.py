"""Cluster 41 regression: single-target bin counts exclude unlabelled rows (audit 2026-11-20, finding 41-F9)."""

import numpy as np
import pandas as pd

from mlframe.training.targets._target_temporal_audit_aggregate import _aggregate_by_time_pandas


def test_pandas_single_path_counts_only_labelled_rows():
    """Two rows in one day, one unlabelled: n_obs is 1, matching the multi-target path (41-F9)."""
    df = pd.DataFrame({
        "ts": pd.to_datetime(["2024-01-01", "2024-01-01", "2024-01-02"]),
        "y": [1.0, np.nan, 0.0],
    })
    agg = _aggregate_by_time_pandas(df, "ts", "y", "day", target_type="binary_classification")
    assert agg.loc[agg["bin_start"] == pd.Timestamp("2024-01-01"), "n_obs"].item() == 1
