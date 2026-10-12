"""A snapshot whose rows are all NaN in a feature must not break the neighbor search.

The per-snapshot mean is NaN for such a cell, and the kNN search rejects NaN input. The fuzz combo c0013 lost its ``xsnn_*`` columns on
the validation split this way: the failure was swallowed upstream, so the split silently kept its raw schema.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mlframe.feature_engineering import compute_cross_sectional_neighbor_features


def _frame_with_all_nan_snapshot() -> pd.DataFrame:
    """Four snapshots of 5 rows each; snapshot 2 is entirely NaN in ``f0``."""
    rng = np.random.default_rng(0)
    n_snap, per = 4, 5
    snap = np.repeat(np.arange(n_snap), per)
    f0 = rng.normal(size=n_snap * per)
    f1 = rng.normal(size=n_snap * per)
    f0[snap == 2] = np.nan
    return pd.DataFrame({"snap": snap, "f0": f0, "f1": f1})


def test_all_nan_snapshot_feature_does_not_raise():
    """The neighbor features compute and keep one row per input row."""
    df = _frame_with_all_nan_snapshot()

    out = compute_cross_sectional_neighbor_features(df, snapshot_col="snap", feature_cols=["f0", "f1"], k=2)

    assert out.shape[0] == len(df)


def test_all_nan_snapshot_feature_yields_finite_distance_ratio():
    """The all-NaN snapshot gets a finite distance ratio, not a NaN propagated from its missing mean."""
    df = _frame_with_all_nan_snapshot()

    out = compute_cross_sectional_neighbor_features(df, snapshot_col="snap", feature_cols=["f0", "f1"], k=2).to_pandas()

    ratio = out["xsnn_distance_ratio"].to_numpy()
    assert np.isfinite(ratio).all()
