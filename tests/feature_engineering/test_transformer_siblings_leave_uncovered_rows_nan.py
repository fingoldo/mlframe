"""Transformer feature modules leave rows no fold validates on as NaN, not a real-looking 0.0 (ks_shift end to end)."""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import TimeSeriesSplit

from mlframe.feature_engineering.transformer.ks_shift import compute_ks_shift_features


def test_ks_shift_first_timeseries_block_is_nan():
    """TimeSeriesSplit(4) over 100 rows never validates the first 20; those feature values must be NaN."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, 4)).astype(np.float32)
    y = rng.normal(size=100).astype(np.float32)
    df = compute_ks_shift_features(X, y, X_query=None, splitter=TimeSeriesSplit(4), seed=0, k=8, task="regression")
    col = df["ksshift_mean_shift"].to_numpy()
    assert np.isnan(col[:20]).all() and np.isfinite(col[20:]).all()
