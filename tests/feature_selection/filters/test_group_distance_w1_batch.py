"""The opt-in batched Wasserstein-1 of the group-distance family gives the per-group path's columns exactly."""

from __future__ import annotations

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters._group_distance_fe import generate_group_distance_features


def test_batched_w1_equals_per_group(monkeypatch):
    """Two group columns, numeric columns with ties and missing values."""
    rng = np.random.default_rng(3)
    n = 6000
    X = pd.DataFrame({"g": rng.integers(0, 7, n), "h": rng.integers(0, 3, n), "a": rng.standard_normal(n), "b": np.round(rng.exponential(2, n), 1), "c": rng.standard_normal(n)})
    X.loc[::31, "c"] = np.nan
    monkeypatch.setenv("MLFRAME_GROUP_DISTANCE_W1_BATCH", "0")
    want, _ = generate_group_distance_features(X, ["g", "h"], ["a", "b", "c"])
    monkeypatch.setenv("MLFRAME_GROUP_DISTANCE_W1_BATCH", "1")
    got, _ = generate_group_distance_features(X, ["g", "h"], ["a", "b", "c"])
    pd.testing.assert_frame_equal(got, want)
