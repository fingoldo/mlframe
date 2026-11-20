"""Regression tests: _cluster_anchors must cluster correlated columns despite NaN cells and large offsets."""

from __future__ import annotations

import numpy as np
import pandas as pd

from mlframe.feature_selection.varying_size_top_k_subsets import _cluster_anchors


def test_cluster_anchors_nan_cell_does_not_split_correlated_pair():
    """A single NaN in one column must not make a perfectly correlated pair fall into separate clusters."""
    rng = np.random.default_rng(0)
    a = rng.normal(size=200)
    df = pd.DataFrame({"a": a, "b": 2.0 * a + 1.0, "c": rng.normal(size=200)})
    df.loc[5, "a"] = np.nan
    clusters = _cluster_anchors(["a", "b", "c"], df, 0.8)
    assert clusters["a"] == ["a", "b"]


def test_cluster_anchors_large_offset_columns_are_still_correlated():
    """Columns with a 1e8 offset and unit spread must keep their correlation (float32 narrowing would flatten them)."""
    rng = np.random.default_rng(1)
    z = rng.normal(size=300)
    df = pd.DataFrame({"a": 1e8 + z, "b": 1e8 + 3.0 * z + 0.01 * rng.normal(size=300), "c": rng.normal(size=300)})
    clusters = _cluster_anchors(["a", "b", "c"], df, 0.8)
    assert clusters["a"] == ["a", "b"]
