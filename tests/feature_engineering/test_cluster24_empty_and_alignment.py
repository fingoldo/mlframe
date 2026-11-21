"""Cluster 24 regressions: an empty support returns NaN per query point, and reindexed MA differences keep rows aligned (audit 2026-11-20, findings 24-F25, 24-F36)."""

import numpy as np
import pandas as pd

from mlframe.feature_engineering.nadaraya_watson import nadaraya_watson_smooth


def test_empty_support_gives_nan_per_query_point():
    """An empty x used to return length 0 even with query points (24-F25)."""
    out = nadaraya_watson_smooth(np.array([]), np.array([]), x_query=np.array([1.0, 2.0]))
    assert out.shape == (2,)
    assert np.isnan(out).all()


def test_ma_crossover_keeps_rows_aligned_when_the_index_is_reordered():
    """A reordered long average used to shift the difference against the output index (24-F36)."""
    from mlframe.feature_engineering.ma_crossover import ma_crossover_features

    idx = pd.RangeIndex(6)
    short = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], index=idx)
    long_ = pd.Series([0.5, 1.5, 2.5, 3.5, 4.5, 5.5], index=idx[::-1])  # label 0 sits at the end of this series
    out = ma_crossover_features({2: short, 4: long_})
    diff_col = next(c for c in out.columns if "diff" in c)
    # label 0: short=1.0, long(label 0)=5.5 -> diff -4.5
    assert np.isclose(float(np.asarray(out[diff_col])[0]), 1.0 - 5.5)
