"""CorrelatedFeaturesSelector bridges a polars frame without copying its numeric columns, and clusters it like the pandas frame."""

from __future__ import annotations

import numpy as np
import polars as pl
from sklearn.base import BaseEstimator

from mlframe.feature_selection.filters.correlated_features import CorrelatedFeaturesSelector


class _Spy(BaseEstimator):
    """Inner selector that records the frame it is fitted on and keeps every column."""

    seen: list = []

    def fit(self, X, y=None, **kwargs):
        """Record X and select everything."""
        _Spy.seen.append(X)
        self.support_ = np.arange(X.shape[1])
        self.n_features_ = X.shape[1]
        return self


def _frame():
    """Two near-duplicate columns, an independent float column, an integer pair and a categorical."""
    rng = np.random.default_rng(0)
    n = 5000
    a = rng.normal(size=n)
    i = rng.integers(0, 40, n)
    return pl.DataFrame(
        {"a": a, "a2": a + rng.normal(0, 0.02, n), "b": rng.normal(size=n), "i": i, "i2": i + rng.integers(0, 2, n), "c": rng.choice(["x", "y", "z"], n)}
    ).with_columns(pl.col("c").cast(pl.Categorical))


def test_full_frame_handed_to_the_inner_selector_shares_memory_with_the_polars_frame():
    """When the reduction is below ``min_reduction`` the inner selector gets the bridged frame itself, whose numeric buffers must be the polars ones."""
    df = _frame()
    _Spy.seen = []
    CorrelatedFeaturesSelector(estimator=_Spy(), corr_threshold=0.9, min_reduction=0.99).fit(df, np.zeros(len(df)))
    full = _Spy.seen[-1]
    assert full.shape[1] == df.width
    assert np.shares_memory(full["b"].to_numpy(), df["b"].to_numpy())


def test_polars_and_pandas_fits_pick_the_same_clusters():
    """Bridging does not change the clustering or the selected columns."""
    df = _frame()
    sel_pl = CorrelatedFeaturesSelector(estimator=_Spy(), corr_threshold=0.9).fit(df, np.zeros(len(df)))
    sel_pd = CorrelatedFeaturesSelector(estimator=_Spy(), corr_threshold=0.9).fit(df.to_pandas(), np.zeros(len(df)))
    assert list(sel_pl.cluster_assignments_) == list(sel_pd.cluster_assignments_)
    assert list(sel_pl.support_) == list(sel_pd.support_)
