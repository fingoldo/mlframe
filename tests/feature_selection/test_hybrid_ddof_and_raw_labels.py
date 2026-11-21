"""Correlation clustering uses the exact correlation (audit 2026-11-20, finding 24-F21); cohort drops return the caller's column labels (24-F22)."""

import numpy as np
import pandas as pd

from mlframe.feature_selection.drop_noninformative_vs_reference import drop_noninformative_vs_reference
from mlframe.feature_selection.hybrid_selector import corr_clusters


def _pair_with_correlation(rho: float, n: int) -> pd.DataFrame:
    """Two standardised columns whose Pearson correlation is exactly ``rho``."""
    rng = np.random.default_rng(3)
    x = rng.standard_normal(n)
    x = (x - x.mean()) / x.std()
    e = rng.standard_normal(n)
    e = e - (e @ x) / (x @ x) * x  # orthogonal to x
    e = (e - e.mean()) / e.std()
    y = rho * x + np.sqrt(1.0 - rho * rho) * e
    return pd.DataFrame({"a": x, "b": y})


def test_corr_clusters_does_not_inflate_correlation_by_n_over_n_minus_one():
    """rho=0.9 at n=10 read as 1.0 under the n-1 divisor and merged the pair at thr=0.92; the exact correlation stays apart."""
    X = _pair_with_correlation(0.9, n=10)
    reps, _members = corr_clusters(X, thr=0.92)
    assert len(reps) == 2


def test_cohort_drop_returns_the_callers_non_string_labels():
    """Integer column labels used to come back as strings and never matched the intersection, so nothing was dropped."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.standard_normal((200, 3)), columns=[0, 1, 2])
    mask = np.arange(200) < 100
    result = drop_noninformative_vs_reference(df, mask, feature_cols=[0, 1, 2])
    assert result
    assert all(isinstance(c, (int, np.integer)) for c in result)
