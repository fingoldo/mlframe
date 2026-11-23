"""Cluster 15 regression: an all-NaN feature gets a finite knockoff, not NaN (audit 2026-11-20, finding 15-F17)."""

import numpy as np

from mlframe.feature_selection.wrappers._knockoffs import make_gaussian_knockoffs


def test_all_nan_column_knockoff_is_finite():
    """The knockoff of an all-NaN column is finite; the other columns stay finite too (15-F17)."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 3))
    X[:, 1] = np.nan
    out = make_gaussian_knockoffs(X, random_state=0)
    assert np.all(np.isfinite(out))


def test_clean_matrix_still_produces_finite_knockoffs():
    """Control: an ordinary matrix is unaffected (15-F17)."""
    X = np.random.default_rng(1).normal(size=(40, 3))
    assert np.all(np.isfinite(make_gaussian_knockoffs(X, random_state=1)))
