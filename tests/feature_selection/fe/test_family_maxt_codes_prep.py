"""The block conversion of ``_family_codes`` equals the per-column preparation it replaced, for finite and non-finite columns, ndarray and DataFrame."""

from __future__ import annotations

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters import _family_maxt_gate as G


def _reference(X, n_bins):
    """The previous per-column preparation, then the same coder."""
    n, k = X.shape
    v = np.empty((k, n), dtype=G._crit_np_dtype())
    for j in range(k):
        col = G._column_values(X, j)
        bad = ~np.isfinite(col)
        if bad.any():
            col = col.copy()
            col[bad] = np.median(col[~bad]) if (~bad).any() else 0.0
        v[j] = col
    cols = np.empty((k, n), dtype=np.int16)
    nx = np.empty(k, dtype=np.int64)
    G.quantile_codes_block(v, n_bins, cols, nx)
    return cols, nx


def test_block_preparation_equals_the_per_column_one():
    """NaN, +-inf, an all-NaN column, a constant column and heavy ties: same codes and counts as the per-column path."""
    rng = np.random.default_rng(8)
    n, k = 2000, 12
    a = rng.standard_normal((n, k)).astype(np.float32)
    a[::7, 2] = np.nan
    a[3::11, 4] = np.inf
    a[5::13, 5] = -np.inf
    a[:, 6] = np.nan
    a[:, 7] = 1.5
    a[:, 8] = np.round(a[:, 8], 1)
    for X in (pd.DataFrame(a, columns=[f"c{i}" for i in range(k)]), a, pd.DataFrame(a.astype(np.float64))):
        got = G._family_codes(X, 10)
        want = _reference(X, 10)
        np.testing.assert_array_equal(got[0], want[0])
        np.testing.assert_array_equal(got[1], want[1])
