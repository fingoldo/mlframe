"""Identity regression for the _copula_mi_batch y-rank hoist (CPX11).

Hoisting v_full = _rank_to_uniform(y) above the per-column loop must not change
any score vs re-ranking y inside the loop. Covers all-finite y and y/column
NaN masks (which still re-rank the masked subset).
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_selection.filters._orthogonal_copula_mi_fe import (
    _bin_mi_uniform_pair,
    _copula_mi_batch,
    _rank_to_uniform,
)


def _reference(X, y, n_bins=20):
    """Helper that reference."""
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    y = np.asarray(y, dtype=np.float64).ravel()
    yf = np.isfinite(y)
    out = np.empty(X.shape[1])
    for j in range(X.shape[1]):
        col = X[:, j]
        fin = yf & np.isfinite(col)
        if fin.all():
            u = _rank_to_uniform(col)
            v = _rank_to_uniform(y)  # re-ranked per column (pre-hoist)
        elif fin.sum() >= 2:
            u = _rank_to_uniform(col[fin])
            v = _rank_to_uniform(y[fin])
        else:
            out[j] = 0.0
            continue
        out[j] = _bin_mi_uniform_pair(u, v, n_bins=n_bins)
    return out


@pytest.mark.parametrize("seed", [0, 1, 4])
@pytest.mark.parametrize("with_nan", [False, True])
def test_copula_batch_hoist_identical(seed, with_nan):
    """Copula batch hoist identical."""
    rng = np.random.default_rng(seed)
    n, p = 800, 25
    X = rng.standard_normal((n, p))
    y = rng.standard_normal(n)
    if with_nan:
        X[rng.integers(0, n, 30), rng.integers(0, p, 30)] = np.nan
        y[rng.integers(0, n, 10)] = np.nan
    ref = _reference(X, y)
    got = _copula_mi_batch(X, y)
    assert np.array_equal(ref, got)
