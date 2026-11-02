"""The pipelined permutation draw of the xi batch leaves the values and the generator stream unchanged."""

from __future__ import annotations

import numpy as np

from mlframe.feature_selection.filters._fe_usability_signal import _crit_np_dtype
from mlframe.feature_selection.filters._orthogonal_xi_fe import _xi_columns_njit, xi_correlation_batch


def test_batch_equals_a_serial_draw_over_many_blocks():
    """More columns than one block: the values equal those of one permutation per column drawn serially, in column order, from a single generator."""
    rng = np.random.default_rng(2)
    n, k = 600, 40
    X = rng.standard_normal((n, k))
    X[:, 5] = np.round(X[:, 5], 1)
    y = (X[:, 0] + rng.standard_normal(n) > 0).astype(np.int64)
    got = xi_correlation_batch(X, y, random_state=7)
    gen = np.random.default_rng(7)
    yf = y.astype(np.float64)
    _, inv, counts = np.unique(yf, return_counts=True, return_inverse=True)
    r_row = np.cumsum(counts)[inv].astype(np.float64)
    l_row = np.cumsum(counts[::-1])[::-1][inv].astype(np.float64)
    want = np.empty(k)
    Xc = np.asarray(X, dtype=_crit_np_dtype())  # the batch reads its columns at the criterion dtype
    for j in range(k):
        perm = gen.permutation(n)[None, :].astype(np.int64)
        out = np.empty(1)
        _xi_columns_njit(np.ascontiguousarray(Xc[:, j])[None, :], perm, r_row, l_row, out)
        want[j] = out[0]
    np.testing.assert_allclose(got, want, rtol=0, atol=1e-6)
