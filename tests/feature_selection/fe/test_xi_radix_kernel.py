"""The radix Xi kernel orders the rows as ``(value, tie key, row)`` and returns the comparison-sort kernel's values."""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_selection.filters._orthogonal_xi_fe import _xi_columns_njit, _xi_columns_radix_njit, column_seeds


def _inputs(n: int, rng):
    """Continuous, tied, few-level, signed-zero and negative columns, their tie seeds and the target's tie-corrected ranks."""
    signed_zero = np.where(rng.random(n) < 0.5, -0.0, 0.0) + np.round(rng.standard_normal(n))
    cols = np.stack(
        [rng.standard_normal(n), np.round(rng.standard_normal(n) * 3), rng.integers(0, 4, n).astype(float), signed_zero, -np.abs(rng.standard_normal(n)), np.full(n, 2.5)]
    ).astype(np.float32)
    perms = column_seeds(3, cols.shape[0])
    y = rng.integers(0, 3, n).astype(np.float64)
    _u, inv, counts = np.unique(y, return_counts=True, return_inverse=True)
    r_row = np.cumsum(counts)[inv].astype(np.float64)
    l_row = np.cumsum(counts[::-1])[::-1][inv].astype(np.float64)
    return np.ascontiguousarray(cols), perms, r_row, l_row


@pytest.mark.parametrize("n", [2, 17, 5000])
def test_radix_kernel_equals_comparison_kernel(n):
    """Bit-identical Xi for every column shape, including ties broken by the tie keys and -0.0 equal to +0.0."""
    cols, perms, r_row, l_row = _inputs(n, np.random.default_rng(n))
    want = np.empty(cols.shape[0])
    got = np.empty(cols.shape[0])
    _xi_columns_njit(cols, perms, r_row, l_row, want)
    _xi_columns_radix_njit(cols, perms, r_row, l_row, got)
    assert np.array_equal(got, want)
