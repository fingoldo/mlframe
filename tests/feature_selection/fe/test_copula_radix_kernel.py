"""The radix copula-MI kernel returns the mergesort kernel's values for float32 columns of every shape."""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_selection.filters._orthogonal_copula_mi_fe import _copula_columns_njit, _copula_columns_radix_njit


@pytest.mark.parametrize("n", [2, 31, 6000])
@pytest.mark.parametrize("nb", [5, 20])
def test_radix_equals_mergesort(n, nb):
    """Bit-identical MI for continuous, tied, few-level, signed-zero, negative and constant float32 columns."""
    rng = np.random.default_rng(n + nb)
    signed_zero = np.where(rng.random(n) < 0.5, -0.0, 0.0) + np.round(rng.standard_normal(n))
    rows = np.ascontiguousarray(
        np.stack([rng.standard_normal(n), np.round(rng.standard_normal(n) * 3), rng.integers(0, 4, n).astype(float), signed_zero, -np.abs(rng.standard_normal(n)), np.full(n, 1.5)]).astype(np.float32)
    )
    vi = rng.integers(0, nb, n).astype(np.int64)
    want = np.empty(rows.shape[0])
    got = np.empty(rows.shape[0])
    _copula_columns_njit(rows, vi, nb, want)
    _copula_columns_radix_njit(rows, vi, nb, got)
    assert np.array_equal(got, want)
