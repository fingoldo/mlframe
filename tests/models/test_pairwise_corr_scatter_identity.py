"""Identity regression for the vectorised ``np.ix_`` scatter in ``base._pairwise_corr_or_nan``.

The NaN-padded correlation-matrix scatter was changed from an O(K^2) Python double loop to a single
``np.ix_`` block assignment. This pins that the optimized helper reproduces the reference double-loop
scatter EXACTLY (it is a pure index reshuffle -- bit-identical by construction), including the
NaN-padded skipped-member rows when ``return_full_shape=True``.
"""
from __future__ import annotations

import numpy as np
import pytest

from mlframe.models.ensembling.base import _pairwise_corr_or_nan


def _reference_full_scatter(corr_used, idx_use, original_k, K_use):
    out = np.full((original_k, original_k), np.nan, dtype=np.float64)
    for ii in range(K_use):
        for jj in range(K_use):
            out[idx_use[ii], idx_use[jj]] = corr_used[ii, jj]
    return out


@pytest.mark.parametrize("K", [3, 5, 10])
def test_pairwise_corr_all_finite_matches_corrcoef(K):
    rng = np.random.default_rng(K)
    M_stack = rng.normal(size=(K, 200))
    out = _pairwise_corr_or_nan(M_stack)
    assert out is not None
    assert out.shape == (K, K)
    # No skipped members (all non-constant, all finite) -> diagonal is 1.0, no NaN pads.
    expected = np.corrcoef(M_stack)
    np.testing.assert_allclose(out, expected, rtol=0, atol=0)


def test_pairwise_corr_nan_padded_skips_constant_member():
    # Member 1 is constant (std==0) -> dropped; its row/col must be NaN-padded, the rest matches.
    rng = np.random.default_rng(7)
    K = 4
    M_stack = rng.normal(size=(K, 150))
    M_stack[1, :] = 3.0  # constant member -> skipped by the std>0 mask
    out = _pairwise_corr_or_nan(M_stack, return_full_shape=True, original_k=K)
    assert out.shape == (K, K)

    # Build the reference via the kept submatrix + explicit double-loop scatter.
    nonconst = M_stack.std(axis=1) > 0
    idx_use = np.flatnonzero(nonconst)
    corr_used = np.corrcoef(M_stack[nonconst])
    expected = _reference_full_scatter(corr_used, idx_use, K, idx_use.shape[0])

    np.testing.assert_array_equal(np.isnan(out), np.isnan(expected))
    finite = ~np.isnan(expected)
    np.testing.assert_allclose(out[finite], expected[finite], rtol=0, atol=0)
    # Skipped member's entire row + column are NaN.
    assert np.all(np.isnan(out[1, :]))
    assert np.all(np.isnan(out[:, 1]))
