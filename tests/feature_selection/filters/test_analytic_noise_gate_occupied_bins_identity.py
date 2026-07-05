"""Identity pins for the fused occupied-bin count in ``analytic_batch_noise_gate``.

The per-column ``np.unique(disc_2d[:, k]).size`` loop (O(K * n log n) sort) was replaced by a single
fused O(n*K) njit pass (``_occupied_bins_per_col``). These pin that the occupied-bin counts -- which
drive each candidate's chi2 df, hence the keep/reject decision -- are BIT-IDENTICAL to np.unique, and
that the end-to-end gate output is unchanged. A future "just use a faster approximate count" cannot
slip through unnoticed.
"""
from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_selection.filters._analytic_mi_null import (
    _occupied_bins_per_col,
    analytic_batch_noise_gate,
)


def _np_unique_counts(disc):
    return np.array([int(np.unique(disc[:, k]).size) for k in range(disc.shape[1])], dtype=np.int64)


@pytest.mark.parametrize("nbins", [2, 5, 8, 16])
@pytest.mark.parametrize("dtype", [np.int16, np.int32, np.int64])
def test_occupied_bins_matches_np_unique(nbins, dtype):
    rng = np.random.default_rng(nbins)
    n, K = 4000, 25
    disc = rng.integers(0, nbins, size=(n, K)).astype(dtype)
    assert np.array_equal(_occupied_bins_per_col(disc), _np_unique_counts(disc))


def test_occupied_bins_sparse_and_unoccupied_columns():
    # columns with gaps (not every code in [0, max] occupied) + a constant column + a single-row col.
    disc = np.array(
        [[0, 5, 3, 0],
         [2, 5, 3, 0],
         [0, 5, 7, 0],
         [4, 5, 3, 0]],
        dtype=np.int16,
    )
    # col0 occupies {0,2,4}=3; col1 {5}=1; col2 {3,7}=2; col3 {0}=1.
    assert list(_occupied_bins_per_col(disc)) == [3, 1, 2, 1]
    assert np.array_equal(_occupied_bins_per_col(disc), _np_unique_counts(disc))


def test_occupied_bins_empty_matrix():
    disc = np.zeros((0, 5), dtype=np.int16)
    assert list(_occupied_bins_per_col(disc)) == [0, 0, 0, 0, 0]


def test_gate_output_identical_to_np_unique_path():
    rng = np.random.default_rng(7)
    n, K, nbins = 60_000, 20, 8
    y = rng.integers(0, nbins, n).astype(np.int64)
    cols = []
    for k in range(K):
        if k % 4 == 0:
            cols.append(np.where(rng.random(n) < 0.1, y, rng.integers(0, nbins, n)).astype(np.int64))
        else:
            cols.append(rng.integers(0, nbins, n).astype(np.int64))
    disc = np.column_stack(cols).astype(np.int16)
    observed = rng.random(K) * 0.01

    # reference gate using the old np.unique df path, recomputed inline.
    from mlframe.feature_selection.filters._analytic_mi_null import analytic_mi_null
    by = int(np.unique(y).size)
    ref = observed.astype(np.float64).copy()
    alpha = 1.0 - 0.95
    for k in range(K):
        if ref[k] <= 0.0:
            ref[k] = 0.0
            continue
        bx = int(np.unique(disc[:, k]).size)
        _nm, p = analytic_mi_null(float(ref[k]), n, bx, by)
        if p >= alpha:
            ref[k] = 0.0

    got = analytic_batch_noise_gate(disc, observed, y, n, min_nonzero_confidence=0.95)
    assert np.array_equal(got, ref)


@pytest.mark.parametrize("ncls", [2, 3, 5])
def test_analytic_gate_by_arg_matches_np_unique_derivation(ncls):
    """The dispatcher passes ``by`` (its already-computed ``np.count_nonzero(freqs_y)``) so the gate
    skips an O(n log n) ``np.unique(classes_y)`` re-sort on every per-pair-batch call. Passing ``by``
    must be BYTE-IDENTICAL to the internal np.unique derivation, and ``count_nonzero(freqs_y)`` must
    equal ``np.unique(classes_y).size`` (both count the occupied y-categories).
    """
    rng = np.random.default_rng(100 + ncls)
    n, K, nbins = 6000, 30, 8
    y = rng.integers(0, ncls, size=n).astype(np.int64)
    freqs_y = np.bincount(y, minlength=ncls).astype(np.float64) / n
    assert int(np.count_nonzero(freqs_y)) == int(np.unique(y).size)
    disc = rng.integers(0, nbins, size=(n, K)).astype(np.int32)
    observed = np.abs(rng.standard_normal(K)) * 0.01
    got_default = analytic_batch_noise_gate(disc, observed, y, n, min_nonzero_confidence=0.95)
    got_by = analytic_batch_noise_gate(
        disc, observed, y, n, min_nonzero_confidence=0.95, by=int(np.count_nonzero(freqs_y))
    )
    assert np.array_equal(got_default, got_by)
