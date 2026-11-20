"""Streaming-cache signatures of different widths do not crash, and the pair kernel rejects out-of-range codes instead of writing outside its histogram."""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_selection.filters import cat_interactions as ci


def test_cached_signature_of_another_width_is_compared_not_crashed():
    """A refit that sees a new category (wider signature) marks the column non-reusable instead of raising on the shape mismatch."""
    rng = np.random.default_rng(0)
    col = rng.integers(0, 4, size=500).astype(np.int64)
    fd = col.reshape(-1, 1)
    old_sig = ci._column_signature(rng.integers(0, 3, size=500), 3)
    cache = {"col_signatures": {0: old_sig}, "marginal_mis": {0: 0.1}, "target_sig": "t"}
    mask, reused, sigs = ci._restore_cached_marginal_mis(fd, np.array([0]), np.array([4]), cache, kl_threshold=1e-6, target_sig="t")
    assert not mask[0] and np.isnan(reused[0]) and sigs[0].shape[0] == 4


def test_pair_kernel_rejects_code_beyond_nbins():
    """A code equal to nbins would silently land in another cell; it must raise."""
    n = 40
    fd = np.zeros((n, 2), dtype=np.int32)
    fd[0, 0] = 3
    nbins = np.array([3, 3], dtype=np.int64)
    classes_y = (np.arange(n) % 2).astype(np.int32)
    freqs_y = np.array([0.5, 0.5])
    with pytest.raises(ValueError):
        ci._pair_search_kernel_njit(fd, np.array([0], dtype=np.int64), np.array([1], dtype=np.int64), np.zeros(2), nbins, classes_y, freqs_y, np.int32)


def test_pair_kernel_valid_codes_still_score():
    """In-range codes give a finite joint MI."""
    rng = np.random.default_rng(0)
    fd = rng.integers(0, 3, size=(200, 2)).astype(np.int32)
    y = (fd[:, 0] == fd[:, 1]).astype(np.int32)
    freqs_y = np.bincount(y, minlength=2) / 200.0
    mi, _ii, nu = ci._pair_search_kernel_njit(fd, np.array([0]), np.array([1]), np.zeros(2), np.array([3, 3]), y, freqs_y, np.int32)
    assert np.isfinite(mi[0]) and mi[0] > 0 and nu[0] == 9
