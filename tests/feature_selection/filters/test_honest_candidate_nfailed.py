"""The honest per-candidate null for supervised binning: exceedance count matches a direct loop, and separates signal from noise."""

from __future__ import annotations

import numpy as np

from mlframe.feature_selection.filters._honest_mdlp_null import honest_candidate_nfailed
from mlframe.feature_selection.filters._numba_utils_cross_module import _mi_codes_njit


def _fixed_codes_fn(codes: np.ndarray, nbins: np.ndarray):
    """A codes function that ignores the permutation, so the null is the plain fixed-codes permutation null."""

    def _fn(row_perm):
        """Return the fixed codes for any permutation."""
        return codes, nbins

    return _fn


def test_honest_candidate_nfailed_matches_direct_loop():
    """The exceedance count equals a brute-force loop over the same permutations."""
    rng = np.random.default_rng(3)
    n = 400
    full = np.column_stack([rng.integers(0, 4, n), rng.integers(0, 3, n)]).astype(np.int64)  # col 0 = candidate, col 1 = target
    nbins = np.array([4, 3], dtype=np.int64)
    seed, B = 17, 40
    nfailed, observed = honest_candidate_nfailed(_fixed_codes_fn(full, nbins), full, nbins, 1, 0, B, seed)
    expected_obs = _mi_codes_njit(full[:, 0].copy(), 4, full[:, 1].copy(), 3)
    assert observed == expected_obs
    rng2 = np.random.default_rng(seed)
    expected = 0
    for _ in range(B):
        perm = rng2.permutation(n)
        if _mi_codes_njit(full[:, 0].copy(), 4, np.ascontiguousarray(full[perm, 1]), 3) >= observed:
            expected += 1
    assert nfailed == expected


def test_honest_candidate_nfailed_signal_is_never_exceeded_noise_often_is():
    """A column identical to the target is never matched by a shuffle; an independent column often is."""
    rng = np.random.default_rng(5)
    n = 2000
    y = rng.integers(0, 3, n)
    signal = y.copy()
    noise = rng.integers(0, 3, n)
    full = np.column_stack([signal, noise, y]).astype(np.int64)
    nbins = np.array([3, 3, 3], dtype=np.int64)
    nf_signal, _ = honest_candidate_nfailed(_fixed_codes_fn(full, nbins), full, nbins, 2, 0, 99, 1)
    nf_noise, _ = honest_candidate_nfailed(_fixed_codes_fn(full, nbins), full, nbins, 2, 1, 99, 1)
    assert nf_signal == 0
    assert nf_noise > 10


def test_honest_candidate_nfailed_returns_none_for_uncovered_column():
    """A column beyond the honest codes width returns None instead of guessing."""
    full = np.zeros((50, 3), dtype=np.int64)
    nbins = np.array([2, 2, 2], dtype=np.int64)
    short = (np.zeros((50, 1), dtype=np.int64), np.array([2], dtype=np.int64))
    assert honest_candidate_nfailed(lambda p: short, full, nbins, 2, 1, 5, 0) is None
