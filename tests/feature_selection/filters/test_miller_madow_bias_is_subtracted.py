"""The Miller-Madow correction removes the plug-in MI bias instead of doubling it (tracker 2026-11-02 wave 11, NJ11-1)."""

import numpy as np

from mlframe.feature_selection.filters._numba_utils import _pair_mm_mi_njit
from mlframe.feature_selection.filters._orthogonal_copula_mi_fe import _bin_mi_uniform_pair


def test_copula_pair_mi_of_independent_columns_is_near_zero():
    """20x20 bins, n=2000: the plug-in bias is ~0.09 nats. The corrected estimate averages ~0; the old code added the bias again and averaged ~0.18."""
    rng = np.random.default_rng(0)
    vals = [_bin_mi_uniform_pair(rng.uniform(size=2000), rng.uniform(size=2000), n_bins=20) for _ in range(30)]
    assert np.mean(vals) < 0.03


def test_pair_mm_mi_of_independent_codes_is_near_zero():
    """Joint MI of two independent 10-level codes against an independent binary target stays well below the plug-in bias."""
    rng = np.random.default_rng(1)
    n = 3000
    vals = []
    for _ in range(20):
        cx = rng.integers(0, 10, n).astype(np.int64)
        cy = rng.integers(0, 10, n).astype(np.int64)
        t = rng.integers(0, 2, n).astype(np.int64)
        vals.append(_pair_mm_mi_njit(cx, cy, t, 10, 10, 2, 1))
    assert np.mean(vals) < 0.02
