"""Regression (N3): conditional_symmetric_uncertainty (CSU) must stay in [0, 1]. CSU is bounded by
[0,1] in exact arithmetic (I(X;Y|Z) <= min(H(X|Z),H(Y|Z)) <= denom/2), but its Miller-Madow-corrected
terms can break that bound: a tiny-but-positive denom (just above the 1e-12 guard) divided into an
MM-inflated numerator yielded CSU >> 1, which would spuriously dominate the redundancy ranking.
"""

from __future__ import annotations

import numpy as np

from mlframe.feature_selection.filters.info_theory._entropy_kernels import (
    conditional_symmetric_uncertainty,
)


def test_csu_stays_in_unit_interval_high_card_small_n():
    """conditional_symmetric_uncertainty stays within [0,1] even when a tiny-but-positive denominator meets a Miller-Madow-inflated numerator on sparse high-cardinality joints."""
    xi, yi, zi = np.array([0]), np.array([1]), np.array([2])
    for seed in range(60):
        rng = np.random.default_rng(seed)
        n = 60
        xcol = rng.integers(0, 5, n)
        ycol = rng.integers(0, 5, n)
        zcol = rng.integers(0, 45, n)  # high cardinality vs n -> sparse joints -> large MM correction
        data = np.column_stack([xcol, ycol, zcol]).astype(np.int32)
        nbins = np.array([5, 5, 45], dtype=np.int64)
        csu = conditional_symmetric_uncertainty(data, xi, yi, zi, nbins)
        assert 0.0 <= csu <= 1.0, f"seed={seed}: CSU={csu} escaped [0, 1]"
