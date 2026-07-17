"""Regression tests for the tied-score bootstrap AUC resampler fast path.

``make_bootstrap_auc_resampler`` scores each bootstrap resample without re-argsorting.
On TIED base scores it now uses the O(n+K) grouped path (``_fused_resample_auc_grouped``)
instead of a per-resample argsort. The grouped AUC must be BIT-IDENTICAL to the exact
``fast_roc_auc_unstable(y[idx], score[idx])`` reference, because AUC is invariant to the
within-tie argsort order.

These tests would FAIL if the grouped kernel emitted an AUC boundary per tied element
(the naive rank-binning bug) instead of per distinct score group.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.metrics._core_auc_brier import (
    make_bootstrap_auc_resampler,
    fast_roc_auc_unstable,
)


def _exact(y, s, idx):
    return fast_roc_auc_unstable(y[idx], s[idx])


@pytest.mark.parametrize("ndistinct", [2, 5, 20, 200])
@pytest.mark.parametrize("n", [500, 5000, 50000])
def test_tied_resampler_bit_identical(n, ndistinct):
    rng = np.random.default_rng(ndistinct * 1000 + n)
    raw = rng.random(n)
    score = np.round(raw * (ndistinct - 1)) / (ndistinct - 1)  # tied / low-cardinality
    y = (rng.random(n) < (0.2 + 0.6 * score)).astype(np.int64)

    # sanity: the base scores actually have ties
    assert np.unique(score).size <= ndistinct

    resampler = make_bootstrap_auc_resampler(y, score)
    for t in range(40):
        idx = rng.integers(0, n, n)
        fast = resampler(idx)
        exact = _exact(y, score, idx)
        if np.isnan(exact):
            assert np.isnan(fast)
        else:
            # bit-identical: exact == comparison, no tolerance
            assert fast == exact, f"n={n} nd={ndistinct} trial={t}: fast={fast!r} exact={exact!r}"


def test_tie_free_resampler_still_bit_identical():
    """Tie-free continuous scores must still use (and match) the existing fused fast path."""
    rng = np.random.default_rng(7)
    n = 20000
    score = rng.random(n)  # all-distinct (tie-free) w.h.p.
    y = (rng.random(n) < score).astype(np.int64)
    resampler = make_bootstrap_auc_resampler(y, score)
    for _ in range(30):
        idx = rng.integers(0, n, n)
        fast = resampler(idx)
        exact = _exact(y, score, idx)
        assert fast == exact


def test_tied_resampler_matches_full_auc_on_identity_resample():
    """An identity resample (idx = arange) must reproduce the plain AUC of the full data."""
    rng = np.random.default_rng(3)
    n = 4000
    score = np.round(rng.random(n) * 9) / 9  # 10 distinct values
    y = (rng.random(n) < (0.3 + 0.4 * score)).astype(np.int64)
    resampler = make_bootstrap_auc_resampler(y, score)
    idx = np.arange(n)
    assert resampler(idx) == fast_roc_auc_unstable(y, score)


def test_float_labels_tied_supported():
    """Labels given as float {0.0, 1.0} still hit the grouped fast path bit-identically."""
    rng = np.random.default_rng(11)
    n = 3000
    score = np.round(rng.random(n) * 4) / 4  # 5 distinct
    y = (rng.random(n) < 0.5).astype(np.float64)
    resampler = make_bootstrap_auc_resampler(y, score)
    for _ in range(20):
        idx = rng.integers(0, n, n)
        assert resampler(idx) == _exact(y, score, idx)
