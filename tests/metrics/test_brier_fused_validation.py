"""Regression tests for the fused validation+Brier kernel.

``fast_brier_score_loss`` now validates probabilities (finite, in [0,1]) INSIDE a single
njit sweep instead of 3 separate numpy passes (isfinite/min/max). The Brier value must be
bit-identical to the prior ``np.mean((y-p)**2)`` kernel on valid input, and nan on any
non-finite / out-of-range probability (unchanged contract).
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.metrics._core_auc_brier import (
    fast_brier_score_loss,
    _fast_brier_score_loss_seq,
    _fast_brier_score_loss_par,
    _PARALLEL_REDUCTION_THRESHOLD,
)


def _old_kernel(y, p):
    if len(y) >= _PARALLEL_REDUCTION_THRESHOLD:
        return _fast_brier_score_loss_par(y, p)
    return _fast_brier_score_loss_seq(y, p)


@pytest.mark.parametrize("n", [1, 100, 5000, 200000, 1_000_000])
def test_brier_bit_identical_valid(n):
    rng = np.random.default_rng(n)
    y = (rng.random(n) < 0.3).astype(np.float64)
    p = rng.random(n)
    got = fast_brier_score_loss(y, p)
    ref = _old_kernel(y, p)
    assert got == ref, f"n={n}: {got!r} != {ref!r}"


@pytest.mark.parametrize("bad", [1.2, np.nan, np.inf, -0.1, -np.inf])
def test_brier_nan_on_invalid(bad):
    y = np.array([0.0, 1.0, 0.0])
    p = np.array([0.5, bad, 0.3])
    assert np.isnan(fast_brier_score_loss(y, p))


def test_brier_empty_is_nan():
    assert np.isnan(fast_brier_score_loss(np.array([]), np.array([])))


def test_brier_boundary_probs_valid():
    """Exact 0.0 and 1.0 are VALID (inclusive range); must not be flagged nan."""
    y = np.array([0.0, 1.0, 1.0, 0.0])
    p = np.array([0.0, 1.0, 0.0, 1.0])
    got = fast_brier_score_loss(y, p)
    assert not np.isnan(got)
    assert got == _fast_brier_score_loss_seq(y, p)
