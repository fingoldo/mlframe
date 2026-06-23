"""Regression: ewma_residual njit recurrence is bit-identical to the prior Python loop.

Pins that ``_ewma_recurrence_njit`` produces exactly the same float64 output as the
original scalar Python loop (``ewma[i] = alpha*x[i] + (1-alpha)*ewma[i-1]``) across
scalar, multi-half-life, grouped, NaN-containing, and ``adjust=True`` paths.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_engineering.stationarity import ewma_residual


def _old_ewma_single(seg, hl, adjust=False):
    alpha = 1.0 - 2.0 ** (-1.0 / hl)
    seg_f = np.where(np.isfinite(seg), seg, 0.0)
    ewma = np.empty_like(seg_f)
    ewma[0] = seg_f[0]
    for i in range(1, seg_f.size):
        ewma[i] = alpha * seg_f[i] + (1.0 - alpha) * ewma[i - 1]
    if adjust:
        w = (1.0 - alpha) ** np.arange(seg_f.size)
        w_cum = np.cumsum(w[::-1])
        ewma = ewma * w[::-1] / w_cum
    return seg - ewma


@pytest.mark.parametrize("n", [3, 100, 5000])
def test_ewma_scalar_bit_identical(n):
    rng = np.random.default_rng(n)
    x = rng.standard_normal(n).cumsum()
    new = ewma_residual(x, half_life=20.0)
    old = _old_ewma_single(x, 20.0)
    assert np.array_equal(new, old, equal_nan=True)


def test_ewma_multi_half_life_bit_identical():
    rng = np.random.default_rng(1)
    x = rng.standard_normal(2000).cumsum()
    hl = [5.0, 20.0, 60.0, 240.0]
    new = ewma_residual(x, half_life=hl)
    for j, h in enumerate(hl):
        assert np.array_equal(new[:, j], _old_ewma_single(x, h), equal_nan=True)


def test_ewma_with_nans_bit_identical():
    rng = np.random.default_rng(2)
    x = rng.standard_normal(500).cumsum()
    x[::17] = np.nan
    new = ewma_residual(x, half_life=15.0)
    old = _old_ewma_single(x, 15.0)
    assert np.array_equal(new, old, equal_nan=True)


def test_ewma_adjust_bit_identical():
    rng = np.random.default_rng(3)
    x = rng.standard_normal(400).cumsum()
    new = ewma_residual(x, half_life=30.0, adjust=True)
    old = _old_ewma_single(x, 30.0, adjust=True)
    assert np.array_equal(new, old, equal_nan=True)


def test_ewma_grouped_bit_identical():
    rng = np.random.default_rng(4)
    n = 600
    x = rng.standard_normal(n).cumsum()
    groups = rng.integers(0, 3, size=n)
    new = ewma_residual(x, half_life=10.0, group_ids=groups)
    old = np.full(n, np.nan)
    for g in np.unique(groups):
        mask = groups == g
        if mask.sum() < 2:
            continue
        old[mask] = _old_ewma_single(x[mask], 10.0)
    assert np.array_equal(new, old, equal_nan=True)
