"""Stratified MI-screen sampling must spend its whole row budget on tie-heavy targets.

A mostly-zero heavy-tail target (hourly rates of fixed-price jobs) collapses the quantile cuts into a couple of strata. A
fixed ``sample_n // n_strata`` share per stratum then screened 6666 rows out of a 100_000 budget, which made the per-pair
MI and the permutation-null filter noisy enough to drop most features.
"""

from __future__ import annotations

import numpy as np

from mlframe.training.composite.discovery.screening import _sample_indices


def test_tie_heavy_target_uses_full_budget():
    rng = np.random.default_rng(0)
    n = 200_000
    y = np.where(rng.random(n) < 0.85, 0.0, rng.lognormal(size=n))
    idx = _sample_indices(n, 100_000, 0, strategy="stratified_quantile", y=y, n_strata=30)
    assert idx.size == 100_000
    assert np.unique(idx).size == idx.size
    assert np.all(np.diff(idx) > 0)


def test_tail_is_still_oversampled():
    """The strategy's purpose survives: the rare non-zero tail gets more than its natural share."""
    rng = np.random.default_rng(1)
    n = 200_000
    y = np.where(rng.random(n) < 0.85, 0.0, rng.lognormal(size=n))
    idx = _sample_indices(n, 20_000, 0, strategy="stratified_quantile", y=y, n_strata=30)
    assert idx.size == 20_000
    assert (y[idx] > 0).mean() > (y > 0).mean()


def test_small_population_returns_all_rows():
    y = np.zeros(50)
    idx = _sample_indices(50, 100, 0, strategy="stratified_quantile", y=y, n_strata=10)
    np.testing.assert_array_equal(idx, np.arange(50))
