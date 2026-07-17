"""biz_value: DEFAULT_ECE_NBINS minimises RMSE of the ECE estimate vs ground truth.

The equal-width ECE estimator is biased upward when over-binned (each per-bin
|mean_y - mean_p| absorbs sampling noise that does not cancel under |.|). The bench
``calibration/_benchmarks/bench_ece_nbins.py`` shows nbins=10 beats nbins=15 in 14/18
(scenario x n) cells on RMSE vs a high-resolution ground-truth ECE. This pins that win:
the shipped default must not be worse than 15 on the aggregate honest metric.

Fails on the pre-flip default (15): with DEFAULT_ECE_NBINS=15 the aggregate RMSE is
higher than the nbins=10 reference, so the strict-improvement assertion trips.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.calibration.policy import _ece_score, DEFAULT_ECE_NBINS

EPS = 1e-6


def _g_over(p):
    z = np.log(p / (1 - p)) * 0.6
    return 1.0 / (1.0 + np.exp(-z))


def _g_sshape(p):
    z = np.log(p / (1 - p))
    return 1.0 / (1.0 + np.exp(-(z * 0.7 + 0.4 * np.sin(2.5 * z))))


_SCENARIOS = [
    ("overconfident", 2.0, 2.0, _g_over),
    ("s_shape_skew", 1.3, 3.0, _g_sshape),
    ("shift_up", 3.0, 3.0, lambda p: np.clip(p + 0.08, 0, 1)),
]


def _ece_true(g, a, b, rng):
    s = np.clip(rng.beta(a, b, size=200_000), EPS, 1 - EPS)
    return float(np.mean(np.abs(g(s) - s)))


def _sample(g, a, b, n, rng):
    s = np.clip(rng.beta(a, b, size=n), EPS, 1 - EPS)
    y = (rng.random(n) < g(s)).astype(np.float64)
    return np.ascontiguousarray(y), np.ascontiguousarray(s)


def _aggregate_rmse(nbins, seeds=10, ns=(2000, 20000)):
    grid_rng = np.random.default_rng(7)
    sq = []
    for name, a, b, g in _SCENARIOS:
        et = _ece_true(g, a, b, grid_rng)
        for n in ns:
            for seed in range(seeds):
                rng = np.random.default_rng(1000 * seed + n + len(name))
                y, s = _sample(g, a, b, n, rng)
                sq.append((_ece_score(y, s, nbins) - et) ** 2)
    return float(np.sqrt(np.mean(sq)))


def test_biz_val_default_ece_nbins_beats_15_on_rmse():
    """The shipped DEFAULT_ECE_NBINS yields strictly lower aggregate ECE-RMSE than 15.

    Pre-flip (default=15) this compares 15 vs 15 -> equal -> assertion fails.
    Post-flip (default=10) the bench-measured ~10% RMSE reduction trips the True branch.
    """
    rmse_default = _aggregate_rmse(DEFAULT_ECE_NBINS)
    rmse_15 = _aggregate_rmse(15)
    assert rmse_default < rmse_15 * 0.97, (
        f"DEFAULT_ECE_NBINS={DEFAULT_ECE_NBINS} rmse={rmse_default:.5f} must beat nbins=15 rmse={rmse_15:.5f} by >=3% on ground-truth ECE-RMSE"
    )


def test_biz_val_default_ece_nbins_is_10():
    """Guard the flipped default so a silent revert to 15 is caught."""
    assert DEFAULT_ECE_NBINS == 10
