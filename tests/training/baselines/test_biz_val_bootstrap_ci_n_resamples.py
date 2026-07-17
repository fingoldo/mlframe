"""biz_value test for the bootstrap-CI n_resamples default flip 1000 -> 2000.

The reported 95% percentile CI (2.5/97.5) on the strongest baseline's metric
is a Monte-Carlo estimate whose seed-to-seed jitter scales ~1/sqrt(B). The
default flip to B=2000 buys a lower-wobble CI. This test pins the WIN: at
B=2000 the seed jitter of the CI bounds is materially smaller than at B=1000
on a synthetic where the bound is well-defined, and the config default is 2000.

Bench: src/mlframe/training/baselines/_benchmarks/bench_bootstrap_ci_n_resamples.py
"""

from __future__ import annotations

import numpy as np

from mlframe.training.baselines._dummy_bootstrap import _numba_bootstrap_rmse_samples
from mlframe.training.configs import DummyBaselinesConfig


def _ci_bounds(y, p, B, seed):
    s = _numba_bootstrap_rmse_samples(
        np.ascontiguousarray(y, np.float64),
        np.ascontiguousarray(p, np.float64),
        int(B),
        int(seed),
    )
    return float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))


def test_default_n_resamples_is_2000():
    """The flipped default. Guards against silent revert to 1000."""
    assert DummyBaselinesConfig().bootstrap_ci_n_resamples == 2000


def test_biz_val_n_resamples_2000_lowers_ci_jitter_vs_1000():
    """At B=2000 the seed-to-seed jitter of the CI bounds is materially
    (>=20%) lower than at B=1000. Measured ~28% reduction; gate at 20%."""
    rng = np.random.default_rng(0)
    n = 800
    y = rng.normal(0, 1, n)
    p = y + rng.normal(0, 0.7, n)
    # warm JIT
    _ci_bounds(y, p, 50, 0)

    n_seeds = 40

    def jitter(B):
        los = np.empty(n_seeds)
        his = np.empty(n_seeds)
        for k in range(n_seeds):
            los[k], his[k] = _ci_bounds(y, p, B, 1000 + k)
        return (los.std() + his.std()) / 2.0

    j1000 = jitter(1000)
    j2000 = jitter(2000)
    assert j2000 < j1000 * 0.80, f"B=2000 should cut CI jitter >=20% vs B=1000: j1000={j1000:.5f} j2000={j2000:.5f} ratio={j2000 / j1000:.3f}"
