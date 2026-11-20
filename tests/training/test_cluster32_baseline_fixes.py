"""Cluster 32 regressions: multiclass log-loss CI and a NaN-safe empirical-Bayes shrinkage (audit 2026-11-20, findings 32-F2, 32-F4)."""

import numpy as np

from mlframe.training.baselines._dummy_baseline_compute import PER_GROUP_SMOOTHING_MIN_PSEUDOCOUNTS, _empirical_bayes_pseudocounts
from mlframe.training.baselines._dummy_bootstrap import _UNHANDLED, _multiclass_logloss_ci


def test_multiclass_logloss_ci_is_computed_for_a_probability_matrix():
    """A (n, K) probability matrix used to fail the broadcast and drop the CI (32-F2)."""
    rng = np.random.default_rng(0)
    n, K = 200, 3
    y = rng.integers(0, K, n)
    p = rng.dirichlet(np.ones(K), size=n)
    out = _multiclass_logloss_ci(y, p, 200, 0)
    assert out is not _UNHANDLED
    lo, hi = out[0], out[1]
    point = float(np.mean(-np.log(np.clip(p[np.arange(n), y], 1e-15, 1 - 1e-15))))
    assert lo <= point <= hi


def test_empirical_bayes_is_finite_when_the_global_mean_is_nan():
    """One NaN target made the shrinkage NaN; the estimate now stays finite (32-F4)."""
    y = np.array([1.0, np.nan, 2.0, 3.0])
    group_sizes = np.array([2, 2])
    group_means = np.array([1.5, 2.5])
    m = _empirical_bayes_pseudocounts(y, group_sizes, group_means, float("nan"))
    assert np.isfinite(m)
    assert m == PER_GROUP_SMOOTHING_MIN_PSEUDOCOUNTS
