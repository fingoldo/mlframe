"""Regression: BorutaShap's accept-side binomial null used the infinite-pool rate (100 - percentile)/100.

With ~10 shadows per trial the 99th-percentile gate sits between the two largest shadow importances, so an exchangeable
(pure-noise) column clears it ~10% of trials, not 1%; ~3 hits in 30 trials then looked significant and noise was accepted.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from mlframe.feature_selection.boruta_shap import _shadow_stats
from mlframe.feature_selection.boruta_shap._shadow_stats import finite_pool_null_hit_p


@pytest.mark.parametrize("m,q", [(10, 99.0), (10, 100.0), (5, 99.0), (40, 95.0), (3, 50.0)])
def test_finite_pool_rate_matches_monte_carlo(m, q):
    """The closed-form finite-pool null hit rate matches a Monte-Carlo estimate."""
    rng = np.random.default_rng(0)
    draws = rng.random((200_000, m + 1))
    thr = np.percentile(draws[:, :m], q, axis=1)
    empirical = float(np.mean(draws[:, m] > thr))
    assert finite_pool_null_hit_p(m, q) == pytest.approx(empirical, abs=0.004)


def test_limit_and_max_gate():
    """The null rate is 1/(m+1) at the max gate and tends to 1 - q/100 for a large shadow pool."""
    assert finite_pool_null_hit_p(10, 100.0) == pytest.approx(1 / 11)
    assert finite_pool_null_hit_p(100_000, 99.0) == pytest.approx(0.01, abs=1e-4)


def _run_test_features(hits, n_trials, recorded_p):
    """Run the Boruta hit test on ``hits`` with the recorded null rate ``recorded_p`` and return how many features it accepts."""
    def _binom(array, n, p, alternative):
        """Per-feature binomial-test p-values, the shape Boruta's ``binomial_H0_test`` returns."""
        from scipy.stats import binomtest

        return [binomtest(int(x), n, p, alternative=alternative).pvalue for x in array]

    self = SimpleNamespace(
        hits=hits, percentile=99, pvalue=0.05, all_columns=np.array([f"f{i}" for i in range(len(hits))]), binomial_H0_test=_binom,
        bonferoni_corrections=_shadow_stats.bonferoni_corrections, find_index_of_true_in_array=_shadow_stats.find_index_of_true_in_array,
        rejected_columns=[], accepted_columns=[], _null_hit_p_sum=recorded_p * n_trials, _null_hit_p_trials=n_trials,
    )
    _shadow_stats.test_features(self, iteration=n_trials)
    return len(self.accepted_columns[-1])


def test_exchangeable_noise_not_accepted_with_ten_shadows():
    """With ten shadows, pure-noise features are not accepted under the finite-pool null (the infinite-pool null accepts several)."""
    n_features, n_trials, m = 10, 30, 10
    p_true = finite_pool_null_hit_p(m, 99.0)
    hits = np.random.default_rng(1).binomial(n_trials, p_true, size=n_features).astype(float)
    assert _run_test_features(hits, n_trials, p_true) == 0
    # The old infinite-pool null (recorded rate == 0.01) accepts several of the same pure-noise columns.
    assert _run_test_features(hits, n_trials, 0.01) >= 2
