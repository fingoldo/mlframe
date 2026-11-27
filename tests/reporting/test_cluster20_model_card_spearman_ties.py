"""Cluster 20 regression: the model-card heteroscedasticity Spearman ranks ties by their average, so it does not depend on row order (audit 2026-11-20, 20-F13)."""

import numpy as np

from mlframe.reporting.charts.model_card import _spearman


def test_tied_predictions_give_a_row_order_independent_value():
    """Quantised predictions: shuffling the rows leaves the correlation unchanged (20-F13)."""
    rng = np.random.default_rng(0)
    yp = np.repeat(np.arange(5.0), 200)
    resid = np.abs(rng.normal(size=1000)) * (1 + yp)
    base = _spearman(resid, yp)
    for seed in range(5):
        perm = np.random.default_rng(seed).permutation(1000)
        assert _spearman(resid[perm], yp[perm]) == base


def test_matches_scipy_average_rank_definition_and_degenerate_cases():
    """Agrees with scipy on tied data; a constant side or fewer than 3 rows returns 0.0 (20-F13)."""
    from scipy.stats import spearmanr

    rng = np.random.default_rng(1)
    a = rng.integers(0, 6, 300).astype(float)
    b = a + rng.integers(0, 3, 300)
    assert abs(_spearman(a, b) - spearmanr(a, b).statistic) < 1e-9
    assert _spearman(np.ones(10), np.arange(10.0)) == 0.0
    assert _spearman(np.array([1.0, 2.0]), np.array([2.0, 1.0])) == 0.0
