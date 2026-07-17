"""Regression: ``fast_roc_auc`` must honour ``sample_weight`` (sklearn's scorer forwards it on weighted fits).

Pre-fix the function raised ``NotImplementedError("fast_roc_auc does not support sample_weight")``, which crashed
weighted-eval cb/linear fuzz combos. It now computes a weighted ROC AUC matching ``sklearn.metrics.roc_auc_score``.
"""

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from mlframe.metrics.core import fast_roc_auc


def _data(n=500, seed=0):
    """Helper: Data."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, size=n).astype(np.float64)
    score = rng.random(n) * 0.6 + 0.2 * y  # signal so AUC > 0.5
    w = rng.random(n) * 4 + 0.1
    return y, score, w


def test_fast_roc_auc_sample_weight_does_not_raise():
    """Fast roc auc sample weight does not raise."""
    y, score, w = _data()
    val = fast_roc_auc(y, score, sample_weight=w)
    assert np.isfinite(val)


def test_fast_roc_auc_sample_weight_matches_sklearn():
    """Fast roc auc sample weight matches sklearn."""
    for seed in range(5):
        y, score, w = _data(seed=seed)
        got = fast_roc_auc(y, score, sample_weight=w)
        expected = roc_auc_score(y, score, sample_weight=w)
        assert got == pytest.approx(expected, abs=1e-9), f"seed={seed}: {got} vs {expected}"


def test_fast_roc_auc_unit_weights_match_unweighted():
    """Fast roc auc unit weights match unweighted."""
    y, score, _ = _data()
    w = np.ones_like(y)
    assert fast_roc_auc(y, score, sample_weight=w) == pytest.approx(fast_roc_auc(y, score), abs=1e-12)


def test_fast_roc_auc_none_weight_still_works():
    """Fast roc auc none weight still works."""
    y, score, _ = _data()
    assert np.isfinite(fast_roc_auc(y, score, sample_weight=None))
