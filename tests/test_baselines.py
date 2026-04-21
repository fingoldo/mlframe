"""Tests for mlframe.baselines — covers audit 05 T6."""

import numpy as np
import pytest
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, make_scorer, r2_score

from mlframe.baselines import get_best_dummy_score


def _xy_clf():
    rng = np.random.default_rng(0)
    X = rng.random((60, 4))
    y = (rng.random(60) > 0.5).astype(int)
    # ensure both classes present
    y[0], y[1] = 0, 1
    return X, y


def _xy_reg():
    rng = np.random.default_rng(0)
    X = rng.random((60, 4))
    y = rng.random(60)
    return X, y


def test_get_best_dummy_score_classifier_returns_finite_score():
    X, y = _xy_clf()
    split = 40
    scorer = make_scorer(accuracy_score)
    score = get_best_dummy_score(
        LogisticRegression(),
        X[:split], y[:split],
        X[split:], y[split:],
        scorer,
    )
    assert np.isfinite(score)
    # dummy accuracy is somewhere in [0, 1]
    assert 0.0 <= score <= 1.0


def test_get_best_dummy_score_regressor_returns_finite_score():
    X, y = _xy_reg()
    split = 40
    scorer = make_scorer(r2_score)
    score = get_best_dummy_score(
        Ridge(),
        X[:split], y[:split],
        X[split:], y[split:],
        scorer,
    )
    assert np.isfinite(score)


def test_get_best_dummy_score_raises_on_invalid_estimator():
    X, y = _xy_clf()
    # KMeans is neither a classifier nor a regressor
    with pytest.raises(TypeError, match="classifier or regressor"):
        get_best_dummy_score(
            KMeans(n_clusters=2),
            X, y, X, y,
            make_scorer(accuracy_score),
        )
