"""Unit + biz_value tests for mlframe.competition.naive_bayes_log_odds.

COMPETITION/EXPLORATORY ONLY — see module docstring under src/mlframe/competition/.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from mlframe.competition.naive_bayes_log_odds import NaiveBayesLogOddsEnsembler


def _make_conditionally_independent(n: int = 8000, n_features: int = 80, n_informative: int = 3, seed: int = 0):
    """Santander-style synthetic dataset with GENUINE conditional independence given y.

    A minority of features (``n_informative``) carry a class-conditional mean
    shift; the rest are pure Gaussian noise unrelated to ``y``. Crucially,
    every feature's noise is drawn independently per-sample and per-column,
    so conditional on ``y`` the features are exactly independent -- the
    textbook setup the log-odds Naive-Bayes combination rule is optimal for.
    """
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, size=n)
    X = np.empty((n, n_features))
    for i in range(n_features):
        if i < n_informative:
            mean = np.where(y == 1, 0.7, -0.7)
        else:
            mean = np.zeros(n)
        X[:, i] = mean + rng.normal(0, 1.0, size=n)
    return X, y


def _make_conditionally_dependent(n: int = 8000, n_features: int = 80, seed: int = 0):
    """Dataset where features share a common latent factor -> conditionally DEPENDENT given y.

    All features are noisy copies of a single latent variable ``L``, and ``y``
    is a noisy function of that same ``L``. Because ``y`` doesn't fully
    determine ``L``, features remain strongly correlated with each other even
    after conditioning on ``y`` -- violating the conditional-independence
    assumption the log-odds combination rule requires.
    """
    rng = np.random.default_rng(seed)
    latent = rng.normal(0, 1.0, size=n)
    y_prob = 1.0 / (1.0 + np.exp(-latent))
    y = (rng.uniform(size=n) < y_prob).astype(int)
    X = np.empty((n, n_features))
    for i in range(n_features):
        X[:, i] = latent + rng.normal(0, 0.3, size=n)
    return X, y


def test_naive_bayes_log_odds_ensembler_fit_predict_shapes():
    X, y = _make_conditionally_independent(n=500, seed=0)
    ens = NaiveBayesLogOddsEnsembler(calibrate=False)
    ens.fit(X, y)
    proba = ens.predict_proba(X)
    assert proba.shape == (500, 2)
    assert np.allclose(proba.sum(axis=1), 1.0)
    preds = ens.predict(X)
    assert set(np.unique(preds)) <= {0, 1}

    avg_proba = ens.predict_proba_average_baseline(X)
    assert avg_proba.shape == (500, 2)
    assert np.allclose(avg_proba.sum(axis=1), 1.0)


def test_biz_val_naive_bayes_log_odds_ensembler_beats_averaging_under_conditional_independence():
    """POSITIVE case: with genuinely conditionally-independent features, log-odds summation beats averaging."""
    X_train, y_train = _make_conditionally_independent(seed=0)
    X_test, y_test = _make_conditionally_independent(seed=1)

    ens = NaiveBayesLogOddsEnsembler(calibrate=False)
    ens.fit(X_train, y_train)

    proba_logodds = ens.predict_proba(X_test)[:, 1]
    proba_avg = ens.predict_proba_average_baseline(X_test)[:, 1]

    auc_logodds = roc_auc_score(y_test, proba_logodds)
    auc_avg = roc_auc_score(y_test, proba_avg)

    # measured on this fixture: auc_logodds ~= 0.9576, auc_avg ~= 0.9553
    assert auc_logodds >= 0.945, f"log-odds AUC {auc_logodds} below threshold"
    assert auc_logodds - auc_avg >= 0.001, f"log-odds ({auc_logodds}) did not beat averaging ({auc_avg}) by enough margin"


def test_biz_val_naive_bayes_log_odds_ensembler_honest_negative_dependent_features():
    """HONEST-NEGATIVE case: with conditionally-DEPENDENT features, log-odds summation does NOT beat averaging.

    This demonstrates the tracker's own critique: the method's validity is
    tied to conditional independence, which is almost never true in real
    production data. When it's violated, log-odds summation over-multiplies
    correlated evidence and does no better (here: measurably worse) than
    plain probability averaging over the exact same per-feature models.
    """
    X_train, y_train = _make_conditionally_dependent(seed=0)
    X_test, y_test = _make_conditionally_dependent(seed=1)

    ens = NaiveBayesLogOddsEnsembler(calibrate=False)
    ens.fit(X_train, y_train)

    proba_logodds = ens.predict_proba(X_test)[:, 1]
    proba_avg = ens.predict_proba_average_baseline(X_test)[:, 1]

    auc_logodds = roc_auc_score(y_test, proba_logodds)
    auc_avg = roc_auc_score(y_test, proba_avg)

    # measured on this fixture: auc_logodds ~= 0.7421, auc_avg ~= 0.7436 -- averaging wins
    assert auc_avg - auc_logodds >= 0.0005, (
        f"expected averaging ({auc_avg}) to beat or match log-odds ({auc_logodds}) under feature dependence, "
        "but log-odds unexpectedly won -- honest-negative fixture no longer demonstrates the limitation"
    )


def test_naive_bayes_log_odds_ensembler_feature_blocks():
    X, y = _make_conditionally_independent(n=1000, n_features=6, n_informative=2, seed=0)
    ens = NaiveBayesLogOddsEnsembler(calibrate=False, feature_blocks=[(0, 1), (2, 3), (4, 5)])
    ens.fit(X, y)
    proba = ens.predict_proba(X)
    assert proba.shape == (1000, 2)
    assert len(ens.models_) == 3


def test_naive_bayes_log_odds_ensembler_rejects_multiclass():
    X = np.random.default_rng(0).normal(size=(30, 3))
    y = np.array([0, 1, 2] * 10)
    ens = NaiveBayesLogOddsEnsembler(calibrate=False)
    with pytest.raises(ValueError):
        ens.fit(X, y)
