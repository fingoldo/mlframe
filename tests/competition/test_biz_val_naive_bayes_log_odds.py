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
    """predict_proba/predict and the averaging-baseline helper both return valid shapes and probability sums."""
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

    brier_logodds = float(np.mean((proba_logodds - y_test) ** 2))
    brier_avg = float(np.mean((proba_avg - y_test) ** 2))
    print(f"[log-odds independent] AUC {auc_logodds:.4f} vs {auc_avg:.4f}; Brier {brier_logodds:.4f} vs {brier_avg:.4f}")

    assert auc_logodds >= 0.945, f"log-odds AUC {auc_logodds} below threshold"
    # The AUC delta is NOT asserted: it measures 0.9576 against 0.9553, and separating two numbers 0.002
    # apart by a 0.001 floor is a coin toss against a BLAS summation order or a tie-handling change. What
    # log-odds summation actually buys under conditional independence is CONFIDENCE -- multiplying genuinely
    # independent evidence instead of averaging it away -- and that is a first-order effect: Brier 0.0809
    # against 0.2436, because averaging leaves every prediction bunched near the base rate (std 0.008).
    assert brier_avg - brier_logodds >= 0.05, (
        f"log-odds summation is no better calibrated than averaging on conditionally-independent features: "
        f"Brier {brier_logodds:.4f} against {brier_avg:.4f}. The whole point of multiplying independent "
        "evidence is the confidence it earns."
    )


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

    brier_logodds = float(np.mean((proba_logodds - y_test) ** 2))
    brier_avg = float(np.mean((proba_avg - y_test) ** 2))
    print(f"[log-odds dependent] AUC {auc_logodds:.4f} vs {auc_avg:.4f}; Brier {brier_logodds:.4f} vs {brier_avg:.4f}")

    # Asserted on the MECHANISM the docstring names, not on a 0.0005 AUC delta. That delta could be satisfied
    # by noise the moment the log-odds arm merely differed from averaging, reporting the limitation as still
    # demonstrated while nothing about conditional dependence was being shown. Over-multiplying correlated
    # evidence produces OVER-CONFIDENT probabilities, which is what Brier punishes: 0.3135 against 0.2058
    # here, with 96.8% of the log-odds predictions pushed past |p - 0.5| > 0.45 versus 0.1% for averaging.
    # The sign flips against the positive test above (0.0809 against 0.2436), which is the actual claim: the
    # method's validity is tied to conditional independence.
    assert brier_logodds - brier_avg >= 0.05, (
        f"log-odds summation is not over-confident under feature dependence: Brier {brier_logodds:.4f} against "
        f"averaging's {brier_avg:.4f}. The honest-negative fixture no longer demonstrates the limitation."
    )


def test_naive_bayes_log_odds_ensembler_feature_blocks():
    """feature_blocks groups columns into one sub-model per block instead of one per column."""
    X, y = _make_conditionally_independent(n=1000, n_features=6, n_informative=2, seed=0)
    ens = NaiveBayesLogOddsEnsembler(calibrate=False, feature_blocks=[(0, 1), (2, 3), (4, 5)])
    ens.fit(X, y)
    proba = ens.predict_proba(X)
    assert proba.shape == (1000, 2)
    assert len(ens.models_) == 3


def test_naive_bayes_log_odds_ensembler_rejects_multiclass():
    """Fitting on a 3-class target raises ValueError since the log-odds combination is binary-only."""
    X = np.random.default_rng(0).normal(size=(30, 3))
    y = np.array([0, 1, 2] * 10)
    ens = NaiveBayesLogOddsEnsembler(calibrate=False)
    with pytest.raises(ValueError):
        ens.fit(X, y)
