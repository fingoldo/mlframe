"""EarlyStoppingWrapper stops a HistGradientBoosting model at its best stage instead of keeping every stage."""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

from mlframe.estimators.early_stopping import EarlyStoppingWrapper


def _noisy_classification():
    """Mostly-noise binary data with a weak signal, so the validation curve peaks early and then decays."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(600, 6))
    y = (X[:, 0] * 0.4 + rng.normal(size=600) > 0).astype(int)
    return X, y


def test_classifier_budget_is_used_and_snapshot_is_cut():
    """The wrapper grows the model to its own max_iter and keeps only the best prefix of iterations."""
    X, y = _noisy_classification()
    wrapper = EarlyStoppingWrapper(
        HistGradientBoostingClassifier(early_stopping=False, learning_rate=0.3, max_depth=6), max_iter=80, patience=80, monotonic_decline_patience=None, random_state=0
    )
    wrapper.fit(X, y)
    n_kept = len(wrapper.best_model_._predictors)
    assert n_kept < 80, "the snapshot kept every grown iteration"
    assert wrapper.best_model_.n_iter_ == n_kept
    assert wrapper.predict_proba(X[:5]).shape == (5, 2)


def test_snapshot_predicts_like_the_best_stage():
    """The cut model's probabilities equal the full model's staged prediction at the kept stage."""
    X, y = _noisy_classification()
    wrapper = EarlyStoppingWrapper(
        HistGradientBoostingClassifier(early_stopping=False, learning_rate=0.3, max_depth=6), max_iter=60, patience=60, monotonic_decline_patience=None, random_state=0
    )
    wrapper.fit(X, y)
    stage = len(wrapper.best_model_._predictors)
    full = wrapper.estimator_
    staged = list(full.staged_predict_proba(X[:20]))
    np.testing.assert_allclose(wrapper.best_model_.predict_proba(X[:20]), staged[stage - 1], atol=1e-9)


def test_regressor_is_cut_too():
    """The same holds for the regressor, whose default scoring is negative RMSE."""
    rng = np.random.default_rng(1)
    X = rng.normal(size=(500, 5))
    y = X[:, 0] * 0.3 + rng.normal(size=500)
    wrapper = EarlyStoppingWrapper(HistGradientBoostingRegressor(early_stopping=False, learning_rate=0.3, max_depth=6), max_iter=60, patience=60, monotonic_decline_patience=None, random_state=0)
    wrapper.fit(X, y)
    assert len(wrapper.best_model_._predictors) < 60


def test_wrapper_exposes_classes_and_the_base_estimator_type():
    """sklearn utilities see a classifier (or regressor) with classes_, as for the wrapped model."""
    from sklearn.base import is_classifier, is_regressor

    X, y = _noisy_classification()
    clf = EarlyStoppingWrapper(HistGradientBoostingClassifier(early_stopping=False), max_iter=10, random_state=0)
    assert is_classifier(clf) and not is_regressor(clf)
    clf.fit(X, y)
    np.testing.assert_array_equal(clf.classes_, np.array([0, 1]))
    reg = EarlyStoppingWrapper(HistGradientBoostingRegressor(early_stopping=False), max_iter=10, random_state=0)
    assert is_regressor(reg) and not is_classifier(reg)
