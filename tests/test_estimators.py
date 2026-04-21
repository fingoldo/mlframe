"""Tests for mlframe.estimators — covers audit 05 T4 (pickle roundtrip)."""

import pickle

import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.linear_model import LogisticRegression, Ridge

from mlframe.estimators import (
    ClassifierWithEarlyStopping,
    EstimatorWithEarlyStopping,
    RegressorWithEarlyStopping,
)


def _xy_clf():
    return make_classification(n_samples=50, n_features=4, random_state=0)


def _xy_reg():
    return make_regression(n_samples=50, n_features=4, noise=0.1, random_state=0)


def test_classifier_with_early_stopping_fit_predict():
    X, y = _xy_clf()
    clf = ClassifierWithEarlyStopping(base_estimator=LogisticRegression(max_iter=500), random_state=0)
    clf.fit(X, y)
    preds = clf.predict(X)
    assert preds.shape == (len(y),)
    probs = clf.predict_proba(X)
    assert probs.shape == (len(y), 2)
    assert np.allclose(probs.sum(axis=1), 1.0)


def test_regressor_with_early_stopping_fit_predict():
    X, y = _xy_reg()
    reg = RegressorWithEarlyStopping(base_estimator=Ridge(), random_state=0)
    reg.fit(X, y)
    preds = reg.predict(X)
    assert preds.shape == (len(y),)


def test_classifier_decision_function_raises_when_unavailable():
    # LogisticRegression has decision_function — use a model that doesn't.
    from sklearn.tree import DecisionTreeClassifier

    X, y = _xy_clf()
    clf = ClassifierWithEarlyStopping(base_estimator=DecisionTreeClassifier(random_state=0))
    clf.fit(X, y)
    with pytest.raises(AttributeError):
        clf.decision_function(X)


def test_pickle_roundtrip_classifier_with_early_stopping():
    X, y = _xy_clf()
    clf = ClassifierWithEarlyStopping(base_estimator=LogisticRegression(max_iter=500), random_state=0)
    clf.fit(X, y)

    restored = pickle.loads(pickle.dumps(clf))
    np.testing.assert_array_equal(clf.predict(X), restored.predict(X))
    np.testing.assert_allclose(clf.predict_proba(X), restored.predict_proba(X))


def test_pickle_roundtrip_regressor_with_early_stopping():
    X, y = _xy_reg()
    reg = RegressorWithEarlyStopping(base_estimator=Ridge(), random_state=0)
    reg.fit(X, y)

    restored = pickle.loads(pickle.dumps(reg))
    np.testing.assert_allclose(reg.predict(X), restored.predict(X))


def test_estimator_with_early_stopping_logs_and_fits_non_catboost(caplog):
    """The non-catboost path should log a warning and still fit."""
    import logging

    X, y = _xy_reg()
    base = Ridge()
    est = EstimatorWithEarlyStopping(base_estimator=base, random_state=0)
    with caplog.at_level(logging.WARNING, logger="mlframe.estimators"):
        est.fit(X, y)
    assert any("Early stopping" in rec.message for rec in caplog.records)
    assert est.n_features_in_ == X.shape[1]
