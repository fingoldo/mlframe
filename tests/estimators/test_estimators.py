"""Tests for mlframe.estimators.base — covers audit 05 T4 (pickle roundtrip)."""

import pickle  # nosec B403 -- test-only local pickle round-trip, never untrusted/network data

import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.linear_model import LogisticRegression, Ridge

from mlframe.estimators.base import (
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

    restored = pickle.loads(pickle.dumps(clf))  # nosec B301 -- round-trip of a locally-created, trusted object
    np.testing.assert_array_equal(clf.predict(X), restored.predict(X))
    np.testing.assert_allclose(clf.predict_proba(X), restored.predict_proba(X))


def test_pickle_roundtrip_regressor_with_early_stopping():
    X, y = _xy_reg()
    reg = RegressorWithEarlyStopping(base_estimator=Ridge(), random_state=0)
    reg.fit(X, y)

    restored = pickle.loads(pickle.dumps(reg))  # nosec B301 -- round-trip of a locally-created, trusted object
    np.testing.assert_allclose(reg.predict(X), restored.predict(X))


def test_estimator_with_early_stopping_logs_and_fits_non_catboost(caplog):
    """The non-catboost path should log a warning and still fit."""
    import logging

    X, y = _xy_reg()
    base = Ridge()
    est = EstimatorWithEarlyStopping(base_estimator=base, random_state=0)
    with caplog.at_level(logging.WARNING, logger="mlframe.estimators.base"):
        est.fit(X, y)
    assert any("Early stopping" in rec.message for rec in caplog.records)
    assert est.n_features_in_ == X.shape[1]


def test_default_stratify_none_auto_stratifies_imbalanced_classifier_split():
    """Regression test for audit F1: stratify=None must not silently mean 'unstratified'.

    Before the fix, ``stratify`` was passed straight through to ``train_test_split`` as ``None`` unless the
    caller explicitly re-passed a stratify vector at construction time (impossible, since ``y`` only exists
    at fit-time). On a 95:5 imbalanced target, an unstratified split could produce an eval_set with a skewed
    or single-class distribution. Verify the internal split (captured via monkeypatching train_test_split)
    is now stratified on y by default.
    """
    import mlframe.estimators.base as base_mod
    from lightgbm import LGBMClassifier

    rng = np.random.RandomState(0)
    n_majority, n_minority = 190, 10
    X = rng.normal(size=(n_majority + n_minority, 4))
    y = np.array([0] * n_majority + [1] * n_minority)

    captured = {}
    real_train_test_split = base_mod.train_test_split

    def _spy(*args, **kwargs):
        captured["stratify"] = kwargs.get("stratify")
        return real_train_test_split(*args, **kwargs)

    est = ClassifierWithEarlyStopping(base_estimator=LGBMClassifier(n_estimators=10, verbose=-1), test_size=0.2, random_state=0)
    import mlframe.estimators.base as _b

    orig = _b.train_test_split
    _b.train_test_split = _spy
    try:
        est.fit(X, y)
    finally:
        _b.train_test_split = orig

    assert captured["stratify"] is not None, "stratify must be auto-derived from y for a classifier estimator, not left None"
    np.testing.assert_array_equal(captured["stratify"], y)


def test_default_stratify_none_stays_none_for_regressor():
    """A regressor base estimator with continuous y must NOT be spuriously stratified."""

    _X, y = _xy_reg()
    est = EstimatorWithEarlyStopping(base_estimator=Ridge(), random_state=0)
    stratify = est._resolve_stratify(y)
    assert stratify is None


def test_fit_preserves_dataframe_columns_and_sets_feature_names_in(caplog):
    """Regression test for audit F2: a bare check_array(X) silently discarded DataFrame columns / feature_names_in_.

    Verify a DataFrame input keeps its column names visible via ``feature_names_in_`` and that ``predict``
    also accepts a DataFrame without erroring (i.e. it is not forced through check_array either).
    """
    import pandas as pd

    X_arr, y = _xy_reg()
    columns = [f"feat_{i}" for i in range(X_arr.shape[1])]
    X = pd.DataFrame(X_arr, columns=columns)

    est = RegressorWithEarlyStopping(base_estimator=Ridge(), random_state=0)
    est.fit(X, y)

    assert hasattr(est, "feature_names_in_")
    np.testing.assert_array_equal(est.feature_names_in_, np.asarray(columns, dtype=object))

    preds = est.predict(X)
    assert preds.shape == (len(y),)
