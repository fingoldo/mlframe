"""Tests for mlframe.custom_estimators.

Covers audit 05 T1-T3: averager classifier probability sum, fit/predict smoke
across shapes, and sklearn `check_estimator` status (xfail for averagers that
expect pre-computed probability columns as features).
"""

import numpy as np
import pickle
import pytest
from hypothesis import given, settings, strategies as st
from hypothesis.extra.numpy import arrays

from mlframe.custom_estimators import (
    ArithmAvgClassifier,
    GeomAvgClassifier,
    PureRandomClassifier,
    MyDecorrelator,
)


# ---------------------------------------------------------------------------
# T1 — proba sums to 1
# ---------------------------------------------------------------------------

@given(
    X=arrays(
        dtype=np.float64,
        shape=st.tuples(st.integers(2, 40), st.integers(2, 6)),
        elements=st.floats(1e-6, 1.0, allow_nan=False, allow_infinity=False),
    )
)
@settings(max_examples=30, deadline=None)
def test_arithm_avg_proba_sum_to_one(X):
    y = np.zeros(len(X), dtype=int)
    y[:: max(1, len(X) // 2)] = 1  # ensure at least 2 classes
    clf = ArithmAvgClassifier(nprobs=X.shape[1]).fit(X, y)
    probs = clf.predict_proba(X)
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-6)


@given(
    X=arrays(
        dtype=np.float64,
        shape=st.tuples(st.integers(2, 40), st.integers(2, 6)),
        elements=st.floats(1e-3, 1.0, allow_nan=False, allow_infinity=False),
    )
)
@settings(max_examples=30, deadline=None)
def test_geom_avg_proba_sum_to_one(X):
    y = np.zeros(len(X), dtype=int)
    y[:: max(1, len(X) // 2)] = 1
    clf = GeomAvgClassifier(nprobs=X.shape[1]).fit(X, y)
    probs = clf.predict_proba(X)
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-6)


# ---------------------------------------------------------------------------
# T2 — fit/predict smoke across shapes
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n_samples,n_cols", [(5, 2), (10, 3), (50, 5), (1, 2)])
@pytest.mark.parametrize("clf_cls", [ArithmAvgClassifier, GeomAvgClassifier])
def test_averager_fit_predict_smoke(n_samples, n_cols, clf_cls):
    rng = np.random.default_rng(0)
    X = rng.uniform(1e-3, 1.0, size=(n_samples, n_cols))
    y = (rng.random(n_samples) > 0.5).astype(int)
    # guarantee both classes present when possible
    if n_samples >= 2:
        y[0], y[1] = 0, 1
    clf = clf_cls(nprobs=n_cols).fit(X, y)
    preds = clf.predict(X)
    assert preds.shape == (n_samples,)
    # predict must return values from classes_
    assert set(np.unique(preds)).issubset(set(clf.classes_.tolist()))
    assert clf.n_features_in_ == n_cols


@pytest.mark.parametrize("n_samples_mult", [1, 2, 10])
@pytest.mark.parametrize("n_classes", [2, 3, 5])
def test_pure_random_classifier_smoke(n_samples_mult, n_classes):
    # ensure every class label is present so classes_ has size n_classes
    rng = np.random.default_rng(0)
    n_samples = n_samples_mult * n_classes
    X = rng.random((n_samples, 4))
    y = np.tile(np.arange(n_classes), n_samples_mult)
    clf = PureRandomClassifier(nprobs=n_classes, random_state=42).fit(X, y)
    probs = clf.predict_proba(X)
    assert probs.shape == (n_samples, n_classes)
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-6)
    preds = clf.predict(X)
    assert preds.shape == (n_samples,)
    assert set(np.unique(preds)).issubset(set(clf.classes_.tolist()))


def test_pure_random_classifier_respects_random_state():
    X = np.random.rand(20, 3)
    y = np.array([0, 1] * 10)
    a = PureRandomClassifier(random_state=123).fit(X, y).predict_proba(X)
    b = PureRandomClassifier(random_state=123).fit(X, y).predict_proba(X)
    np.testing.assert_array_equal(a, b)
    c = PureRandomClassifier(random_state=456).fit(X, y).predict_proba(X)
    assert not np.array_equal(a, c)


# ---------------------------------------------------------------------------
# T3 — check_estimator (xfail: averagers assume X is already prob columns, not
# arbitrary features, so the generic sklearn smoke suite mis-feeds them)
# ---------------------------------------------------------------------------

@pytest.mark.xfail(
    reason="Averager classifiers interpret X as pre-computed probability columns, "
           "not generic features — the sklearn check_estimator synthetic data violates that."
)
@pytest.mark.parametrize("clf_cls", [ArithmAvgClassifier, GeomAvgClassifier, PureRandomClassifier])
def test_check_estimator_averagers(clf_cls):
    from sklearn.utils.estimator_checks import check_estimator

    if clf_cls is PureRandomClassifier:
        check_estimator(clf_cls())
    else:
        check_estimator(clf_cls(nprobs=2))


# ---------------------------------------------------------------------------
# Pickle roundtrip for averagers (needed for joblib/loky parallelism)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "clf",
    [
        ArithmAvgClassifier(nprobs=2),
        GeomAvgClassifier(nprobs=2),
        PureRandomClassifier(random_state=0),
    ],
)
def test_pickle_roundtrip_averagers(clf):
    X = np.random.rand(10, 2)
    y = np.array([0, 1] * 5)
    clf.fit(X, y)
    restored = pickle.loads(pickle.dumps(clf))
    np.testing.assert_array_equal(clf.predict(X), restored.predict(X))


# ---------------------------------------------------------------------------
# MyDecorrelator: attribute-name regression (M6 in audit)
# ---------------------------------------------------------------------------

def test_my_decorrelator_uses_correlated_features_consistently():
    import pandas as pd

    rng = np.random.default_rng(0)
    a = rng.normal(size=50)
    # b correlated with a; c independent
    df = pd.DataFrame({"a": a, "b": a * 1.0 + rng.normal(0, 0.01, 50), "c": rng.normal(size=50)})
    dec = MyDecorrelator(threshold=0.9).fit(df)
    assert hasattr(dec, "correlated_features_")
    out = dec.transform(df)
    # the correlated pair member dropped should not be in output columns
    assert set(out.columns).issubset(set(df.columns))
    assert len(out.columns) < len(df.columns)
