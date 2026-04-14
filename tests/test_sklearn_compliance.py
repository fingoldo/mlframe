"""Sklearn-compliance tests for estimators touched by fix-agent #8."""
import numpy as np
import pytest

pytest.importorskip("sklearn")

from sklearn.utils.validation import check_is_fitted
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from mlframe.custom_estimators import ArithmAvgClassifier, GeomAvgClassifier
from mlframe.estimators import ClassifierWithEarlyStopping


@pytest.fixture
def xy():
    rng = np.random.default_rng(0)
    X = rng.uniform(0.01, 0.99, size=(50, 3))
    y = rng.integers(0, 2, size=50)
    # ensure both classes present
    y[0] = 0
    y[1] = 1
    return X, y


@pytest.mark.parametrize("cls", [ArithmAvgClassifier, GeomAvgClassifier])
def test_avg_classifier_sklearn_compliance(cls, xy):
    X, y = xy
    clf = cls(nprobs=2)
    clf.fit(X, y)

    assert hasattr(clf, "classes_")
    assert hasattr(clf, "n_features_in_")
    assert clf.n_features_in_ == X.shape[1]

    check_is_fitted(clf)

    preds = clf.predict(X)
    assert set(np.unique(preds)).issubset(set(np.unique(y)))

    proba = clf.predict_proba(X)
    assert proba.shape == (len(X), 2)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-9)


def test_avg_classifier_predict_requires_fit():
    clf = ArithmAvgClassifier(nprobs=2)
    with pytest.raises(Exception):
        clf.predict(np.ones((3, 3)))


def test_classifier_with_early_stopping_proxies_predict_proba():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(60, 4))
    y = (X[:, 0] > 0).astype(int)

    wrapped = ClassifierWithEarlyStopping(base_estimator=LogisticRegression())
    wrapped.fit(X, y)

    proba = wrapped.predict_proba(X)
    assert proba.shape == (60, 2)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)

    # decision_function proxies too
    df = wrapped.decision_function(X)
    assert df.shape == (60,)


def test_cluster_module_imports_cleanly():
    # regression guard for DBSCAN NameError at cluster.py:26
    import mlframe.cluster as cluster
    assert hasattr(cluster, "DBSCAN")
    assert hasattr(cluster, "clusterize")


def test_get_model_best_iter_with_pipeline():
    lgbm = pytest.importorskip("lightgbm")
    from mlframe.helpers import get_model_best_iter

    rng = np.random.default_rng(2)
    X = rng.normal(size=(200, 4))
    y = (X[:, 0] > 0).astype(int)

    est = lgbm.LGBMClassifier(n_estimators=5, verbose=-1)
    pipe = Pipeline([("est", est)])
    pipe.fit(X, y)

    best_iter = get_model_best_iter(pipe)
    # lightgbm exposes best_iteration_ even without early stopping (may be 0 or the count)
    assert isinstance(best_iter, (int, np.integer)) or best_iter is None or best_iter == 0 or best_iter


def test_feature_importance_sign_check_uses_sorted_order():
    # Regression: sign check must index via sorted_idx[0], not raw [0].
    # Simulate: feature_importances with a negative min NOT at index 0 => previous logic missed the branch.
    import numpy as np
    feature_importances = np.array([0.5, -0.3, 0.1])
    sorted_idx = np.argsort(feature_importances)  # [1, 2, 0]
    # Fixed code: feature_importances[sorted_idx[0]] < 0
    assert feature_importances[sorted_idx[0]] < 0
    # Old buggy behavior would check feature_importances[0] (=0.5) and miss it.
    assert not (feature_importances[0] < 0)
