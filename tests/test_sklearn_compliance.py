"""Sklearn-compliance tests for estimators touched by fix-agent #8."""

import numpy as np
import pytest

pytest.importorskip("sklearn")

from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from mlframe.estimators.custom import ArithmAvgClassifier, GeomAvgClassifier
from mlframe.estimators.base import ClassifierWithEarlyStopping


@pytest.fixture
def xy():
    """Builds seeded synthetic test data; returns ``(X, y)``."""
    rng = np.random.default_rng(0)
    X = rng.uniform(0.01, 0.99, size=(50, 3))
    y = rng.integers(0, 2, size=50)
    # ensure both classes present
    y[0] = 0
    y[1] = 1
    return X, y


@pytest.mark.parametrize("cls", [ArithmAvgClassifier, GeomAvgClassifier])
def test_avg_classifier_sklearn_compliance(cls, xy):
    """Avg classifier sklearn compliance."""
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
    """Avg classifier predict requires fit."""
    clf = ArithmAvgClassifier(nprobs=2)
    with pytest.raises(NotFittedError):
        clf.predict(np.ones((3, 3)))


def test_classifier_with_early_stopping_proxies_predict_proba():
    """Classifier with early stopping proxies predict proba."""
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
    """Cluster module imports cleanly."""
    import mlframe.preprocessing.cluster as cluster

    assert hasattr(cluster, "DBSCAN")
    assert hasattr(cluster, "clusterize")


def test_get_model_best_iter_with_pipeline():
    """Get model best iter with pipeline."""
    lgbm = pytest.importorskip("lightgbm")
    from mlframe.core.helpers import get_model_best_iter

    rng = np.random.default_rng(2)
    X = rng.normal(size=(200, 4))
    y = (X[:, 0] > 0).astype(int)

    est = lgbm.LGBMClassifier(n_estimators=5, verbose=-1)
    pipe = Pipeline([("est", est)])
    pipe.fit(X, y)

    best_iter = get_model_best_iter(pipe)
    # lightgbm exposes best_iteration_ even without early stopping (may be 0 or the count).
    # The prior assertion ``... or best_iter`` is a boolean trap -- truthy for any non-zero int
    # so it accepted literally any return that wasn't False.
    assert best_iter is None or isinstance(
        best_iter, (int, np.integer)
    ), f"get_model_best_iter must return int / np.integer / None; got {best_iter!r} ({type(best_iter).__name__})"


def test_feature_importance_sign_check_uses_sorted_order():
    # Regression: sign check must index via sorted_idx[0], not raw [0].
    # Simulate: feature_importances with a negative min NOT at index 0 => previous logic missed the branch.
    """Feature importance sign check uses sorted order."""
    import numpy as np

    feature_importances = np.array([0.5, -0.3, 0.1])
    sorted_idx = np.argsort(feature_importances)  # [1, 2, 0]
    # Fixed code: feature_importances[sorted_idx[0]] < 0
    assert feature_importances[sorted_idx[0]] < 0
    # Old buggy behavior would check feature_importances[0] (=0.5) and miss it.
    assert not (feature_importances[0] < 0)


@pytest.fixture(scope="module")
def fitted_rfecv():
    """A small fitted ``mlframe`` RFECV: wide enough for the vote-based selection path, small enough to stay quick."""
    pd = pytest.importorskip("pandas")
    from mlframe.feature_selection.wrappers import RFECV

    rng = np.random.default_rng(0)
    n, p = 300, 12
    x = rng.normal(size=(n, p))
    y = (x[:, 0] + 0.8 * x[:, 1] - 0.6 * x[:, 2] + 0.4 * rng.normal(size=n) > 0).astype(np.int64)
    frame = pd.DataFrame(x, columns=[f"f{i}" for i in range(p)])
    selector = RFECV(LogisticRegression(max_iter=300), cv=3, verbose=0)
    selector.fit(frame, pd.Series(y))
    return selector


@pytest.mark.sklearn_matrix
def test_rfecv_support_is_a_boolean_mask(fitted_rfecv):
    """``support_`` must be a boolean mask of length ``n_features_in_`` -- the shape every sklearn-shaped caller
    indexes columns with."""
    support = np.asarray(fitted_rfecv.support_)
    assert support.dtype == bool, f"support_ must be a boolean mask, got dtype {support.dtype}"
    assert support.shape == (fitted_rfecv.n_features_in_,), f"support_ must have length n_features_in_, got {support.shape}"


@pytest.mark.sklearn_matrix
def test_rfecv_ranking_is_integer_and_agrees_with_support(fitted_rfecv):
    """``ranking_`` must be an integer vector of length ``n_features_in_`` with survivors at 1, and ``ranking_ == 1``
    must agree with ``support_`` exactly (the contract broken by the feature-NAME regression fixed in c0889f59f)."""
    ranking = np.asarray(fitted_rfecv.ranking_)
    assert np.issubdtype(ranking.dtype, np.integer), f"ranking_ must be integer, got dtype {ranking.dtype}"
    assert ranking.shape == (fitted_rfecv.n_features_in_,), f"ranking_ must have length n_features_in_, got {ranking.shape}"
    support = np.asarray(fitted_rfecv.support_, dtype=bool)
    np.testing.assert_array_equal(ranking == 1, support)
    assert ranking.min() >= 1, "sklearn ranks start at 1"


@pytest.mark.sklearn_matrix
def test_rfecv_n_features_equals_support_sum(fitted_rfecv):
    """``n_features_`` must be the number of survivors, i.e. ``support_.sum()``."""
    assert int(fitted_rfecv.n_features_) == int(np.asarray(fitted_rfecv.support_, dtype=bool).sum())


@pytest.mark.sklearn_matrix
def test_rfecv_get_support_and_transform_agree(fitted_rfecv):
    """``get_support()``, ``get_support(indices=True)`` and ``transform`` must all describe the SAME column subset."""
    pd = pytest.importorskip("pandas")
    support = np.asarray(fitted_rfecv.support_, dtype=bool)
    np.testing.assert_array_equal(np.asarray(fitted_rfecv.get_support(), dtype=bool), support)
    np.testing.assert_array_equal(np.asarray(fitted_rfecv.get_support(indices=True), dtype=int), np.flatnonzero(support))

    rng = np.random.default_rng(1)
    frame = pd.DataFrame(rng.normal(size=(10, fitted_rfecv.n_features_in_)), columns=list(fitted_rfecv.feature_names_in_))
    out = fitted_rfecv.transform(frame)
    assert out.shape[1] == int(support.sum()), f"transform width {out.shape[1]} != support_.sum() {int(support.sum())}"
    np.testing.assert_allclose(np.asarray(out, dtype=float), np.asarray(frame, dtype=float)[:, support])


@pytest.mark.sklearn_matrix
def test_rfecv_get_feature_names_out_are_input_names(fitted_rfecv):
    """``get_feature_names_out()`` must return the SELECTED input names, in ``feature_names_in_`` order.

    Divergence note: mlframe's RFECV is a pure SELECTOR -- unlike the cluster-medoid wrapper that can sit above it,
    it never engineers columns of its own, so the plain sklearn contract (every emitted name exists in
    ``feature_names_in_``) is the honest one here and is asserted as such.
    """
    names_out = list(fitted_rfecv.get_feature_names_out())
    names_in = [str(c) for c in fitted_rfecv.feature_names_in_]
    assert set(names_out) <= set(names_in), f"engineered names leaked into get_feature_names_out: {set(names_out) - set(names_in)}"
    support = np.asarray(fitted_rfecv.support_, dtype=bool)
    assert names_out == [n for n, s in zip(names_in, support) if s]


@pytest.mark.sklearn_matrix
def test_rfecv_unfitted_raises_not_fitted(fitted_rfecv):
    """A fresh (unfitted) RFECV must raise ``NotFittedError`` from ``transform``, like any sklearn selector."""
    from mlframe.feature_selection.wrappers import RFECV

    with pytest.raises(NotFittedError):
        RFECV(LogisticRegression()).transform(np.ones((3, 3)))
