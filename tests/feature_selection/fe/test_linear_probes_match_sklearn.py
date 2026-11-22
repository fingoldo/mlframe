"""The scaled linear probes give the predictions of the scikit-learn pipelines they replace (``make_pipeline(StandardScaler(), model)``)."""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_selection.filters._linear_probes import ScaledLogistic, ScaledOLS, ScaledRidge, cross_val_r2


def _reg(n, p, seed):
    """A regression frame with columns of different scales and one constant column."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p)) * rng.uniform(0.2, 5.0, size=p) + rng.normal(size=p)
    X[:, -1] = 3.0
    y = X[:, 0] - 0.5 * X[:, 1] + 0.3 * rng.standard_normal(n)
    return X, y


@pytest.mark.parametrize("n,p", [(120, 3), (800, 6), (3000, 10)])
def test_ols_matches_the_pipeline(n, p):
    """Predictions and R^2 equal ``StandardScaler`` + ``LinearRegression`` on held-out rows."""
    from sklearn.linear_model import LinearRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    X, y = _reg(n, p, n)
    tr, va = slice(0, n * 2 // 3), slice(n * 2 // 3, n)
    ref = make_pipeline(StandardScaler(), LinearRegression()).fit(X[tr], y[tr])
    got = ScaledOLS().fit(X[tr], y[tr])
    np.testing.assert_allclose(got.predict(X[va]), ref.predict(X[va]), rtol=1e-8, atol=1e-8)
    assert got.score(X[va], y[va]) == pytest.approx(ref.score(X[va], y[va]), abs=1e-9)


@pytest.mark.parametrize("alpha", [0.1, 1.0, 25.0])
def test_ridge_matches_the_pipeline(alpha):
    """Predictions equal ``StandardScaler`` + ``Ridge(alpha)``."""
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    X, y = _reg(900, 7, 5)
    ref = make_pipeline(StandardScaler(), Ridge(alpha=alpha)).fit(X[:600], y[:600])
    got = ScaledRidge(alpha).fit(X[:600], y[:600])
    np.testing.assert_allclose(got.predict(X[600:]), ref.predict(X[600:]), rtol=1e-7, atol=1e-7)


def test_binary_logistic_matches_the_pipeline():
    """Probabilities equal ``StandardScaler`` + ``LogisticRegression`` to the solver tolerance; labels may be strings."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(2)
    X = rng.standard_normal((1500, 5)) * [1, 2, 0.5, 3, 1]
    y = np.where(X[:, 0] - 0.4 * X[:, 3] + 0.8 * rng.standard_normal(1500) > 0, "pos", "neg")
    ref = make_pipeline(StandardScaler(), LogisticRegression(max_iter=200)).fit(X[:1000], y[:1000])
    got = ScaledLogistic().fit(X[:1000], y[:1000])
    assert list(got.classes_) == list(ref.classes_)
    np.testing.assert_allclose(got.predict_proba(X[1000:]), ref.predict_proba(X[1000:]), atol=2e-4)
    assert (got.predict(X[1000:]) == ref.predict(X[1000:])).mean() > 0.995


def test_multiclass_logistic_matches_the_pipeline():
    """Three classes: probabilities equal the pipeline's to the solver tolerance."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(3)
    X = rng.standard_normal((1800, 4))
    y = np.digitize(X[:, 0] + 0.5 * X[:, 1] + 0.7 * rng.standard_normal(1800), [-0.6, 0.6])
    ref = make_pipeline(StandardScaler(), LogisticRegression(max_iter=200)).fit(X[:1200], y[:1200])
    got = ScaledLogistic().fit(X[:1200], y[:1200])
    np.testing.assert_allclose(got.predict_proba(X[1200:]), ref.predict_proba(X[1200:]), atol=5e-4)


def test_cross_val_r2_averages_the_fold_scores():
    """The helper is the mean of the per-fold scores."""
    X, y = _reg(600, 4, 1)
    splits = [(np.arange(0, 400), np.arange(400, 600)), (np.arange(200, 600), np.arange(0, 200))]
    expected = np.mean([ScaledOLS().fit(X[a], y[a]).score(X[b], y[b]) for a, b in splits])
    assert cross_val_r2(ScaledOLS, X, y, splits) == pytest.approx(float(expected))
    assert cross_val_r2(ScaledOLS, X, y, []) is None


def test_standardisation_treats_a_column_constant_up_to_rounding_like_standard_scaler():
    """A column of one repeated float that sums to a mean off by rounding (std ~1e-17) keeps the spread 1 in both; the standardised matrix equals ``StandardScaler().fit_transform``."""
    from sklearn.preprocessing import StandardScaler

    from mlframe.feature_selection.filters._linear_probes import standardised

    rng = np.random.default_rng(0)
    X = np.column_stack([rng.standard_normal(500), np.full(500, 0.1), np.full(500, 7.3) + 1e-17, rng.uniform(1e5, 1e5 + 1, 500)])
    np.testing.assert_allclose(standardised(X), StandardScaler().fit_transform(X), rtol=1e-12, atol=1e-12)
