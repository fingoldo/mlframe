"""Audit 2026-11-20 finding 35-F43: three paths that had no test now run end to end.

predict_cdf with enforce_non_crossing=False, the gated mixture with soft_routing=True, and the gated outlier with calibrate_classifier=True.
"""

import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression

from mlframe.training.composite.distributional import CompositeDistributionEstimator
from mlframe.training.composite.gated_outlier import GatedOutlierEstimator
from mlframe.training.composite.gated_regression_mixture import GatedRegressionMixture


class _Stub(CompositeDistributionEstimator):
    """Skips fitting and serves a fixed, unsorted quantile matrix."""

    def _check_fitted(self):
        """Pretend the model is fitted."""

    def _quantile_matrix(self, X):
        """Return one unsorted quantile row per sample."""
        return np.tile(np.array([3.0, 1.0, 2.0]), (len(X), 1))


def test_predict_cdf_without_non_crossing_sorts_rows():
    """With enforce_non_crossing=False the CDF of an unsorted row still matches its sorted version."""
    est = _Stub(quantiles=[0.05, 0.5, 0.95], enforce_non_crossing=False)
    est.quantiles_ = np.array([0.05, 0.5, 0.95])
    cdf = est.predict_cdf(np.zeros((2, 1)), [2.5])
    np.testing.assert_allclose(cdf, [[0.5], [0.5]])


def test_gated_mixture_soft_routing_predicts_finite_values():
    """soft_routing=True blends the two branches; every prediction is finite and one per row."""
    rng = np.random.default_rng(0)
    n = 200
    X = rng.normal(size=(n, 2))
    y = X[:, 0] + 0.1 * rng.normal(size=n)
    labels = (X[:, 0] > 0).astype(int)
    m = GatedRegressionMixture(gate_classifier=LogisticRegression(), low_regressor=LinearRegression(),
                               high_regressor=LinearRegression(), n_splits=2, soft_routing=True)
    m.fit(X, y, labels)
    pred = m.predict(X)
    assert pred.shape == (n,)
    assert np.all(np.isfinite(pred))


def test_gated_outlier_calibrated_classifier_predicts_finite_values():
    """calibrate_classifier=True wraps the gate; the blended prediction is finite and one per row."""
    rng = np.random.default_rng(1)
    n = 300
    X = rng.normal(size=(n, 2))
    y = np.where(X[:, 0] > 0.5, 0.0, X[:, 1] + 1.0)
    est = GatedOutlierEstimator(regressor=LinearRegression(), classifier=LogisticRegression(), calibrate_classifier=True)
    est.fit(X, y)
    pred = est.predict(X)
    assert pred.shape == (n,)
    assert np.all(np.isfinite(pred))
