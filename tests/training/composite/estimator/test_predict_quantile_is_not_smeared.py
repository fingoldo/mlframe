"""A quantile of T maps to the same quantile of y through the monotone inverse; the mean-correcting smearing must not be applied to it."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from mlframe.training.composite import CompositeTargetEstimator


class _MedianLinear(LinearRegression):
    """Linear model whose median of T is its point prediction (symmetric residuals)."""

    def predict_quantile(self, X, alpha=0.5):
        """Return the point prediction as the requested quantile."""
        return self.predict(X)


def test_median_of_a_log_composite_is_exp_of_the_t_median_not_the_mean():
    """Lognormal y: the median prediction equals exp(T-median), well below the smeared mean."""
    rng = np.random.default_rng(0)
    n = 20000
    x = rng.normal(size=n)
    X = pd.DataFrame({"x": x})
    y = np.exp(1.0 + 0.5 * x + rng.normal(size=n))
    est = CompositeTargetEstimator(base_estimator=_MedianLinear(), transform_name="log_y").fit(X, y)
    median = np.asarray(est.predict_quantile(X, alpha=0.5)).reshape(-1)
    expected = np.exp(est.base_estimator_.predict(X)) if hasattr(est, "base_estimator_") else None
    mean_pred = est.predict(X)
    assert median.mean() < 0.8 * mean_pred.mean()
    if expected is not None:
        np.testing.assert_allclose(median, expected, rtol=1e-6)
