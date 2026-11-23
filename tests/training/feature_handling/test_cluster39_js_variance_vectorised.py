"""Cluster 39 regression: James-Stein sigma2 equals the pooled within-category variance (audit 2026-11-20, finding 39-F25)."""

import numpy as np

from mlframe.training.feature_handling.target_encoders import LeakageSafeEncoder


def test_sigma2_is_pooled_within_category_variance():
    """sigma2 = sum of squared deviations from each row's own category mean, over (n - K) (39-F25)."""
    cats = np.array(["a", "a", "a", "b", "b", "c", "c", "c"], dtype=object)
    y = np.array([1.0, 2.0, 3.0, 10.0, 12.0, 0.0, 0.5, 1.0])
    counts = {"a": 3, "b": 2, "c": 3}
    means = {"a": 2.0, "b": 11.0, "c": 0.5}
    sigma2, _tau2 = LeakageSafeEncoder._fit_james_stein_variance_params(cats, y, counts, means, None)
    # a: 1 + 0 + 1 = 2; b: 1 + 1 = 2; c: 0.25 + 0 + 0.25 = 0.5; pooled over n - K = 8 - 3 = 5 degrees of freedom.
    expected = (2 + 2 + 0.5) / (8 - 3)
    assert abs(sigma2 - expected) < 1e-12
