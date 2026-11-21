"""The jackknife upper and lower endpoints weight the classes differently, so their spread is informative (audit 2026-11-20, finding 09-F4)."""
import numpy as np

from mlframe.feature_engineering.transformer.jackknife_endpoint_stability import compute_jackknife_endpoint_stability_features


def test_upper_and_lower_endpoints_are_not_the_same_model():
    """On imbalanced binary data the upper endpoint sits above the lower one: a pair of proportional weightings returned the same model."""
    rng = np.random.default_rng(0)
    n = 600
    X = rng.standard_normal((n, 3))
    y = (X[:, 0] + 0.5 * rng.standard_normal(n) > 1.2).astype(float)
    Xq = rng.standard_normal((20, 3))
    df = compute_jackknife_endpoint_stability_features(X, y, Xq, splitter=None, seed=0, task="binary", n_subsamples=4)
    arr = df.to_numpy()
    upper_mean, lower_mean = arr[:, 2], arr[:, 3]
    assert float(np.mean(upper_mean - lower_mean)) > 0.02
