"""A degenerate fold gives NaN distance features, not 0.0 (audit 2026-11-20, finding 09-F6)."""
import numpy as np

from mlframe.feature_engineering.transformer.density_weighted_smote import compute_density_weighted_smote_features


def test_single_positive_gives_nan_not_zero():
    """With one positive row there is no positive manifold to measure against: the output must say so with NaN."""
    rng = np.random.default_rng(0)
    n = 120
    X = rng.standard_normal((n, 3))
    y = np.zeros(n)
    y[0] = 1.0
    Xq = rng.standard_normal((10, 3))
    df = compute_density_weighted_smote_features(X, y, Xq, splitter=None, seed=0, task="binary")
    assert np.isnan(df.to_numpy()).all()
