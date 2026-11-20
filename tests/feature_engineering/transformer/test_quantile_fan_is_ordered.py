"""The three quantile heads of the regression spread fan are ordered q10 <= q50 <= q90 on every query row (audit 2026-11-20, finding 09-F5)."""
import numpy as np

from mlframe.feature_engineering.transformer.quantile_spread_fan import compute_quantile_spread_fan_features


def test_fan_quantiles_never_cross():
    """Independently fit quantile models can cross on noisy data; the fan must still report an ordered band."""
    rng = np.random.default_rng(1)
    n = 300
    X = rng.standard_normal((n, 2))
    y = rng.standard_t(2.0, n)
    Xq = rng.standard_normal((50, 2))
    df = compute_quantile_spread_fan_features(X, y, Xq, splitter=None, seed=0, task="regression")
    arr = df.to_numpy()
    q10, q50, q90 = arr[:, 0], arr[:, 1], arr[:, 2]
    assert np.all(q10 <= q50 + 1e-6) and np.all(q50 <= q90 + 1e-6)
