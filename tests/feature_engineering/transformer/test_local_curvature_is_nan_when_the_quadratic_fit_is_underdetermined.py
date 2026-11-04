"""local_curvature must not report a curvature from a quadratic fit that has fewer neighbours than coefficients."""
import numpy as np

from mlframe.feature_engineering.transformer.local_curvature import compute_local_curvature_features


def _fit(d, k):
    """Curvature features of 5 query rows on a smooth target."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((300, d)).astype(np.float32)
    y = (X[:, 0] ** 2 + X[:, 1]).astype(np.float32)
    return compute_local_curvature_features(X, y, X[:5], seed=1, k_neighbors=k).to_numpy()


def test_underdetermined_quadratic_fit_gives_nan_curvature_and_keeps_the_linear_value():
    """d = 8 needs 45 coefficients; 40 neighbours were interpolated exactly (resid_quad ~ 0) and the Hessian columns were arbitrary numbers."""
    out = _fit(8, 40)
    assert np.isnan(out[:, [0, 1, 2, 4]]).all() and np.isfinite(out[:, 3]).all()


def test_identifiable_quadratic_fit_is_unchanged():
    """d = 3 has 10 coefficients; 40 neighbours identify them and every column is finite."""
    assert np.isfinite(_fit(3, 40)).all()
