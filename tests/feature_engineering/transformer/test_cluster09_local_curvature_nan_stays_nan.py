"""Cluster 09 regression: a failed row in an underdetermined local-curvature fit stays NaN instead of becoming 0.0 (audit 2026-11-20, 09-F15)."""

import numpy as np

import mlframe.feature_engineering.transformer.local_curvature as lc


def _fail(*args, **kwargs):
    """Simulate a singular local design matrix."""
    raise np.linalg.LinAlgError("singular")


def test_failed_rows_stay_nan_when_the_quadratic_is_not_identifiable(monkeypatch):
    """d=4 with 10 neighbours cannot identify 15 quadratic coefficients: a row whose fit raises is NaN, not 0.0 (09-F15)."""
    monkeypatch.setattr(np.linalg, "lstsq", _fail)
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 4)).astype(np.float32)
    y = rng.normal(size=40).astype(np.float32)
    out = lc.compute_local_curvature_features(X, y, X_query=X[:5], splitter=None, seed=0, k_neighbors=10).to_numpy()
    assert np.isnan(out).all()


def test_failed_rows_still_zero_when_identifiable(monkeypatch):
    """With enough neighbours the documented zero fallback is unchanged (09-F15)."""
    monkeypatch.setattr(np.linalg, "lstsq", _fail)
    rng = np.random.default_rng(0)
    X = rng.normal(size=(30, 2)).astype(np.float32)
    y = rng.normal(size=30).astype(np.float32)
    out = lc.compute_local_curvature_features(X, y, X_query=X[:5], splitter=None, seed=0, k_neighbors=10).to_numpy()
    assert (out == 0.0).all()
