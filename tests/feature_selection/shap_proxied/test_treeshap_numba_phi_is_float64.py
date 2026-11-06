"""The numba TreeSHAP path returns float64 phi and float64 leaf values even when the crit dtype is relaxed to float32.

``treeshap_phi_base_numba`` documents phi as double precision, and the cupy twin already is. The numba twin took its buffers from
``crit_np_dtype()``, which is float32 under the default ``MLFRAME_CRIT_DTYPE_RELAXED``, so every phi accumulation was rounded to
float32 and the additivity ``base + phi.sum(1) == margin`` drifted by about 1e-7 per term.
"""

from __future__ import annotations

import numpy as np
import pytest

xgb = pytest.importorskip("xgboost")


def _fit_small_xgb():
    """Fit a small xgboost regressor and return it with its float32 design matrix."""
    rng = np.random.default_rng(7)
    X = rng.normal(size=(220, 9)).astype(np.float32)
    y = X[:, 0] * 2.0 - X[:, 3] + 0.3 * rng.normal(size=220)
    model = xgb.XGBRegressor(n_estimators=40, max_depth=6, learning_rate=0.2)
    model.fit(X, y)
    return model, X


def test_numba_treeshap_phi_is_float64_under_relaxed_crit_dtype(monkeypatch):
    """phi and leaf values stay float64 when MLFRAME_CRIT_DTYPE_RELAXED=1, and additivity holds."""
    monkeypatch.setenv("MLFRAME_CRIT_DTYPE_RELAXED", "1")
    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_treeshap import extract_ensemble, treeshap_phi_base_numba

    model, X = _fit_small_xgb()
    ensemble = extract_ensemble(model)
    assert ensemble is not None

    phi, base = treeshap_phi_base_numba(ensemble, X)

    assert phi.dtype == np.float64
    assert ensemble.values.dtype == np.float64
    assert ensemble.node_sample_weight.dtype == np.float64
    # xgboost returns the margin in float32, so its own rounding (about 5e-7 at margin magnitude 4) sets the floor here.
    margin = model.predict(X, output_margin=True).astype(np.float64)
    np.testing.assert_allclose(base + phi.sum(axis=1), margin, rtol=0, atol=5e-6)
