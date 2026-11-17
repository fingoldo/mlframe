"""Dispatcher routing (``_pick_backend`` / ``_shap_phi_and_base``): which backend is chosen, and that the result equals the official shap whichever it is."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_selection.shap_proxied_fs import _shap_proxy_explain as spe
from tests.feature_selection.shap_proxied.treeshap_vs_shap_common import (
    fit_lgb,
    fit_xgb,
    gpu_available,
    make_data,
    margin,
    shap_ref,
    xgb_ref64,
)


@pytest.fixture(params=["lgb", "xgb"])
def model_and_frame(request):
    """A small fitted model of each family and its DataFrame."""
    X, y = make_data()
    fit = fit_lgb if request.param == "lgb" else fit_xgb
    return fit(X, y), pd.DataFrame(X)


def _reference(model, df):
    """Reference phi: shap's float64 kernel (xgboost via the parsed trees, lightgbm directly)."""
    X = df.to_numpy()
    return xgb_ref64(model, X) if type(model).__module__.startswith("xgboost") else shap_ref(model, X)[0]


def test_narrow_data_routes_to_shap(model_and_frame):
    """Below the feature-count crossover the auto choice is the shap library."""
    m, df = model_and_frame
    assert df.shape[1] < spe._treeshap_numba_min_features()
    assert spe._pick_backend(m, df, "auto") == "shap"


def test_wide_data_routes_to_custom_kernel(model_and_frame, monkeypatch):
    """At or above the crossover (and below the GPU row*column bar) the numba kernel is chosen."""
    m, df = model_and_frame
    monkeypatch.setattr(spe, "_treeshap_numba_min_features", lambda: df.shape[1])
    assert spe._pick_backend(m, df, "auto") == "treeshap_numba"


@pytest.mark.skipif(not gpu_available(), reason="no CUDA device / cupy unavailable")
def test_large_data_routes_to_gpu(model_and_frame, monkeypatch):
    """Above 1e6 cells with a device present the GPU kernel is chosen."""
    m, df = model_and_frame
    monkeypatch.setattr(spe, "_treeshap_numba_min_features", lambda: 1)
    big = pd.DataFrame(np.zeros((1_000_000 // df.shape[1] + 1, df.shape[1]), dtype=np.float32))
    assert spe._pick_backend(m, big, "auto") == "treeshap_gpu"


def test_explicit_backends_are_honoured(model_and_frame):
    """Explicit names pass through unchanged."""
    m, df = model_and_frame
    for name in ("shap", "treeshap_numba", "treeshap_gpu"):
        assert spe._pick_backend(m, df, name) == name


@pytest.mark.parametrize("backend", ["auto", "shap", "treeshap_numba"])
def test_every_cpu_route_equals_shap(model_and_frame, backend, monkeypatch):
    """Whichever CPU route runs, phi equals shap (xgboost: its float32 native path within 1e-5, float64 kernel within 1e-9) and base + sum(phi) is the margin."""
    m, df = model_and_frame
    monkeypatch.setattr(spe, "_treeshap_numba_min_features", lambda: 1)
    phi, base = spe._shap_phi_and_base(m, df, backend=backend)
    is_xgb = type(m).__module__.startswith("xgboost")
    np.testing.assert_allclose(phi, _reference(m, df), rtol=0, atol=(1e-9 if backend != "shap" or not is_xgb else 1e-5))
    np.testing.assert_allclose(phi, shap_ref(m, df.to_numpy())[0], rtol=0, atol=1e-5)
    np.testing.assert_allclose(base + phi.sum(1), margin(m, df.to_numpy()), rtol=0, atol=5e-6)


@pytest.mark.skipif(not gpu_available(), reason="no CUDA device / cupy unavailable")
def test_forced_gpu_route_equals_shap(model_and_frame, monkeypatch):
    """The forced GPU route (float32 default) equals shap within 1e-4."""
    monkeypatch.delenv("MLFRAME_SHAP_TREESHAP_REAL", raising=False)
    m, df = model_and_frame
    phi, base = spe._shap_phi_and_base(m, df, backend="treeshap_gpu")
    np.testing.assert_allclose(phi, _reference(m, df), rtol=0, atol=1e-4)
    np.testing.assert_allclose(base + phi.sum(1), margin(m, df.to_numpy()), rtol=0, atol=1e-4)


def test_gpu_failure_falls_back_to_numba(model_and_frame, monkeypatch):
    """A device error in the GPU kernel is swallowed and the numba result is returned."""
    from mlframe.feature_selection.shap_proxied_fs import _shap_proxy_treeshap_gpu as g

    def boom(*a, **k):
        """Stand-in for a device failure."""
        raise RuntimeError("device error")

    monkeypatch.setattr(g, "treeshap_phi_base_gpu", boom)
    m, df = model_and_frame
    phi, _ = spe._shap_phi_and_base(m, df, backend="treeshap_gpu")
    np.testing.assert_allclose(phi, _reference(m, df), rtol=0, atol=1e-9)


def test_unsupported_model_falls_back_to_shap_even_when_forced():
    """A multiclass model under backend=treeshap_numba is not extracted; the shap library answers and its multiclass output is rejected with a clear message."""
    X, y = make_data(kind="multi")
    m = fit_lgb(X, y, "bin")
    with pytest.raises(ValueError, match="binary|multiclass|classes"):
        spe._shap_phi_and_base(m, pd.DataFrame(X), backend="treeshap_numba")


def test_declined_lightgbm_mode_gives_shap_values_under_auto(monkeypatch):
    """zero_as_missing LightGBM under auto routing is declined by the support check and answered by shap."""
    monkeypatch.setattr(spe, "_treeshap_numba_min_features", lambda: 1)
    X, y = make_data()
    X[::3, 0] = 0.0
    m = fit_lgb(X, y, zero_as_missing=True)
    phi, base = spe._shap_phi_and_base(m, pd.DataFrame(X), backend="auto")
    ref, ref_base = shap_ref(m, X)
    np.testing.assert_allclose(phi, ref, rtol=0, atol=1e-12)
    assert base == pytest.approx(ref_base, abs=1e-12)
