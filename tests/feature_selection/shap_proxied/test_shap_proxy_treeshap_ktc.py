"""The TreeSHAP backend choice (numba table kernel or device table kernel) reads a per-host kernel-tuning entry and falls back to a measured work mark."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_selection.shap_proxied_fs import _shap_proxy_treeshap_ktc as ktc


def test_fallback_work_mark_is_rows_times_leaves():
    """Small problems stay on the host, large ones go to the device, and a bigger ensemble moves the mark to fewer rows."""
    assert ktc._fallback(100, 20) == "numba"
    assert ktc._fallback(50_000, 20) == "gpu"
    assert ktc._fallback(2_000, 100) == "gpu" and ktc._fallback(2_000, 20) == "numba"


def test_entry_is_registered_with_both_axes():
    """The tuner entry exists and sweeps rows and trees."""
    assert ktc._SPEC is not None
    assert set(ktc._SPEC.axes) == {"n_rows", "trees"}


def test_pick_backend_follows_the_choice(monkeypatch):
    """``_pick_backend`` asks the entry with the row and tree counts of the model."""
    xgb = pytest.importorskip("xgboost")
    from mlframe.feature_selection.shap_proxied_fs import _shap_proxy_explain as ex

    rng = np.random.default_rng(0)
    X = rng.standard_normal((300, 5)).astype(np.float32)
    model = xgb.XGBRegressor(n_estimators=7, max_depth=3, verbosity=0).fit(X, X[:, 0])
    assert ex._n_trees(model) == 7
    seen = []
    monkeypatch.setattr(ktc, "use_gpu", lambda n, t: seen.append((n, t)) or False)
    monkeypatch.setattr("mlframe.feature_selection.shap_proxied_fs._shap_proxy_treeshap_gpu.gpu_treeshap_available", lambda: True)
    assert ex._pick_backend(model, pd.DataFrame(X), "auto") == "treeshap_numba"
    assert seen == [(300, 7)]
    monkeypatch.setattr(ktc, "use_gpu", lambda n, t: True)
    assert ex._pick_backend(model, pd.DataFrame(X), "auto") == "treeshap_gpu"
