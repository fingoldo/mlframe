"""GPU-to-numba fallbacks in the SHAP proxy log a WARNING instead of staying silent."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from mlframe.feature_selection.shap_proxied_fs import _shap_proxy_explain as explain
from mlframe.feature_selection.shap_proxied_fs import _shap_proxy_interactions as inter
from mlframe.utils.log_throttle import reset_throttle_counts


def test_gpu_treeshap_failure_falls_back_to_numba_with_a_warning(monkeypatch, caplog):
    """A raising GPU TreeSHAP returns the numba result and logs a WARNING."""
    from mlframe.feature_selection.shap_proxied_fs import _shap_proxy_treeshap as ts
    from mlframe.feature_selection.shap_proxied_fs import _shap_proxy_treeshap_gpu as tsg

    monkeypatch.setattr(ts, "extract_ensemble", lambda est: object())
    monkeypatch.setattr(ts, "treeshap_phi_base_numba", lambda ens, X: ("numba-phi", 0.5))

    def boom(ens, X):
        """Simulate a device failure."""
        raise RuntimeError("device lost")

    monkeypatch.setattr(tsg, "treeshap_phi_base_gpu", boom)
    reset_throttle_counts()
    with caplog.at_level(logging.WARNING, logger=explain.logger.name):
        out = explain._treeshap_phi_and_base(object(), pd.DataFrame(np.zeros((2, 2))), use_gpu=True)
    assert out == ("numba-phi", 0.5)
    assert any(r.levelno == logging.WARNING and "GPU TreeSHAP failed" in r.getMessage() for r in caplog.records)


def test_gpu_interaction_tensor_failure_falls_back_to_numba_with_a_warning(monkeypatch, caplog):
    """A raising GPU interaction kernel falls back to numba and logs a WARNING."""
    monkeypatch.setattr(inter, "_fit_one", lambda *a, **k: object())

    def boom(*a, **k):
        """Simulate a device failure."""
        raise RuntimeError("cupy hiccup")

    sentinel = (np.zeros((2, 2, 2)), np.zeros(2))
    monkeypatch.setattr(inter, "_interaction_tensor_gpu", boom)
    monkeypatch.setattr(inter, "_interaction_tensor_numba", lambda *a, **k: sentinel)
    reset_throttle_counts()
    with caplog.at_level(logging.WARNING, logger=inter.logger.name):
        out = inter.compute_interaction_tensor(object(), pd.DataFrame(np.zeros((2, 2))), np.zeros(2), classification=False, backend="treeshap_gpu")
    assert out is sentinel
    assert any(r.levelno == logging.WARNING and "GPU interaction tensor path failed" in r.getMessage() for r in caplog.records)
