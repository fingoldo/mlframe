"""Regression pins: the residual pass keeps the CV policy, the registry report accepts array selections, the tuned CUDA block size stays in hardware limits."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

from mlframe.feature_selection.registry import _report_extract_shap_proxied_fs
from mlframe.feature_selection.shap_proxied_fs import _gpu_block_size
from mlframe.feature_selection.shap_proxied_fs._shap_proxied_fit_residual import _residual_pass2_shap


def test_residual_pass_forwards_the_search_cv_policy(monkeypatch):
    """Pass 2 partitions its OOF folds with the same group / time policy as pass 1 instead of a random KFold."""
    from mlframe.feature_selection.shap_proxied_fs import _shap_proxy_explain

    seen = {}

    def _fake(*args, **kwargs):
        """Record the call and return a minimal OOF-SHAP tuple."""
        seen.update(kwargs)
        return np.zeros((4, 2)), np.zeros(4), np.zeros(4), np.zeros((3, 2))

    monkeypatch.setattr(_shap_proxy_explain, "compute_shap_matrix", _fake)
    policy = object()
    me = SimpleNamespace(n_splits=3, _rng=None, tqdm=False, n_jobs=1, oof_shap_n_estimators=10, inner_n_jobs_cap=False, cache_dir=None, _search_cv_policy=policy)
    _residual_pass2_shap(me, object(), pd.DataFrame(np.zeros((4, 2))), np.zeros(4), lambda name: __import__("contextlib").nullcontext())
    assert seen["cv_policy"] is policy


def test_registry_report_handles_numpy_selected_features():
    """A selector exposing ``selected_features_`` as an ndarray still yields the kept/dropped reason map (``arr or []`` raised on it)."""
    sel = SimpleNamespace(shap_proxy_report_=None, selected_features_=np.array(["a", "c"]), feature_names_in_=np.array(["a", "b", "c"]))
    out = _report_extract_shap_proxied_fs(sel, ["a", "c"])
    assert out["reason_per_feature"] == {"a": "selected", "b": "dropped", "c": "selected"}


def test_tuned_block_size_outside_hardware_limit_falls_back(monkeypatch):
    """A cache entry of 4096 threads per block is not a launchable configuration; the default is used instead."""
    import mlframe.feature_selection.filters as filters

    ktc = SimpleNamespace(lookup=lambda key: {"block": 4096})
    monkeypatch.setattr(filters, "get_kernel_tuning_cache", lambda: ktc)
    assert _gpu_block_size.block_size_from_tuning_cache("k", "block", 256) == 256
    ktc2 = SimpleNamespace(lookup=lambda key: {"block": 128})
    monkeypatch.setattr(filters, "get_kernel_tuning_cache", lambda: ktc2)
    assert _gpu_block_size.block_size_from_tuning_cache("k", "block", 256) == 128
