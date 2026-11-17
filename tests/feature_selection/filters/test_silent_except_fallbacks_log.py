"""Handlers that substituted a value without a trace now say so (throttled WARNING) when the substitute is used."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pytest

from mlframe.utils.log_throttle import reset_throttle_counts


@pytest.fixture(autouse=True)
def _fresh_throttle() -> None:
    """Each test sees its first-occurrence log, whatever ran before."""
    reset_throttle_counts()


class _Broken:
    """An object whose every attribute read fails."""

    def __getattr__(self, name: str) -> Any:
        """Raise on any attribute read."""
        raise RuntimeError("unreadable")


def test_unreadable_cuda_device_id_is_logged(caplog: pytest.LogCaptureFixture) -> None:
    """The resident-cache key falls back to device 0 and the log names the substitute."""
    from mlframe.feature_selection.filters.info_theory._cmi_resident_cache import _current_device_id

    with caplog.at_level(logging.WARNING):
        assert _current_device_id(_Broken()) == 0
    assert any("device 0 keys the resident cache" in r.getMessage() for r in caplog.records)


def test_unreadable_tree_count_is_logged(caplog: pytest.LogCaptureFixture) -> None:
    """The TreeSHAP backend choice sees 0 trees and the log says so."""
    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_explain import _n_trees

    with caplog.at_level(logging.WARNING):
        assert _n_trees(_Broken()) == 0
    assert any("tree count of the model unreadable" in r.getMessage() for r in caplog.records)


def test_unreadable_catboost_params_is_logged(caplog: pytest.LogCaptureFixture) -> None:
    """A CatBoost model whose params cannot be read is treated as symmetric and the log says so."""
    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_treeshap_catboost import is_nonsymmetric_catboost

    with caplog.at_level(logging.WARNING):
        assert is_nonsymmetric_catboost(_Broken()) is False
    assert any("treated as symmetric" in r.getMessage() for r in caplog.records)


def test_unreadable_catboost_classes_is_logged(caplog: pytest.LogCaptureFixture) -> None:
    """A CatBoost model whose classes_ cannot be read is not explained here and the log says so."""
    from mlframe.inference.tree_shap import TreeExplainer

    with caplog.at_level(logging.WARNING):
        assert TreeExplainer._catboost_single_output(_Broken()) is False
    assert any("classes_ unreadable" in r.getMessage() for r in caplog.records)


def test_failed_device_quantile_binning_is_logged(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    """A device binning failure returns False (the host binner runs) and logs the cause."""
    from mlframe.feature_selection.filters import _mi_greedy_cmi_fe as mod

    monkeypatch.setenv("MLFRAME_QBIN_COLUMNS_DEVICE", "1")
    monkeypatch.setitem(__import__("sys").modules, "cupy", None)
    with caplog.at_level(logging.WARNING):
        assert mod._qbin_block_on_device(np.zeros((3, 4), dtype=np.float32), 4, np.zeros((3, 4), dtype=np.int16)) is False
    assert any("device quantile binning failed" in r.getMessage() for r in caplog.records)
