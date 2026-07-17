"""Capability-gate tests for `xgb_dmatrix_reuse_capable`.

Mirror of the lgb counterpart: probes whether `xgb.QuantileDMatrix` has
both `set_label` and `set_weight` mutators required by the DMatrix reuse
shim. XGBoost >= 1.7 satisfies this.
"""

from __future__ import annotations

import pytest

xgb = pytest.importorskip("xgboost")

from mlframe.training import xgb_shim


def test_xgb_dmatrix_reuse_capable_returns_true_on_supported_build():
    """Xgb dmatrix reuse capable returns true on supported build."""
    assert xgb_shim.xgb_dmatrix_reuse_capable() is True


def test_xgb_dmatrix_reuse_capable_false_when_xgb_unavailable(monkeypatch):
    """Xgb dmatrix reuse capable false when xgb unavailable."""
    monkeypatch.setattr(xgb_shim, "_XGB_AVAILABLE", False)
    assert xgb_shim.xgb_dmatrix_reuse_capable() is False


def test_xgb_dmatrix_reuse_capable_false_when_set_label_missing(monkeypatch):
    """Xgb dmatrix reuse capable false when set label missing."""
    class _NoSetLabelQDM:
        """Groups tests covering no set label q d m."""
        def set_weight(self, *_a, **_kw):
            """Set weight."""
            return None

    monkeypatch.setattr(xgb_shim, "_XGB_AVAILABLE", True)
    monkeypatch.setattr(xgb_shim.xgb, "QuantileDMatrix", _NoSetLabelQDM)
    assert xgb_shim.xgb_dmatrix_reuse_capable() is False


def test_xgb_dmatrix_reuse_capable_false_when_set_weight_missing(monkeypatch):
    """Xgb dmatrix reuse capable false when set weight missing."""
    class _NoSetWeightQDM:
        """Groups tests covering no set weight q d m."""
        def set_label(self, *_a, **_kw):
            """Set label."""
            return None

    monkeypatch.setattr(xgb_shim, "_XGB_AVAILABLE", True)
    monkeypatch.setattr(xgb_shim.xgb, "QuantileDMatrix", _NoSetWeightQDM)
    assert xgb_shim.xgb_dmatrix_reuse_capable() is False


def test_xgb_dmatrix_reuse_capable_repeated_calls_are_consistent():
    """Xgb dmatrix reuse capable repeated calls are consistent."""
    first = xgb_shim.xgb_dmatrix_reuse_capable()
    second = xgb_shim.xgb_dmatrix_reuse_capable()
    third = xgb_shim.xgb_dmatrix_reuse_capable()
    assert first is second is third
