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
    assert xgb_shim.xgb_dmatrix_reuse_capable() is True


def test_xgb_dmatrix_reuse_capable_false_when_xgb_unavailable(monkeypatch):
    monkeypatch.setattr(xgb_shim, "_XGB_AVAILABLE", False)
    assert xgb_shim.xgb_dmatrix_reuse_capable() is False


def test_xgb_dmatrix_reuse_capable_false_when_set_label_missing(monkeypatch):
    class _NoSetLabelQDM:
        def set_weight(self, *_a, **_kw):
            return None

    monkeypatch.setattr(xgb_shim, "_XGB_AVAILABLE", True)
    monkeypatch.setattr(xgb_shim.xgb, "QuantileDMatrix", _NoSetLabelQDM)
    assert xgb_shim.xgb_dmatrix_reuse_capable() is False


def test_xgb_dmatrix_reuse_capable_false_when_set_weight_missing(monkeypatch):
    class _NoSetWeightQDM:
        def set_label(self, *_a, **_kw):
            return None

    monkeypatch.setattr(xgb_shim, "_XGB_AVAILABLE", True)
    monkeypatch.setattr(xgb_shim.xgb, "QuantileDMatrix", _NoSetWeightQDM)
    assert xgb_shim.xgb_dmatrix_reuse_capable() is False


def test_xgb_dmatrix_reuse_capable_repeated_calls_are_consistent():
    first = xgb_shim.xgb_dmatrix_reuse_capable()
    second = xgb_shim.xgb_dmatrix_reuse_capable()
    third = xgb_shim.xgb_dmatrix_reuse_capable()
    assert first is second is third
