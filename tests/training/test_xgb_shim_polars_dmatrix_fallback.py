"""Regression: ``_build_quantile_dmatrix`` must fall back to a pandas view of a polars ``X`` when
the installed XGBoost's ``QuantileDMatrix`` data-proxy layer does not accept polars directly.

CI caught this: XGBoost >=2.0 is the floor here, but an older 2.x resolved on py3.9 (tighter
dependency constraints than newer Python versions) raises ``TypeError: Value type is not
supported for data iterator:<class 'polars.dataframe.frame.DataFrame'>`` deep inside
``QuantileDMatrix`` when handed a bare polars frame -- 13 training-suite tests failed with this
exact error on py3.9/xgboost-2.x CI shards while passing locally on a newer XGBoost.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("polars")
pytest.importorskip("xgboost")

import polars as pl

import mlframe.training.xgb_shim as xgb_shim


@pytest.fixture(autouse=True)
def _reset_polars_support_cache():
    """Each test starts with the module-level probe cache cleared, and restores it after."""
    orig = xgb_shim._XGB_POLARS_DMATRIX_SUPPORT
    xgb_shim._XGB_POLARS_DMATRIX_SUPPORT = None
    yield
    xgb_shim._XGB_POLARS_DMATRIX_SUPPORT = orig


def test_probe_caches_true_when_polars_natively_supported(monkeypatch):
    """A real (or stubbed-successful) QuantileDMatrix probe caches True and is only probed once."""
    calls = []

    class _FakeDMatrix:
        """Stub QuantileDMatrix that records every X it was constructed with."""

        def __init__(self, X, **kwargs):
            """Record the constructor's X argument."""
            calls.append(X)

    monkeypatch.setattr(xgb_shim.xgb, "QuantileDMatrix", _FakeDMatrix)
    assert xgb_shim._xgb_dmatrix_accepts_polars() is True
    assert xgb_shim._xgb_dmatrix_accepts_polars() is True
    assert len(calls) == 1, "the probe must run at most once per process (cached)"


def test_probe_caches_false_when_polars_unsupported(monkeypatch):
    """A QuantileDMatrix that raises TypeError on a polars frame (older XGBoost) caches False."""

    def _raising_dmatrix(X, **kwargs):
        """Stub QuantileDMatrix that always raises the polars-unsupported TypeError."""
        raise TypeError("Value type is not supported for data iterator:<class 'polars.dataframe.frame.DataFrame'>")

    monkeypatch.setattr(xgb_shim.xgb, "QuantileDMatrix", _raising_dmatrix)
    assert xgb_shim._xgb_dmatrix_accepts_polars() is False


def test_build_quantile_dmatrix_converts_polars_to_pandas_when_unsupported(monkeypatch):
    """_build_quantile_dmatrix must hand QuantileDMatrix a pandas frame, not the raw polars one,
    when the polars-support probe says False -- verifies the actual call site, not just the probe."""
    X = pl.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
    y = np.array([0, 1, 0])

    monkeypatch.setattr(xgb_shim, "_xgb_dmatrix_accepts_polars", lambda: False)

    received = {}

    class _FakeDMatrix:
        """Stub QuantileDMatrix that records the X it was constructed with."""

        def __init__(self, X_arg, **kwargs):
            """Record the constructor's X argument."""
            received["X"] = X_arg

    monkeypatch.setattr(xgb_shim.xgb, "QuantileDMatrix", _FakeDMatrix)
    xgb_shim._build_quantile_dmatrix(X, y, None)

    assert isinstance(received["X"], pd.DataFrame), "polars X must be converted to pandas on the unsupported path"
    assert list(received["X"].columns) == ["a", "b"]


def test_build_quantile_dmatrix_passes_polars_through_when_supported(monkeypatch):
    """The verified-XGBoost-3.x fast path must NOT pay the pandas-conversion cost: the raw polars
    frame is passed straight through to QuantileDMatrix."""
    X = pl.DataFrame({"a": [1.0, 2.0, 3.0]})
    y = np.array([0, 1, 0])

    monkeypatch.setattr(xgb_shim, "_xgb_dmatrix_accepts_polars", lambda: True)

    received = {}

    class _FakeDMatrix:
        """Stub QuantileDMatrix that records the X it was constructed with."""

        def __init__(self, X_arg, **kwargs):
            """Record the constructor's X argument."""
            received["X"] = X_arg

    monkeypatch.setattr(xgb_shim.xgb, "QuantileDMatrix", _FakeDMatrix)
    xgb_shim._build_quantile_dmatrix(X, y, None)

    assert received["X"] is X, "the polars fast path must pass X through unconverted"
