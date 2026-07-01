"""Regression sensor: ``run_pysr_fe`` delegates the single-target case to
``bruteforce.run_pysr_feature_engineering`` so the leakage-free OOF + Julia-lock
plumbing is reachable through the legacy entry point. The multi-target case
keeps the legacy in-place path because the new entry point is single-output.

A2#22 / fe-critique row 22 backed delegation: previously the legacy body
duplicated PySR fit logic with no reuse and emitted a generic
``DeprecationWarning`` without action; callers that picked the legacy entry
silently lost preset / OOF / random_state functionality.
"""
from __future__ import annotations

import sys
import warnings
from types import ModuleType
from unittest.mock import MagicMock

import numpy as np
import polars as pl
import pytest


def test_single_target_delegates_to_bruteforce(monkeypatch):
    """One ``target_*`` column -> body calls run_pysr_feature_engineering with
    the right kwargs (target_col, drop_columns, sample_size)."""
    calls: dict = {}

    def fake_run_pysr_feature_engineering(df, *, target_col, drop_columns, sample_size, pysr_params_override, leakage_free, verbose):
        calls["target_col"] = target_col
        calls["drop_columns"] = list(drop_columns)
        calls["sample_size"] = sample_size
        calls["leakage_free"] = leakage_free
        calls["verbose"] = verbose
        calls["override_keys"] = sorted(pysr_params_override)
        fake = MagicMock()
        fake.equations_ = [1, 2, 3]
        return fake

    fake_bruteforce = ModuleType("mlframe.feature_engineering.bruteforce")
    fake_bruteforce.run_pysr_feature_engineering = fake_run_pysr_feature_engineering
    monkeypatch.setitem(sys.modules, "mlframe.feature_engineering.bruteforce", fake_bruteforce)

    df = pl.DataFrame({
        "x0": np.arange(10, dtype=np.float64),
        "x1": np.arange(10, dtype=np.float64) * 2.0,
        "target_y": np.arange(10, dtype=np.float64),
    })

    from mlframe.feature_engineering.basic import run_pysr_fe

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = run_pysr_fe(df, nsamples=8, timeout_mins=2, fill_nans=True)

    assert calls["target_col"] == "target_y"
    assert calls["drop_columns"] == []
    assert calls["sample_size"] == 8
    assert calls["leakage_free"] is False
    assert calls["verbose"] == 0
    assert "timeout_in_seconds" in calls["override_keys"]
    assert "binary_operators" in calls["override_keys"]
    assert hasattr(result, "equations_")

    deprecations = [str(w.message) for w in caught if issubclass(w.category, DeprecationWarning)]
    assert any("delegating to" in m for m in deprecations), deprecations


def test_multi_target_keeps_legacy_path(monkeypatch):
    """Two ``target_*`` columns -> body falls back to the legacy PySR fit and
    surfaces the multi-target migration warning."""
    fake_pysr_mod = ModuleType("pysr")

    class _FakeReg:
        def __init__(self, **kw):
            self._kw = kw

        def fit(self, X, y):
            raise RuntimeError("multi-target legacy path reached")

    fake_pysr_mod.PySRRegressor = _FakeReg
    monkeypatch.setitem(sys.modules, "pysr", fake_pysr_mod)

    df = pl.DataFrame({
        "x0": np.arange(5, dtype=np.float64),
        "target_a": np.arange(5, dtype=np.float64),
        "target_b": np.arange(5, dtype=np.float64) * 2.0,
    })

    from mlframe.feature_engineering.basic import run_pysr_fe

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(RuntimeError, match="multi-target legacy path reached"):
            run_pysr_fe(df, nsamples=5, timeout_mins=1)

    deprecations = [str(w.message) for w in caught if issubclass(w.category, DeprecationWarning)]
    assert any("multi-target" in m for m in deprecations), deprecations
