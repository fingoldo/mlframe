"""Audit 2026-11-20 finding 35-F27: a quantile predict resolves the causal lag once per batch, not once per quantile column.

``apply_smart_fallback`` takes a precomputed ``lag_vals`` and skips the resolution. Without it the lag column is looked up on every call.
"""

from types import SimpleNamespace

import numpy as np

from mlframe.training.composite.estimator import _soft_shrink


def _counting_resolver(monkeypatch):
    """Patch the lag-column resolver to count its calls and return a fixed column name."""
    calls = []

    def fake(self, X):
        """Record the call and report a lag column."""
        calls.append(1)
        return "lag"

    monkeypatch.setattr(_soft_shrink, "_resolve_lag_column", fake)
    return calls


def test_precomputed_lag_skips_per_column_resolution(monkeypatch):
    """With lag_vals supplied, nine calls (one per quantile column) resolve the lag zero times."""
    calls = _counting_resolver(monkeypatch)
    n = 6
    self_ = SimpleNamespace(fallback_predict="y_train_median")
    X = {"lag": np.arange(n, dtype=float)}
    base = np.arange(n, dtype=float)
    deep = np.array([True, False, True, False, True, False])
    ok = np.ones(n, dtype=bool)
    lag_once = _soft_shrink._resolve_lag_values(self_, X, base, n)
    before = len(calls)
    for _ in range(9):
        y = np.zeros(n)
        _soft_shrink.apply_smart_fallback(self_, y, deep, base, ok, X, {"y_train_median": 0.0}, lag_vals=lag_once)
    assert len(calls) == before


def test_default_call_still_resolves_the_lag(monkeypatch):
    """Without lag_vals the function resolves the lag itself, as predict() expects."""
    calls = _counting_resolver(monkeypatch)
    n = 4
    self_ = SimpleNamespace(fallback_predict="y_train_median")
    X = {"lag": np.arange(n, dtype=float)}
    y = np.zeros(n)
    _soft_shrink.apply_smart_fallback(self_, y, np.ones(n, dtype=bool), np.arange(n, dtype=float), np.ones(n, dtype=bool), X, {"y_train_median": 0.0})
    assert len(calls) >= 1
