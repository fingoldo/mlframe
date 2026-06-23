"""``mlframe.metrics.core`` is a public deep-import that must work on a bare ``pip install mlframe``.

plotly ships only in the ``viz`` extra, so importing the module (and calling its non-plotting public
functions) must not require plotly; only a code path that actually asks for a plotly chart should fail,
and then with an actionable ``mlframe[viz]`` install hint.
"""
from __future__ import annotations

import builtins
import importlib
import sys

import numpy as np
import pytest


def _block_plotly(monkeypatch):
    """Make plotly un-importable for the duration of a test (simulate the CORE-install case)."""
    for name in list(sys.modules):
        if name == "plotly" or name.startswith("plotly."):
            monkeypatch.delitem(sys.modules, name, raising=False)

    real_import = builtins.__import__

    def _fake_import(name, *args, **kwargs):
        if name == "plotly" or name.startswith("plotly."):
            raise ImportError("No module named 'plotly' (simulated absence)")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fake_import)


def test_metrics_core_imports_without_plotly(monkeypatch):
    _block_plotly(monkeypatch)
    # A fresh re-import of the module must succeed even though plotly cannot be imported.
    monkeypatch.delitem(sys.modules, "mlframe.metrics.core", raising=False)
    core = importlib.import_module("mlframe.metrics.core")
    # A non-plotting public function works with plotly absent.
    y_true = np.array([0, 1, 0, 1, 1, 0], dtype=np.float64)
    y_pred = np.array([0.1, 0.9, 0.2, 0.8, 0.7, 0.3], dtype=np.float64)
    auc = core.fast_roc_auc(y_true, y_pred)
    assert 0.0 <= float(auc) <= 1.0


def test_require_plotly_raises_actionable_error(monkeypatch):
    _block_plotly(monkeypatch)
    core = importlib.import_module("mlframe.metrics.core")
    with pytest.raises(ImportError, match=r"mlframe\[viz\]"):
        core._require_plotly()
