"""The conditional-gate scan finds the same gates on the device (default) as on the host."""
import numpy as np
import pandas as pd
import pytest

from mlframe.feature_selection.filters._conditional_gate_fe import _gate_auto_device_enabled, hybrid_conditional_gate_fe_with_recipes


def _planted(seed: int = 1, n: int = 8000):
    """A target that switches between two columns on a third column's threshold."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, 12)), columns=[f"x{i}" for i in range(12)])
    y = ((np.where(X.x2 > 0.3, X.x0, X.x1) + rng.normal(0, 0.7, n)) > 0).astype(int)
    return X, y


def _scan(X, y):
    """Names and thresholds of the recipes the scan emits."""
    _, recipes = hybrid_conditional_gate_fe_with_recipes(X, y, cols=list(X.columns), top_k=4, seed=0, max_cols=200, k_gate=8, k_operand=10, subsample_n=0)
    return sorted((r.name, round(float(r.extra["tau"]), 6)) for r in recipes)


def test_opt_out_and_cpu_pin_keep_the_host_path(monkeypatch):
    """MLFRAME_FE_GPU_GATE_AUTO=0 and MLFRAME_FE_GPU_STRICT=0 both turn the automatic device path off."""
    monkeypatch.setenv("MLFRAME_FE_GPU_GATE_AUTO", "0")
    assert not _gate_auto_device_enabled()
    monkeypatch.delenv("MLFRAME_FE_GPU_GATE_AUTO")
    monkeypatch.setenv("MLFRAME_FE_GPU_STRICT", "0")
    assert not _gate_auto_device_enabled()


def test_device_scan_equals_host_scan(monkeypatch):
    """Same recipes (names and thresholds) with the device path on and off, and the planted gate is among them."""
    monkeypatch.delenv("MLFRAME_FE_GPU_STRICT", raising=False)
    monkeypatch.delenv("MLFRAME_FE_GPU_GATE_AUTO", raising=False)
    if not _gate_auto_device_enabled():
        pytest.skip("needs a CUDA device")
    X, y = _planted()
    on = _scan(X, y)
    monkeypatch.setenv("MLFRAME_FE_GPU_GATE_AUTO", "0")
    off = _scan(X, y)
    assert on == off
    assert any("x2" in name for name, _ in on)
