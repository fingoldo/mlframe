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


def test_device_grid_mi_equals_host_on_tied_mask_columns(monkeypatch):
    """The per-column MI of the tau grid is the host's on mask columns whose zeros are half of the rows (the percentile-edge binner differed by up to 0.49 there)."""
    from mlframe.feature_selection.filters import _conditional_gate_fe as gate

    monkeypatch.delenv("MLFRAME_FE_GPU_STRICT", raising=False)
    monkeypatch.delenv("MLFRAME_FE_GPU_GATE_AUTO", raising=False)
    if not _gate_auto_device_enabled():
        pytest.skip("needs a CUDA device")
    rng = np.random.default_rng(3)
    n = 1000
    c, a, b = (rng.normal(size=n) for _ in range(3))
    y = ((np.where(c > 0.2, a, b) + rng.normal(0, 0.5, n)) > 0).astype(np.int64)
    taus = np.quantile(c, np.linspace(0.1, 0.9, 9))
    pending = [("mask", ("a", "c"), (c, a), taus, None), ("select", ("a", "b", "c"), (c, a, b), taus, None)]
    device = gate._gate_device_born_all_mi(pending, y, 12)
    assert device is not None
    host = gate._gate_grid_mi(np.ascontiguousarray(np.concatenate([gate._gate_build_feats(m, o, t) for (m, _c, o, t, _b) in pending], axis=1)), y, 12)
    assert np.abs(np.asarray(device, dtype=np.float64) - np.asarray(host, dtype=np.float64)).max() < 1e-6
