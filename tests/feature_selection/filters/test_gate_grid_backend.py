"""The tau-grid backend of the conditional gate: the AUTO edge-binning mark, the environment overrides, and the host / device grids agreeing."""

from __future__ import annotations

import numpy as np
import pytest
from tests._cuda_require import require_cuda_device

from mlframe.feature_selection.filters import _conditional_gate_fe as gate
from mlframe.feature_selection.filters import _resident_candidate_mi_ktc as ktc


def test_auto_edge_binning_starts_at_the_validated_row_mark(monkeypatch):
    """Below the mark the rank binner stays (the planted-pair frame of 1000 rows lost a tau to the edge binner); from the mark up the edge binner is used; the environment can switch it off."""
    monkeypatch.delenv("MLFRAME_FE_GATE_AUTO_EDGE", raising=False)
    assert not gate._gate_auto_edge_enabled(gate._GATE_AUTO_EDGE_MIN_ROWS - 1)
    assert gate._gate_auto_edge_enabled(gate._GATE_AUTO_EDGE_MIN_ROWS)
    monkeypatch.setenv("MLFRAME_FE_GATE_AUTO_EDGE", "0")
    assert not gate._gate_auto_edge_enabled(10**7)


def test_grid_backend_is_forced_by_the_environment(monkeypatch):
    """``MLFRAME_FE_GATE_GRID_DEVICE`` forces either side of the tuned choice."""
    monkeypatch.setenv("MLFRAME_FE_GATE_GRID_DEVICE", "1")
    assert ktc.gate_grid_use_device(10) is True
    monkeypatch.setenv("MLFRAME_FE_GATE_GRID_DEVICE", "0")
    assert ktc.gate_grid_use_device(10**9) is False


def test_row_mark_fallback_before_any_sweep(monkeypatch):
    """With no tuning entry the choice is the row mark: host below it, device from it."""
    monkeypatch.delenv("MLFRAME_FE_GATE_GRID_DEVICE", raising=False)
    monkeypatch.setattr(ktc, "_GRID_SPEC", None)
    assert ktc.gate_grid_use_device(ktc.DEVICE_FALLBACK_ROWS - 1) is False
    assert ktc.gate_grid_use_device(ktc.DEVICE_FALLBACK_ROWS) is True


def test_rank_binned_device_grid_equals_the_host_scan():
    """The sweep's two callables (host scan and rank-binned device-born grid) agree to the sweep tolerance on a small frame."""
    pytest.importorskip("cupy")
    require_cuda_device()
    specs, y = ktc._grid_inputs({"n_samples": 4000})
    np.testing.assert_allclose(ktc._grid_device(specs, y), ktc._grid_host(specs, y), rtol=1e-6, atol=1e-6)
