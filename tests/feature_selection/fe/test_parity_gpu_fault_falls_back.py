"""A device fault inside the parity grid scan returns None (host scan) with a warning instead of crashing the fit."""

from __future__ import annotations

import logging
import sys
import types

import numpy as np

from mlframe.feature_selection.filters import _parity_gpu as gpu
from mlframe.feature_selection.filters._parity_fe import _parity_grid
from mlframe.utils.log_throttle import reset_throttle_counts


def test_device_fault_returns_none_and_warns(monkeypatch, caplog):
    """A cupy whose module build raises must not propagate out of the scan."""
    monkeypatch.setitem(sys.modules, "cupy", types.ModuleType("cupy"))
    reset_throttle_counts()
    rng = np.random.default_rng(0)
    col = rng.random(500) * 100.0
    y = (np.floor(col / 7.0) % 2).astype(np.int64)
    with caplog.at_level(logging.WARNING):
        out = gpu.parity_mi_grids_gpu([col], y, 2, [_parity_grid(col)])
    assert out is None
    assert any("device parity scan failed" in r.getMessage() for r in caplog.records)
