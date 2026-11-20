"""A device fault in the permutation-label draw is logged at warning level before the host draw takes over."""

from __future__ import annotations

import logging
import sys
import types

import numpy as np

from mlframe.feature_selection.filters import _perm_labels_device as pld
from mlframe.utils.log_throttle import reset_throttle_counts


def test_device_fault_warns_and_returns_none(monkeypatch, caplog):
    """A cupy that cannot build the kernels returns None with a warning."""
    monkeypatch.setitem(sys.modules, "cupy", types.ModuleType("cupy"))
    monkeypatch.setattr(pld, "_KERNELS", {})
    reset_throttle_counts()
    y = np.arange(64) % 2
    with caplog.at_level(logging.WARNING):
        assert pld.device_label_bits(y, 4, 7, 2, 1) is None
    assert any("device permutation draw failed" in r.getMessage() for r in caplog.records)


def test_missing_cupy_stays_quiet(monkeypatch, caplog):
    """No cupy at all is the normal host path and does not warn."""
    monkeypatch.setitem(sys.modules, "cupy", None)
    reset_throttle_counts()
    y = np.arange(64) % 2
    with caplog.at_level(logging.WARNING):
        assert pld.device_label_bits(y, 4, 7, 2, 1) is None
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]
