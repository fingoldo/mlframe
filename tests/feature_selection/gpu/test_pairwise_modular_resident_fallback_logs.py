"""Every device-to-host fallback of the resident modular helpers logs a warning, so a faulted GPU path is visible instead of a silent slow run."""

from __future__ import annotations

import logging
import sys

import numpy as np
import pytest

from mlframe.feature_selection.filters import _gpu_strict_fe
from mlframe.feature_selection.filters import _pairwise_modular_resident as pmr
from mlframe.utils.log_throttle import reset_throttle_counts


@pytest.fixture(autouse=True)
def _no_cupy_and_enabled(monkeypatch):
    """Switch the device-born gate on and make ``import cupy`` fail, so each helper takes its except branch."""
    monkeypatch.setattr(_gpu_strict_fe, "fe_gpu_device_born_modular_enabled", lambda: True)
    monkeypatch.setitem(sys.modules, "cupy", None)
    reset_throttle_counts()


def _warned(caplog, needle):
    """Whether a WARNING record mentions ``needle``."""
    return any(r.levelno >= logging.WARNING and needle in r.getMessage() for r in caplog.records)


def test_combiner_mi_resident_logs_its_fallback(caplog):
    """A failing device baseline MI returns None and warns."""
    c = np.arange(100)
    y = np.arange(100) % 2
    with caplog.at_level(logging.WARNING):
        assert pmr.combiner_mi_resident(c, y, nbins=8, rank_binning=True) is None
    assert _warned(caplog, "combiner_mi_resident")


def test_residue_grid_mis_resident_logs_its_fallback(caplog):
    """A failing device residue grid returns None and warns."""
    c = np.arange(100)
    y = np.arange(100) % 2
    with caplog.at_level(logging.WARNING):
        assert pmr.residue_grid_mis_resident(c, y, [2, 3], nbins=8) is None
    assert _warned(caplog, "residue_grid_mis_resident")


def test_perm_null_residue_mis_resident_logs_its_fallback(caplog):
    """A failing device permutation null returns None and warns."""
    r = (np.arange(100) % 3).astype(np.float64)
    y = np.arange(100) % 2
    perms = [np.random.default_rng(i).permutation(100) for i in range(3)]
    with caplog.at_level(logging.WARNING):
        assert pmr.perm_null_residue_mis_resident(r, y, perms, eff_nbins=8, rank_binning=True) is None
    assert _warned(caplog, "perm_null_residue_mis_resident")


def test_enabled_probe_failure_logs(caplog, monkeypatch):
    """A raising gate probe is reported, not swallowed at debug level."""

    def boom():
        """Simulated probe fault."""
        raise RuntimeError("probe broke")

    monkeypatch.setattr(_gpu_strict_fe, "fe_gpu_device_born_modular_enabled", boom)
    with caplog.at_level(logging.WARNING):
        assert pmr.residue_grid_mis_resident(np.arange(10), np.arange(10) % 2, [2], nbins=4) is None
    assert _warned(caplog, "check failed")
