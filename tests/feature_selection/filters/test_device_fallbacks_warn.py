"""Device-path faults that fall back to the host must be logged at WARNING, not only at debug."""

from __future__ import annotations

import logging
import sys
import types

import numpy as np

from mlframe.utils.log_throttle import reset_throttle_counts


def _boom_cupy(monkeypatch):
    """Install a fake cupy whose attribute access raises, so every device path faults."""

    class _Mod(types.ModuleType):
        """cupy stand-in."""

        def __getattr__(self, name):
            """Fail every lookup."""
            raise RuntimeError("device fault")

    monkeypatch.setitem(sys.modules, "cupy", _Mod("cupy"))


def test_gain_stats_device_fault_warns(monkeypatch, caplog):
    """quantile_codes_columns_device logs a warning when the device faults."""
    from mlframe.feature_selection.filters import _gain_stats_codes_gpu as G

    reset_throttle_counts()
    _boom_cupy(monkeypatch)
    with caplog.at_level(logging.WARNING):
        assert G.quantile_codes_columns_device([np.arange(50.0)], 4) is None
    assert any(r.levelno >= logging.WARNING for r in caplog.records)


def test_fleuret_confirm_device_fault_warns(monkeypatch, caplog):
    """fleuret_confirm_batch logs a warning when the device faults."""
    from mlframe.feature_selection.filters import _fleuret_confirm_gpu as F

    reset_throttle_counts()
    _boom_cupy(monkeypatch)
    data = np.zeros((10, 3), dtype=np.int8)
    with caplog.at_level(logging.WARNING):
        got = F.fleuret_confirm_batch(data, np.array([2, 2, 2]), [(0,)], 2, [1], 2, [0.1], 1, [0], True, [1])
    assert got is None
    assert any(r.levelno >= logging.WARNING for r in caplog.records)


def test_gate_divergence_device_fault_warns(caplog):
    """gate_divergence_device logs a warning when the device faults."""
    from mlframe.feature_selection.filters import _gate_rank_divergence as D

    reset_throttle_counts()
    with caplog.at_level(logging.WARNING):
        assert D.gate_divergence_device(object(), object(), object(), [0], 4) is None
    assert any(r.levelno >= logging.WARNING for r in caplog.records)


def test_gbm_seeder_prefetch_failure_warns(caplog):
    """A prefetch that cannot start is reported at warning level."""
    from mlframe.feature_selection.filters import _gbm_seeder_prefetch as P

    class _Est:
        """Enabled estimator with unusable inputs."""

        fe_gbm_seeder_enable = True

    reset_throttle_counts()
    with caplog.at_level(logging.WARNING):
        P.prefetch_surrogate_seeds_early(_Est(), data=None, cols=[], X=None, target_indices=[0], categorical_vars=[], nbins=None, dtype=None)
    assert any(r.levelno >= logging.WARNING for r in caplog.records)
