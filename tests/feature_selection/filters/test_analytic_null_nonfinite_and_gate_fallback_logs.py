"""Regression tests: non-finite MI in the analytic null and throttled gate-fallback warnings."""
import logging

import numpy as np

from mlframe.feature_selection.filters import _batch_mi_noise_gate_bits as bits_mod
from mlframe.feature_selection.filters import _batch_mi_noise_gate_fused as fused_mod
from mlframe.feature_selection.filters._analytic_mi_null import analytic_batch_noise_gate, analytic_mi_null
from mlframe.utils.log_throttle import reset_throttle_counts


def test_analytic_mi_null_nan_mi_is_non_significant():
    """A NaN observed MI gets p=1.0 (like the batch path), never a NaN p-value."""
    _, p = analytic_mi_null(float("nan"), 50_000, 10, 2)
    assert p == 1.0


def test_analytic_batch_noise_gate_nan_mi_is_zeroed():
    """A NaN observed MI leaves the gate as 0.0, not as NaN."""
    codes = np.zeros((50_000, 2), dtype=np.int8)
    out = analytic_batch_noise_gate(codes, np.array([np.nan, 0.5]), np.zeros(50_000, dtype=np.int64), 50_000, 0.95, bx_per_col=np.array([5, 5]), by=2)
    assert out[0] == 0.0


def test_bits_gate_failure_warning_is_throttled(caplog):
    """Repeated device failures of the bit-set gate log a bounded number of warnings."""
    reset_throttle_counts("noise_gate.bits_unavailable")
    with caplog.at_level(logging.WARNING, logger=bits_mod.logger.name):
        for _ in range(12):
            assert bits_mod.bits_gate_mi(object(), np.array([2]), object(), 2) is None
    assert 1 <= sum("bit-set noise gate unavailable" in r.getMessage() for r in caplog.records) < 12


def test_fused_gate_failure_warning_is_throttled(caplog, monkeypatch):
    """Repeated launch failures of the fused gate log a bounded number of warnings."""
    from mlframe.feature_selection.filters import _batch_mi_noise_gate_kernels as k

    reset_throttle_counts("noise_gate.fused_failed")
    monkeypatch.setattr(fused_mod, "fused_gate_enabled", lambda: True)
    monkeypatch.setattr(fused_mod, "cuda_available", lambda: True)
    monkeypatch.setattr(k, "_cuda_shared_mem_per_block", lambda: 49152)
    monkeypatch.setattr(fused_mod, "_cuda_fused_gate_kernel_factory", lambda: object())
    monkeypatch.setattr(fused_mod, "_FUSED_KERNEL", None)
    with caplog.at_level(logging.WARNING, logger=fused_mod.logger.name):
        for _ in range(12):
            assert fused_mod.fused_gate_mi(object(), np.array([2]), None, None, None, np.array([0.5, 0.5]), 10, 1, 2, 2, 32, False) is None
    assert 1 <= sum("fused noise-gate kernel failed" in r.getMessage() for r in caplog.records) < 12
