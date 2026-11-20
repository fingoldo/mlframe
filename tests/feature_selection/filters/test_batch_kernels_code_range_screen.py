"""The CPU batched FE-candidate MI kernels reject out-of-range codes like their GPU twins, and a failed kernel selection is logged."""

from __future__ import annotations

import logging

import numpy as np
import pytest

from mlframe.feature_selection.filters.info_theory import (
    batch_mi_with_noise_gate,
    batch_mi_with_noise_gate_v2,
    select_batch_mi_kernel,
)
from mlframe.feature_selection.filters.info_theory._batch_kernels_v3 import batch_mi_with_noise_gate_v3
from mlframe.utils.log_throttle import reset_throttle_counts

_KERNELS = [batch_mi_with_noise_gate, batch_mi_with_noise_gate_v2, batch_mi_with_noise_gate_v3]


def _call(kernel, disc, yc, nbins=4, k_y=3):
    """Run ``kernel`` on ``disc`` against the class vector ``yc`` with a fixed permutation gate."""
    fy = np.bincount(np.clip(yc, 0, k_y - 1), minlength=k_y).astype(np.float64) / yc.shape[0]
    return kernel(disc, np.full(disc.shape[1], nbins, dtype=np.int64), yc, yc.copy(), fy, 5, np.uint64(0), 0.0, False, np.int32, np.int16)


def _data(n=200, k=3):
    """A valid code matrix and class vector."""
    rng = np.random.default_rng(0)
    return rng.integers(0, 4, size=(n, k)).astype(np.int8), rng.integers(0, 3, size=n).astype(np.int64)


@pytest.mark.parametrize("kernel", _KERNELS)
def test_valid_codes_pass(kernel):
    """In-range codes give a finite MI per column."""
    disc, yc = _data()
    assert np.all(np.isfinite(_call(kernel, disc, yc)))


@pytest.mark.parametrize("kernel", _KERNELS)
def test_negative_feature_code_is_rejected(kernel):
    """A -1 sentinel in a feature column raises instead of wrapping into the last bin."""
    disc, yc = _data()
    disc[7, 1] = -1
    with pytest.raises(ValueError, match="out|lie in"):
        _call(kernel, disc, yc)


@pytest.mark.parametrize("kernel", _KERNELS)
def test_negative_target_code_is_rejected(kernel):
    """A -1 in the target classes raises instead of wrapping into the last class."""
    disc, yc = _data()
    yc[3] = -1
    with pytest.raises(ValueError, match="out|lie in"):
        _call(kernel, disc, yc)


def test_failed_kernel_selection_is_logged_and_defaults_to_v3(monkeypatch, caplog):
    """A kernel-tuning failure logs a warning and still returns the default kernel."""
    import pyutilz.performance.kernel_tuning.cache as ktc

    def _boom(*a, **k):
        """Stand-in for a broken tuning cache."""
        raise RuntimeError("tuning cache unavailable")

    monkeypatch.delenv("MLFRAME_BATCH_MI_KERNEL", raising=False)
    monkeypatch.setattr(ktc.KernelTuningCache, "load_or_create", staticmethod(_boom))
    reset_throttle_counts()
    with caplog.at_level(logging.WARNING):
        got = select_batch_mi_kernel(1000, 10)
    assert got is batch_mi_with_noise_gate_v3
    assert any("kernel selection failed" in r.getMessage() for r in caplog.records)
