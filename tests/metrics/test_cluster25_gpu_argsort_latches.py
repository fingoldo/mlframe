"""Audit 2026-11-20 finding 25-F41: after one GPU argsort failure the metric kernels stop trying the GPU for the process.

The first failure falls back to the exact CPU sort and latches the GPU path off, so later calls neither retry the device nor
repeat the failure. The test fakes a cupy module whose asarray raises, and counts the attempts.
"""

import sys
import types

import numpy as np

from mlframe.metrics import _core_auc_brier as core


def test_one_failure_latches_gpu_argsort_off(monkeypatch):
    """Two calls: the device is tried once, both results equal the CPU sort, and the flag is latched False."""
    attempts = []

    def failing_device_argsort(values):
        """Simulate a device fault on every GPU sort and count the attempt."""
        attempts.append(1)
        raise RuntimeError("device fault")

    from mlframe.feature_selection.filters import shared as safe_sort

    monkeypatch.setattr(safe_sort, "device_argsort", failing_device_argsort)
    fake_cupy = types.ModuleType("cupy")
    fake_cupy.asarray = lambda values: values
    fake_cupy.asnumpy = lambda values: values
    monkeypatch.setitem(sys.modules, "cupy", fake_cupy)
    monkeypatch.setattr(core, "_GPU_ARGSORT_MIN_N", 1)
    monkeypatch.setattr(core, "_GPU_ARGSORT_AVAILABLE", True)
    monkeypatch.delenv("MLFRAME_METRICS_STABLE_SORT", raising=False)

    scores = np.array([0.2, 0.9, 0.5, 0.1])
    expected = np.argsort(scores)[::-1]
    first = core._argsort_desc_for_metrics(scores)
    second = core._argsort_desc_for_metrics(scores)
    assert attempts == [1]
    assert core._GPU_ARGSORT_AVAILABLE is False
    assert np.array_equal(first, expected) and np.array_equal(second, expected)
