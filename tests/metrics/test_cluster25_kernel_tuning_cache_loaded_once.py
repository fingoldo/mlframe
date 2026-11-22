"""Audit 2026-11-20 finding 25-F36: the kernel tuning cache is loaded once per process, not on every RMSE call.

The test counts load_or_create calls across two loads of the helper and checks the second one reuses the first object.
"""

import pytest

pytest.importorskip("pyutilz.performance.kernel_tuning.cache")

from mlframe.metrics import _gpu_metrics as gm


def test_tuning_cache_is_loaded_once(monkeypatch):
    """Two helper calls return the same object and load it a single time."""
    from pyutilz.performance.kernel_tuning import cache as cache_module

    loads = []

    class _CountingCache:
        """A stand-in cache whose only job is to count how often it is loaded."""

        @staticmethod
        def load_or_create():
            """Record one load and return a fresh stand-in."""
            loads.append(1)
            return object()

    monkeypatch.setattr(cache_module, "KernelTuningCache", _CountingCache)
    monkeypatch.setattr(gm, "_KERNEL_TUNING_CACHE", None)
    first = gm._load_kernel_tuning_cache()
    second = gm._load_kernel_tuning_cache()
    assert first is second
    assert len(loads) == 1
