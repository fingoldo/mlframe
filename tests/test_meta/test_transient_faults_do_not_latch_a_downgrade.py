"""A transient fault must not cost the whole process its fast path, silently.

This library has a documented history of exactly one bug: a broad `except Exception` around a startup probe
treats an unexpected, momentary failure as evidence about the machine, caches that verdict for the process
lifetime, and logs it at `debug`. The original instance pinned every mutual-information computation to a ~100x
slower sklearn path after one transient device fault at import.

The 2026-09-01 audit found two more live instances of the same shape, in two different subsystems. Both are
pinned here, together, because the shape is what recurs -- not the module.

The rule they share: `ImportError` means the dependency is genuinely absent, so latching is correct and silence
is fine. Anything else is unexpected, must be audible at `warning`, and must not permanently disable the fast
path on the strength of one occurrence.
"""

from __future__ import annotations

import importlib
import logging
import sys

import pytest


class TestTheKernelTuningCacheRetries:
    """`_kernel_tuning` gates 268 dispatch sites; latching it off costs every one of them."""

    @pytest.fixture(autouse=True)
    def _clean_singleton(self):
        """Reset the module singleton and the attempt counter around each test."""
        from mlframe.feature_selection.filters import _kernel_tuning as kt

        kt._reset_for_tests()
        yield kt
        kt._reset_for_tests()

    def test_an_unexpected_failure_is_retried_not_latched(self, _clean_singleton, monkeypatch):
        """One corrupt read, or one file lock from a concurrent process, must not end the fast path."""
        kt = _clean_singleton
        calls = {"n": 0}

        class _Flaky:
            """Raises once, then constructs normally -- the transient case."""

            def __init__(self):
                calls["n"] += 1
                if calls["n"] == 1:
                    raise OSError("tuning file locked by another process")

        monkeypatch.setitem(sys.modules, "pyutilz.performance.kernel_tuning.cache", type(sys)("m"))
        sys.modules["pyutilz.performance.kernel_tuning.cache"].KernelTuningCache = _Flaky

        assert kt.get_kernel_tuning_cache() is None, "the failing attempt must not return a half-built cache"
        assert kt.get_kernel_tuning_cache() is not None, "the next lookup must retry rather than stay latched off"
        assert calls["n"] == 2

    def test_the_failure_is_audible(self, _clean_singleton, monkeypatch, caplog):
        """`logger.debug` on a path that silently changes performance is invisible in production."""
        kt = _clean_singleton

        class _Broken:
            """Always raises."""

            def __init__(self):
                raise OSError("tuning file corrupt")

        monkeypatch.setitem(sys.modules, "pyutilz.performance.kernel_tuning.cache", type(sys)("m"))
        sys.modules["pyutilz.performance.kernel_tuning.cache"].KernelTuningCache = _Broken

        with caplog.at_level(logging.WARNING, logger=kt.__name__):
            kt.get_kernel_tuning_cache()
        assert any("KernelTuningCache init failed" in r.getMessage() for r in caplog.records), "the downgrade was not logged at warning"

    def test_it_gives_up_after_a_bounded_number_of_attempts(self, _clean_singleton, monkeypatch):
        """Retrying forever would pay the failing construction on every one of the 268 lookups."""
        kt = _clean_singleton
        calls = {"n": 0}

        class _Broken:
            """Always raises, counting attempts."""

            def __init__(self):
                calls["n"] += 1
                raise OSError("still broken")

        monkeypatch.setitem(sys.modules, "pyutilz.performance.kernel_tuning.cache", type(sys)("m"))
        sys.modules["pyutilz.performance.kernel_tuning.cache"].KernelTuningCache = _Broken

        for _ in range(10):
            kt.get_kernel_tuning_cache()
        assert calls["n"] == kt._MAX_INIT_ATTEMPTS, f"construction was attempted {calls['n']} times"

    def test_a_genuine_absence_latches_immediately(self, _clean_singleton, monkeypatch):
        """`ImportError` IS evidence about the machine, so the latch is correct there and stays."""
        kt = _clean_singleton
        monkeypatch.setitem(sys.modules, "pyutilz.performance.kernel_tuning.cache", None)
        assert kt.get_kernel_tuning_cache() is None
        assert kt._CACHE_SINGLETON is False, "a genuinely absent dependency should not be re-probed"


class TestTheCudaProbeStaysOptimistic:
    """`_gpu_probe` runs at import and its verdict is cached for the process."""

    def test_a_transient_probe_failure_does_not_disable_gpu(self, monkeypatch):
        """A driver reset or a contended device is not evidence that no GPU exists."""
        import numba.cuda

        monkeypatch.setattr(numba.cuda, "is_available", lambda: (_ for _ in ()).throw(OSError("device busy")))
        sys.modules.pop("mlframe.training._gpu_probe", None)
        module = importlib.import_module("mlframe.training._gpu_probe")
        try:
            assert module.CUDA_IS_AVAILABLE is True, "a transient probe failure disabled GPU for the whole process"
        finally:
            sys.modules.pop("mlframe.training._gpu_probe", None)
            importlib.import_module("mlframe.training._gpu_probe")

    def test_a_transient_probe_failure_is_audible(self, monkeypatch, caplog):
        """The one signal an operator has that the run is not deciding GPU support the usual way."""
        import numba.cuda

        monkeypatch.setattr(numba.cuda, "is_available", lambda: (_ for _ in ()).throw(OSError("device busy")))
        sys.modules.pop("mlframe.training._gpu_probe", None)
        with caplog.at_level(logging.WARNING, logger="mlframe.training._gpu_probe"):
            importlib.import_module("mlframe.training._gpu_probe")
        try:
            assert any("transient device/driver condition" in r.getMessage() for r in caplog.records)
        finally:
            sys.modules.pop("mlframe.training._gpu_probe", None)
            importlib.import_module("mlframe.training._gpu_probe")
