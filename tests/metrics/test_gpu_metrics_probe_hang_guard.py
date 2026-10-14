"""Regression: ``is_gpu_metrics_available`` must not block forever on a hung CUDA driver probe.

CI on a GPU-less runner hit ``cp.cuda.runtime.getDeviceCount()`` HANGING (not raising) inside this
probe -- ``except Exception`` never triggers for a hang. It runs from a session-scoped conftest
fixture (``_prewarm_numba_once`` -> ``prewarm_numba_cache`` -> here), so the hang consumed the whole
per-test pytest-timeout budget (300s) on the first test, then cascaded a "Timeout" failure onto every
later test in that worker for the rest of the session (the session fixture's cached failure is
reported, unretried). Fixed by running the actual probe on a bounded-join daemon thread.
"""

from __future__ import annotations

import threading

import pytest

from mlframe.metrics import _gpu_metrics


@pytest.mark.hang_guard
def test_hung_probe_returns_false_within_timeout_bound(monkeypatch):
    """A probe thread that never returns must not block the caller past ``_GPU_PROBE_TIMEOUT_S``."""
    monkeypatch.setattr(_gpu_metrics, "_GPU_AVAILABLE", None)
    monkeypatch.setattr(_gpu_metrics, "_GPU_PROBE_TIMEOUT_S", 0.3)

    release = threading.Event()
    probe_finished = threading.Event()

    class _HungCupy:
        """Stub ``cupy`` module whose device-count call blocks until the test releases it."""

        class cuda:
            """Stub ``cupy.cuda`` namespace."""

            class runtime:
                """Stub ``cupy.cuda.runtime`` namespace with a blocking probe."""

                @staticmethod
                def getDeviceCount():
                    """Block past the test's timeout bound to simulate a hung CUDA driver call."""
                    release.wait(timeout=5.0)
                    probe_finished.set()
                    return 1

    def _fake_import(name, *args, **kwargs):
        """Return the hung stub for ``import cupy``, delegate everything else to the real import."""
        if name == "cupy":
            return _HungCupy()
        return _real_import(name, *args, **kwargs)

    import builtins

    _real_import = builtins.__import__
    monkeypatch.setattr(builtins, "__import__", _fake_import)

    result = _gpu_metrics.is_gpu_metrics_available()
    # The probe is still blocked when the call returns: the caller did not wait for the hung driver call to finish.
    returned_before_probe_finished = not probe_finished.is_set()

    release.set()  # let the leaked daemon thread finish so it doesn't outlive the test
    assert result is False, "a hung probe must resolve to unavailable, not raise or return True"
    assert returned_before_probe_finished, "is_gpu_metrics_available() must not wait for the hung probe to finish"


def test_result_cached_after_hang_so_later_calls_do_not_reprobe(monkeypatch):
    """After one bounded timeout, the cached False must make every subsequent call a no-op (no re-probe)."""
    import builtins

    monkeypatch.setattr(_gpu_metrics, "_GPU_AVAILABLE", False)
    real_import = builtins.__import__
    cupy_imports = []

    def _counting_import(name, *args, **kwargs):
        """Record every ``import cupy`` attempt and delegate to the real import."""
        if name == "cupy":
            cupy_imports.append(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _counting_import)
    for _ in range(1000):
        assert _gpu_metrics.is_gpu_metrics_available() is False
    assert cupy_imports == [], "cached-False calls must not re-probe the driver"
