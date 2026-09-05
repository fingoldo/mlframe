"""A transient device fault must not disable GPU metrics for the rest of the process.

The probe caught bare `Exception`, cached `False`, and said so only at debug. The exceptions it actually
sees are not facts about the machine -- a concurrent process holding the device, a CUDA OOM at probe time, a
WDDM TDR reset, a driver hiccup -- so the first one to land during the first metrics call decided the whole
run. This module's own header measures the CPU fallback at ~32s of a ~55s suite wall on a 1M-row binary run.

`ImportError` is the one genuine absence and is still cached, because cupy not being installed is a fact
about the install rather than a moment.
"""

from __future__ import annotations

import logging

import pytest

from mlframe.metrics import _gpu_metrics


@pytest.fixture(autouse=True)
def _clean_probe_cache():
    """Every test starts from an unprobed state and leaves one behind."""
    _gpu_metrics.reset_gpu_metrics_probe()
    yield
    _gpu_metrics.reset_gpu_metrics_probe()


def _force_probe_to(monkeypatch, exc: BaseException | None, *, available: bool = True):
    """Make the probe's `import cupy` raise `exc`, or succeed with a stub reporting one device."""

    import builtins

    real_import = builtins.__import__

    class _Runtime:
        """Stands in for `cupy.cuda.runtime`."""

        @staticmethod
        def getDeviceCount():
            """One visible device."""
            return 1

    class _Cuda:
        """Stands in for `cupy.cuda`."""

        runtime = _Runtime()

    class _Arr:
        """Stands in for the tiny array the probe reduces."""

        def sum(self):
            """Support the probe's NVRTC compile check."""
            return self

        def item(self):
            """Return a scalar so the probe's reduction succeeds."""
            return 1.0

    class _Cupy:
        """Stands in for the `cupy` module itself."""

        cuda = _Cuda()
        float32 = "float32"

        @staticmethod
        def asarray(*_a, **_k):
            """Return the stub array the probe reduces."""
            return _Arr()

    def _fake_import(name, *args, **kwargs):
        """Intercept only the probe's cupy import; everything else imports normally."""
        if name == "cupy":
            if exc is not None:
                raise exc
            return _Cupy()
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fake_import)


def test_a_transient_failure_is_not_cached(monkeypatch, caplog):
    """A RuntimeError during the probe must leave the cache unset, so a later call can succeed."""
    _force_probe_to(monkeypatch, RuntimeError("device busy"))
    with caplog.at_level(logging.WARNING, logger=_gpu_metrics.logger.name):
        assert _gpu_metrics.is_gpu_metrics_available() is False
    assert any("transiently" in r.message or "transiently" in r.getMessage() for r in caplog.records), "a transient probe failure was not reported above debug"
    assert _gpu_metrics._GPU_AVAILABLE is None, "a transient failure latched the process into the CPU path"

    # The device recovers; the very next call must see it.
    _force_probe_to(monkeypatch, None)
    assert _gpu_metrics.is_gpu_metrics_available() is True, "the probe never re-ran after a transient failure"


def test_an_absent_cupy_is_cached(monkeypatch):
    """ImportError is a fact about the install, so caching it for the process is correct."""
    _force_probe_to(monkeypatch, ImportError("no module named cupy"))
    assert _gpu_metrics.is_gpu_metrics_available() is False
    assert _gpu_metrics._GPU_AVAILABLE is False, "a genuine absence should be cached rather than re-probed"


def test_a_successful_probe_is_cached(monkeypatch):
    """The happy path must still memoise -- re-probing every call would be its own regression."""
    _force_probe_to(monkeypatch, None)
    assert _gpu_metrics.is_gpu_metrics_available() is True
    assert _gpu_metrics._GPU_AVAILABLE is True


def test_the_reset_entry_point_exists_and_clears_both_flags(monkeypatch):
    """Without a reset there is no way back from a cached decision, which is what made this class stick."""
    _force_probe_to(monkeypatch, None)
    _gpu_metrics.is_gpu_metrics_available()
    _gpu_metrics.reset_gpu_metrics_probe()
    assert _gpu_metrics._GPU_AVAILABLE is None
    assert _gpu_metrics._NUMBA_CUDA_AVAILABLE is None
