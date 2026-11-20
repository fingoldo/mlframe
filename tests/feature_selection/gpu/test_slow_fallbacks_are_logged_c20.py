"""Gate fallbacks of the resident-select kernels, the CUDA probe and the group-distance device twin log at WARNING instead of staying silent or debug-only."""

from __future__ import annotations

import logging
import sys
import types

import numpy as np
import pytest

from mlframe.utils.log_throttle import reset_throttle_counts


@pytest.fixture(autouse=True)
def _fresh_throttle():
    """Forget the per-key log counts so every test sees its first warning."""
    reset_throttle_counts()
    yield
    reset_throttle_counts()


@pytest.fixture
def numpy_cupy(monkeypatch):
    """A cupy stand-in backed by numpy for the host-side shape and fallback code of the transpose helpers."""
    fake = types.ModuleType("cupy")
    fake.float32 = np.float32  # type: ignore[attr-defined]
    fake.float64 = np.float64  # type: ignore[attr-defined]
    fake.empty = np.empty  # type: ignore[attr-defined]
    fake.ascontiguousarray = np.ascontiguousarray  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "cupy", fake)
    return fake


def _boom(*_a, **_k):
    """Stand-in kernel getter that fails like a compile or launch error."""
    raise RuntimeError("kernel unavailable")


def _warned(caplog, needle: str) -> bool:
    """Whether a WARNING record mentions ``needle``."""
    return any(r.levelno == logging.WARNING and needle in r.getMessage() for r in caplog.records)


def test_f32_transpose_fallback_warns(monkeypatch, caplog, numpy_cupy):
    """A failing tiled transpose falls back to ``ascontiguousarray(.T)`` and says so at WARNING."""
    from mlframe.feature_selection.filters import _gpu_resident_select_kernels as k

    monkeypatch.setattr(k, "_get_transpose_f32_kernel", lambda: _boom)
    with caplog.at_level(logging.WARNING):
        out = k._transpose_to_cm(np.zeros((4, 3), dtype=np.float32))
    assert out.shape == (3, 4)
    assert _warned(caplog, "tiled transpose kernel failed")


def test_cm_to_rm_transpose_fallback_warns(monkeypatch, caplog, numpy_cupy):
    """The inverse transpose reports its fallback too."""
    from mlframe.feature_selection.filters import _gpu_resident_select_kernels as k

    monkeypatch.setattr(k, "_get_transpose_f32_kernel", lambda: _boom)
    with caplog.at_level(logging.WARNING):
        out = k._transpose_cm_to_rm(np.zeros((3, 4), dtype=np.float32))
    assert out.shape == (4, 3)
    assert _warned(caplog, "cm->rm transpose kernel failed")


def test_int_codes_transpose_fallback_warns(monkeypatch, caplog, numpy_cupy):
    """The int-codes transpose reports its fallback too."""
    from mlframe.feature_selection.filters import _gpu_resident_select_kernels as k

    monkeypatch.setattr(k, "_get_transpose_int_kernel", lambda itemsize: _boom)
    with caplog.at_level(logging.WARNING):
        out = k.transpose_codes_to_cm(np.zeros((4, 3), dtype=np.int8))
    assert out.shape == (3, 4)
    assert _warned(caplog, "int-codes transpose kernel failed")


def test_radix_threads_ktc_failure_warns(monkeypatch, caplog):
    """A failed per-host threads lookup uses the fixed default and says so."""
    from mlframe.feature_selection.filters import _gpu_resident_radix_ktc as ktc
    from mlframe.feature_selection.filters import _gpu_resident_select_kernels as k

    monkeypatch.setattr(k, "_RADIX_THREADS_OVERRIDE", None)
    monkeypatch.setattr(ktc, "radix_select_threads", _boom)
    with caplog.at_level(logging.WARNING):
        assert k._resolve_radix_threads(1000) == k._RADIX_SELECT_THREADS
    assert _warned(caplog, "threads KTC lookup failed")


def test_bsearch_compile_failure_warns(monkeypatch, caplog):
    """A bsearch compile failure falls back to the linear kernel and says so."""
    from mlframe.feature_selection.filters import _gpu_resident_select_kernels as k

    sentinel = object()
    monkeypatch.setattr(k, "_resolve_radix_f32_variant", lambda n: "bsearch")
    monkeypatch.setattr(k, "_get_radix_select_f32_bsearch_kernel", _boom)
    monkeypatch.setattr(k, "_get_radix_select_kernel", lambda is_f32: sentinel)
    with caplog.at_level(logging.WARNING):
        assert k._get_radix_select_f32_dispatch(1000) is sentinel
    assert _warned(caplog, "bsearch radix kernel compile failed")


def test_cuda_present_total_probe_failure_warns(monkeypatch, caplog):
    """When neither probe can answer, the host path is taken and the reason is logged."""
    import numba.cuda
    import pyutilz.core.pythonlib as pl

    from mlframe.feature_selection.filters import _gpu_policy
    from mlframe.feature_selection.filters import _gpu_resident_shared as shared

    monkeypatch.setattr(_gpu_policy, "gpu_globally_disabled", lambda: False)
    monkeypatch.setattr(pl, "is_cuda_available", _boom)
    monkeypatch.setattr(numba.cuda, "is_available", _boom)
    with caplog.at_level(logging.WARNING):
        assert shared._cuda_present() is False
    assert _warned(caplog, "reporting no CUDA")


def test_group_distance_choose_failure_warns(monkeypatch, caplog):
    """A failing kernel-tuning ``choose`` falls back to the row-count rule and says so."""
    from mlframe.feature_selection.filters import _gpu_policy
    from mlframe.feature_selection.filters import _group_distance_gpu as gd

    class _BadSpec:
        """Spec whose choose() fails."""

        def choose(self, **_k):
            """Raise unconditionally."""
            raise RuntimeError("lookup failed")

    monkeypatch.setattr(_gpu_policy, "gpu_globally_disabled", lambda: False)
    monkeypatch.setattr(gd, "_SPEC", _BadSpec())
    with caplog.at_level(logging.WARNING):
        assert gd.use_device(1000) is False
    assert _warned(caplog, "KTC choose() failed")


def test_group_distance_declined_block_warns(caplog):
    """A float64 block is handed back to the host loop with a warning naming the shape."""
    from mlframe.feature_selection.filters import _group_distance_gpu as gd

    xs = np.zeros((2, 10), dtype=np.float64)
    with caplog.at_level(logging.WARNING):
        got = gd.group_stats_device(np.zeros(10, dtype=np.int32), 2, xs, np.zeros((2, 10), dtype=np.int16), [np.zeros(0), np.zeros(0)], 20)
    assert got is None
    assert _warned(caplog, "device group statistics declined")


def test_group_distance_no_float32_columns_warns(monkeypatch, caplog):
    """The device was chosen but every paired column is float64: the host loop runs and the log says why."""
    from mlframe.feature_selection.filters import _group_distance_fe as fe
    from mlframe.feature_selection.filters import _group_distance_gpu as gd

    monkeypatch.setattr(gd, "use_device", lambda n: True)
    x = np.arange(10, dtype=np.float64)
    stats = {"v": (x, 0.0, 1.0, np.array([0.0, 1.0]), np.array([1.0]), np.sort(x), np.zeros(10, dtype=np.int64))}
    with caplog.at_level(logging.WARNING):
        got = fe._group_stats_on_device("g", {"g": (None, {"a": np.arange(10)})}, [("g", "v")], stats, 10)
    assert got is None
    assert _warned(caplog, "no paired numeric column is a float32")
