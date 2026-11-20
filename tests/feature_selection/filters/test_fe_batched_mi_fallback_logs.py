"""The batched-MI launch fallbacks and the recipe dtype probe log a WARNING (driven on CPU with a numpy-backed cupy stand-in and failing kernels)."""

from __future__ import annotations

import logging
import sys
import types

import numpy as np
import pytest

from mlframe.utils.log_throttle import reset_throttle_counts


@pytest.fixture(autouse=True)
def _fresh_throttles():
    """Forget the per-key log counts so each test sees its first occurrence."""
    reset_throttle_counts()
    yield
    reset_throttle_counts()


def _warned(caplog, text):
    """Whether a WARNING record contains ``text``."""
    return any(r.levelno >= logging.WARNING and text in r.getMessage() for r in caplog.records)


def _boom(*args, **kwargs):
    """Fail like a device or compile fault."""
    raise RuntimeError("boom")


def _noop_kernel(*args, **kwargs):
    """A kernel stand-in that does nothing."""
    return None


@pytest.fixture
def numpy_cupy(monkeypatch):
    """A cupy stand-in backed by numpy so the launch wrappers run on CPU."""
    cp = types.ModuleType("cupy")
    cp.__dict__.update({k: getattr(np, k) for k in dir(np) if not k.startswith("__")})
    cp.asnumpy = np.asarray  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "cupy", cp)
    from mlframe.feature_selection.filters import _fe_batched_mi

    monkeypatch.setattr(_fe_batched_mi, "block_size_for", lambda *a, **k: 32)
    return cp


def _codes():
    """A small code matrix and target for the from-codes launch."""
    rng = np.random.default_rng(0)
    return rng.integers(0, 3, (20, 2)).astype(np.int64), rng.integers(0, 2, 20).astype(np.int64)


def test_mi_from_codes_tile_failure_warns(monkeypatch, caplog, numpy_cupy):
    """A failing tiled launch falls back to the row-major kernel and warns."""
    from mlframe.feature_selection.filters import _fe_batched_mi as mod

    monkeypatch.setattr(mod, "_mi_from_codes_tile_width", lambda cells: 1)
    monkeypatch.setattr(mod, "_get_mi_from_codes_tile_kernel", lambda byte_codes=False: _boom)
    monkeypatch.setattr(mod, "_get_mi_from_codes_v2_kernels", lambda: None)
    monkeypatch.setattr(mod, "_get_mi_from_codes_kernel", lambda: _noop_kernel)
    C, y = _codes()
    with caplog.at_level(logging.WARNING):
        mod.binned_mi_from_codes_gpu(C, y, codes_trusted=True)
    assert _warned(caplog, "mi_from_codes tiled launch failed")


def test_mi_from_codes_v2_launch_failure_warns(monkeypatch, caplog, numpy_cupy):
    """A failing v2 launch falls back to the one-kernel path and warns."""
    from mlframe.feature_selection.filters import _fe_batched_mi as mod

    monkeypatch.setattr(mod, "_mi_from_codes_tile_width", lambda cells: 0)
    monkeypatch.setattr(mod, "_get_mi_from_codes_v2_kernels", lambda: _boom)
    monkeypatch.setattr(mod, "_get_mi_from_codes_kernel", lambda: _noop_kernel)
    C, y = _codes()
    with caplog.at_level(logging.WARNING):
        mod.binned_mi_from_codes_gpu(C, y, codes_trusted=True)
    assert _warned(caplog, "mi_from_codes v2 launch failed")


def test_mi_from_codes_v2_compile_failure_warns_and_retries(monkeypatch, caplog, numpy_cupy):
    """A failing v2 compile returns None, warns, and leaves the sentinel unset so the next call retries."""
    from mlframe.feature_selection.filters import _fe_batched_mi as mod

    numpy_cupy.RawKernel = _boom  # type: ignore[attr-defined]
    monkeypatch.setenv("MLFRAME_MI_FROM_CODES_V2", "1")
    monkeypatch.setattr(mod, "_MI_FROM_CODES_V2_KERNELS", None)
    with caplog.at_level(logging.WARNING):
        assert mod._get_mi_from_codes_v2_kernels() is None
    assert mod._MI_FROM_CODES_V2_KERNELS is None
    assert _warned(caplog, "mi_from_codes v2 RawKernel compile failed")


def _values_call(mod, monkeypatch):
    """Run the from-values launch with the split path chosen and the tuning lookup stubbed."""
    from mlframe.feature_selection.filters import _gpu_resident_select

    monkeypatch.setattr(_gpu_resident_select, "_transpose_to_cm", lambda a: np.ascontiguousarray(a.T))
    monkeypatch.setattr(mod, "_get_mi_from_values_kernel", lambda: _noop_kernel)
    rng = np.random.default_rng(0)
    x = rng.normal(size=(20, 2))
    edges = np.quantile(x, [0.5], axis=0)
    y = rng.integers(0, 2, 20).astype(np.int64)
    return mod.binned_mi_from_values_gpu(x, edges, y, 2, 2, codes_trusted=True)


def test_mi_split_n_failure_warns(monkeypatch, caplog, numpy_cupy):
    """A failing split-n launch falls back to the single-launch kernel and warns."""
    from mlframe.feature_selection.filters import _fe_batched_mi as mod
    from mlframe.feature_selection._benchmarks.kernel_tuning_cache import dispatch

    monkeypatch.setattr(dispatch, "lookup_fe_mi_split_backend", lambda n, K: "split")
    monkeypatch.setattr(mod, "_get_mi_split_kernels", _boom)
    with caplog.at_level(logging.WARNING):
        _values_call(mod, monkeypatch)
    assert _warned(caplog, "mi split-n launch failed")


def test_mi_split_lookup_failure_warns(monkeypatch, caplog, numpy_cupy):
    """A failing split-backend lookup uses the size-based fallback and warns."""
    from mlframe.feature_selection.filters import _fe_batched_mi as mod
    from mlframe.feature_selection._benchmarks.kernel_tuning_cache import dispatch

    monkeypatch.setattr(dispatch, "lookup_fe_mi_split_backend", _boom)
    with caplog.at_level(logging.WARNING):
        _values_call(mod, monkeypatch)
    assert _warned(caplog, "kernel_tuning_cache lookup for fe_mi split backend failed")


def test_qbin_radix_edges_failure_warns(monkeypatch, caplog):
    """A failing radix edge select falls back to the percentile edges and warns."""
    from mlframe.feature_selection.filters import _fe_batched_qbin as mod
    from mlframe.feature_selection.filters import _gpu_resident_select

    monkeypatch.setattr(_gpu_resident_select, "_radix_select_interior_edges", _boom)
    monkeypatch.setattr(mod, "device_percentile", lambda a, q, axis=0: np.percentile(a, q, axis=axis))
    X = np.random.default_rng(0).normal(size=(32, 3))
    with caplog.at_level(logging.WARNING):
        edges = mod._edges_of_block(np, X, 4, False, True)
    assert edges.shape == (5, 3)
    assert _warned(caplog, "_radix_select_interior_edges failed")


def test_recipe_vram_f32_check_failure_warns(monkeypatch, caplog):
    """A failing dtype-flag check keeps the f64 discipline and warns."""
    from mlframe.feature_selection.filters._fe_gpu_batch import _devices
    from mlframe.feature_selection.filters.engineered_recipes import _recipe_unary_binary_gpu as mod

    monkeypatch.setattr(_devices, "fe_gpu_f32_enabled", _boom)
    with caplog.at_level(logging.WARNING):
        assert mod._vram_f32() is False
    assert _warned(caplog, "fe_gpu_f32_enabled() check failed")
