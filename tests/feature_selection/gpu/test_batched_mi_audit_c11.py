"""Audit regressions of the batched device MI family: length guards, 8-byte aligned edge cache, per-device tables, WARNING-level fallbacks."""

from __future__ import annotations

import logging
import types

import numpy as np
import pytest

cp = pytest.importorskip("cupy")
if cp.cuda.runtime.getDeviceCount() < 1:
    pytest.skip("no CUDA device", allow_module_level=True)

from mlframe.feature_selection.filters import _fe_batched_mi as M
from mlframe.feature_selection.filters import _fe_batched_mi_cmi as C
from mlframe.feature_selection.filters import _fe_batched_mi_cmi_lut as L
from mlframe.feature_selection.filters import _fe_batched_mi_mm_values as V
from mlframe.feature_selection.filters import _fe_batched_mi_mm_cm as CM


def _raiser(msg):
    """A callable that raises ``RuntimeError(msg)``."""

    def f(*a, **k):
        """Raise."""
        raise RuntimeError(msg)

    return f


def test_mm_values_rejects_short_y():
    """A y with fewer rows than X would be read out of bounds by the kernel."""
    X = cp.asarray(np.random.default_rng(0).standard_normal((100, 3)))
    E = cp.asarray(np.sort(np.random.default_rng(1).standard_normal((9, 3)), axis=0))
    with pytest.raises(ValueError, match="y_codes has 90 rows"):
        V.binned_mm_mi_from_values_gpu(X, E, cp.zeros(90, dtype=cp.int64), 10, 2, 0.69, 2)


def test_mm_values_rejects_edges_of_other_width():
    """Edges with a different column count than X are rejected before the launch."""
    X = cp.asarray(np.random.default_rng(0).standard_normal((100, 3)))
    E = cp.asarray(np.sort(np.random.default_rng(1).standard_normal((9, 2)), axis=0))
    with pytest.raises(ValueError, match="interior_edges shape"):
        V.binned_mm_mi_from_values_gpu(X, E, cp.zeros(100, dtype=cp.int64), 10, 2, 0.69, 2)


def test_mm_cm_rejects_short_y():
    """The column-major twin guards y length the same way."""
    X = cp.asarray(np.random.default_rng(0).standard_normal((3, 100)))
    E = cp.asarray(np.sort(np.random.default_rng(1).standard_normal((9, 3)), axis=0))
    with pytest.raises(ValueError, match="y_codes has 90 rows"):
        CM.binned_mm_mi_from_cm_gpu(X, E, cp.zeros(90, dtype=cp.int64), 10, 2, 0.69, 2)


def test_plugin_mi_from_values_rejects_short_y_and_edges():
    """The plug-in values kernel guards y length and the edge table shape."""
    X = cp.asarray(np.random.default_rng(0).standard_normal((100, 3)))
    E = cp.asarray(np.sort(np.random.default_rng(1).standard_normal((9, 3)), axis=0))
    with pytest.raises(ValueError, match="y_codes has 90 rows"):
        M.binned_mi_from_values_gpu(X, E, cp.zeros(90, dtype=cp.int64), 10, 2)
    with pytest.raises(ValueError, match="interior_edges shape"):
        M.binned_mi_from_values_gpu(X, E[:5], cp.zeros(100, dtype=cp.int64), 10, 2)


def test_plugin_mi_from_codes_rejects_short_y():
    """The plug-in codes kernel guards y length."""
    codes = cp.asarray(np.random.default_rng(0).integers(0, 4, (100, 3)))
    with pytest.raises(ValueError, match="y_codes has 90 rows"):
        M.binned_mi_from_codes_gpu(codes, cp.zeros(90, dtype=cp.int64), ky=2)


def test_batched_cmi_rejects_short_y_and_z():
    """The batched CMI guards the y and z lengths against the candidate rows."""
    X = cp.asarray(np.random.default_rng(0).integers(0, 4, (100, 3)))
    with pytest.raises(ValueError, match="y has 90 rows"):
        M._batched_cmi_gpu_once(X, cp.zeros(90, dtype=cp.int64))
    with pytest.raises(ValueError, match="z has 90 rows"):
        M._batched_cmi_gpu_once(X, cp.zeros(100, dtype=cp.int64), cp.zeros(90, dtype=cp.int64))


def test_mm_nek_tile_with_odd_histogram_cells_launches_aligned():
    """With one tile column and an odd histogram cell count the edge cache sat at a 4-byte offset: the tile launch must succeed."""
    rng = np.random.default_rng(3)
    n, K, nbins, ky = 4000, 3, 3, 1667
    cells = nbins * ky
    assert cells % 2 == 1 and 44000 // (4 * cells + 256 * 8) == 1
    X = cp.asarray(rng.standard_normal((n, K)))
    E = cp.asarray(np.sort(rng.standard_normal((nbins - 1, K)), axis=0))
    y = cp.asarray(rng.integers(0, ky, n))
    cmin, cmax = V._column_min_max(X, n, K, False)
    Ec = cp.empty((nbins, K), dtype=cp.float64)
    ne_k = cp.empty(K, dtype=cp.int32)
    V._launch_dedup_edges(E, cmin, cmax, nbins - 1, K, Ec, ne_k)
    out = cp.empty(K, dtype=cp.float64)
    assert V._launch_mm_nek_tile(X, Ec, y, n, K, nbins, ky, 1.0, ky, ne_k, out, False) is True
    cp.cuda.Device().synchronize()
    assert bool(cp.isfinite(out).all())


def test_column_min_max_failure_is_logged_at_warning(monkeypatch, caplog):
    """A failing min/max kernel falls back to cupy reductions with a WARNING."""
    monkeypatch.setattr(V, "kernel_minmax", _raiser("nvrtc broke"))
    monkeypatch.setattr(V, "_COL_MINMAX_KERNELS", {})
    X = cp.asarray(np.random.default_rng(0).standard_normal((64, 3)))
    with caplog.at_level(logging.WARNING, logger=V.logger.name):
        lo, _hi = V._column_min_max(X, 64, 3, False)
    assert np.allclose(cp.asnumpy(lo), cp.asnumpy(X).min(axis=0))
    assert any(r.levelno == logging.WARNING and "nvrtc broke" in r.getMessage() for r in caplog.records)


def test_cmi_assemble_failure_is_logged_at_warning(monkeypatch, caplog):
    """A failing assemble kernel falls back to the cupy chain with a WARNING."""
    monkeypatch.setattr(C, "kernel_assemble", _raiser("asm broke"))
    monkeypatch.setattr(C, "_CMI_ASSEMBLE_KERNEL", None)
    he1, he2 = cp.asarray([1.0, 2.0]), cp.asarray([0.5, 0.5])
    kc1, kc2 = cp.asarray([3, 3], dtype=cp.int64), cp.asarray([1, 1], dtype=cp.int64)
    with caplog.at_level(logging.WARNING, logger=C.__name__):
        out = C._cmi_assemble(he1, he2, 0.0, kc1, kc2, 0.0, 0.01)
    assert np.allclose(cp.asnumpy(out), [0.5 - 0.02, 1.5 - 0.02])
    assert any(r.levelno == logging.WARNING and "asm broke" in r.getMessage() for r in caplog.records)


def test_marginal_entropies_failure_is_logged_at_warning(monkeypatch, caplog):
    """A failing fused three-joint kernel returns None (caller falls back) with a WARNING."""
    monkeypatch.setattr(C, "kernel_marginal", _raiser("marg broke"))
    monkeypatch.setattr(C, "_MARGINAL_MI_ENTROPIES_KERNEL", None)
    x = cp.zeros(10, dtype=cp.int64)
    with caplog.at_level(logging.WARNING, logger=C.__name__):
        assert C.marginal_mi_entropies_gpu(x, x, 2, 2, 0.1) is None
    assert any(r.levelno == logging.WARNING and "marg broke" in r.getMessage() for r in caplog.records)


def test_xlogx_table_is_keyed_per_device(monkeypatch):
    """A table built on one device must not be served to another device."""
    state = {"dev": 0}
    made: list = []

    class _Dev:
        """Fake current device."""

        @property
        def id(self):
            """Current device id."""
            return state["dev"]

    def fake_asarray(host):
        """Record the build and return a tagged object."""
        obj = (state["dev"], host.size)
        made.append(obj)
        return obj

    fake = types.SimpleNamespace(cuda=types.SimpleNamespace(Device=_Dev), asarray=fake_asarray)
    monkeypatch.setattr(L, "_XLOGX_TABLES", {})
    t0 = L._xlogx_table(fake, 50)
    state["dev"] = 1
    t1 = L._xlogx_table(fake, 50)
    assert t0 != t1 and len(made) == 2
    assert L._xlogx_table(fake, 50) is t1
