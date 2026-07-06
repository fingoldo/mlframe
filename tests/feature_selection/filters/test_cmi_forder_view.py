"""2026-07-06: F-order (column-contiguous) view cache for the CPU CMI melt.

_cpu_cmi_loop is the dominant redundancy cost (1950s in the 1M .prof). factors_data is
(n, nfeat) C-contiguous, so each candidate column is read with an nfeat-element stride.
_cmi_forder_view caches a column-contiguous copy of the fit-constant matrix once per fit;
every round then reads columns contiguously (2-6x, bit-identical). These tests pin: the
result is bit-identical to the C-order path, the cache reuses one copy per fit, the weakref
identity guard prevents a recycled id from returning a stale copy, and the size/toggle gates.
"""
from __future__ import annotations

import gc

import numpy as np
import pytest


@pytest.fixture(autouse=True)
def _clear_cache():
    import mlframe.feature_selection.filters.info_theory._cmi_cuda as cm

    cm.reset_cmi_forder_cache()
    yield
    cm.reset_cmi_forder_cache()


def _matrix(n=4000, p=60, nb=12, seed=0):
    rng = np.random.default_rng(seed)
    nc = p + 2
    fd = np.empty((n, nc), dtype=np.int32, order="C")
    for c in range(nc):
        fd[:, c] = rng.integers(0, nb, n)
    fnb = np.full(nc, nb, dtype=np.int64)
    cand = np.arange(p, dtype=np.int64)
    y = np.array([p], dtype=np.int64)
    z = np.array([p + 1], dtype=np.int64)
    return fd, cand, y, z, fnb


def test_cmi_bit_identical_c_vs_f_order(monkeypatch):
    """_cpu_cmi_loop must return byte-identical CMI with the F-order view on vs off."""
    import mlframe.feature_selection.filters.info_theory._cmi_cuda as cm

    fd, cand, y, z, fnb = _matrix()

    monkeypatch.setenv("MLFRAME_CMI_FORDER", "0")
    cm.reset_cmi_forder_cache()
    ref = cm._cpu_cmi_loop(fd, cand, y, z, fnb)

    monkeypatch.setenv("MLFRAME_CMI_FORDER", "1")
    cm.reset_cmi_forder_cache()
    got = cm._cpu_cmi_loop(fd, cand, y, z, fnb)

    assert np.array_equal(ref, got), np.max(np.abs(ref - got))


def test_view_is_f_contiguous_and_equal():
    import mlframe.feature_selection.filters.info_theory._cmi_cuda as cm

    fd, *_ = _matrix()
    assert fd.flags.c_contiguous and not fd.flags.f_contiguous
    v = cm._cmi_forder_view(fd)
    assert v.flags.f_contiguous
    assert np.array_equal(v, fd)


def test_cache_reuses_one_copy_per_fit():
    """Repeated calls on the same array return the SAME cached object (one transpose per fit)."""
    import mlframe.feature_selection.filters.info_theory._cmi_cuda as cm

    fd, *_ = _matrix()
    v1 = cm._cmi_forder_view(fd)
    v2 = cm._cmi_forder_view(fd)
    assert v1 is v2
    assert len(cm._FORDER_CACHE) == 1


def test_weakref_guard_no_stale_copy_on_recycled_id():
    """A fresh array must never receive another array's cached F-copy. The weakref ``is`` check
    rebuilds when the cached weakref no longer points at the queried object."""
    import mlframe.feature_selection.filters.info_theory._cmi_cuda as cm

    fd1, *_ = _matrix(seed=1)
    v1 = cm._cmi_forder_view(fd1)
    key1 = id(fd1)
    del fd1
    gc.collect()

    # Force allocations to churn ids; whatever we now build must get its OWN correct F-copy.
    fd2 = _matrix(seed=2)[0]
    v2 = cm._cmi_forder_view(fd2)
    assert np.array_equal(v2, fd2)
    # If fd2 happened to reuse fd1's id, the stale entry must NOT have been returned.
    if id(fd2) == key1:
        assert v2 is not v1
    assert not np.shares_memory(v2, v1) or np.array_equal(v2, fd2)


def test_disabled_returns_input_unchanged(monkeypatch):
    import mlframe.feature_selection.filters.info_theory._cmi_cuda as cm

    fd, *_ = _matrix()
    monkeypatch.setenv("MLFRAME_CMI_FORDER", "0")
    assert cm._cmi_forder_view(fd) is fd
    assert len(cm._FORDER_CACHE) == 0


def test_oversize_cap_returns_input_unchanged(monkeypatch):
    """When the copy would exceed the byte cap, return the C-order input (no surprise alloc)."""
    import mlframe.feature_selection.filters.info_theory._cmi_cuda as cm

    fd, *_ = _matrix()
    monkeypatch.setattr(cm, "_CMI_FORDER_MAX_MB", fd.nbytes / (1 << 20) / 2.0)
    assert cm._cmi_forder_view(fd) is fd


def test_f_contiguous_input_returned_asis():
    import mlframe.feature_selection.filters.info_theory._cmi_cuda as cm

    fd, *_ = _matrix()
    ffd = np.asfortranarray(fd)
    assert cm._cmi_forder_view(ffd) is ffd
