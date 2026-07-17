"""XC RESIDENT MATRIX cache (2026-07-12): ``factors_data`` uploaded ONCE per fit, candidate columns
GATHERED on-device instead of re-sliced-and-re-uploaded from the host every greedy round.

Mirrors ``tests/feature_selection/filters/test_cmi_forder_view.py``'s template (the sibling cache for
the SAME ``factors_data`` object in this same module): one-copy-per-fit reuse, the weakref identity
guard against a recycled id, the disabled-A/B toggle, and bit-identical dispatch output. Adds a test
for the NEW one-time whole-matrix OOB guard (``_assert_codes_in_range_2d_per_column``) that replaces
the old per-round Xc-subset check.
"""

from __future__ import annotations

import gc

import numpy as np
import pytest

from mlframe.feature_selection.filters.info_theory._cmi_cuda import (
    conditional_mi_batched_dispatch,
    cupy_available,
)
from mlframe.feature_selection.filters.info_theory._entropy_kernels import conditional_mi

_HAS_GPU = cupy_available()
_gpu_only = pytest.mark.skipif(not _HAS_GPU, reason="no CUDA/cupy GPU available")


@pytest.fixture(autouse=True)
def _clear_cache():
    import mlframe.feature_selection.filters.info_theory._cmi_cuda as cm

    cm.clear_cmi_xc_resident_cache()
    cm.clear_cmi_resident_cache()
    yield
    cm.clear_cmi_xc_resident_cache()
    cm.clear_cmi_resident_cache()


def _matrix(n=4000, p=60, nb=12, seed=0):
    rng = np.random.default_rng(seed)
    nc = p + 2
    fd = np.empty((n, nc), dtype=np.int32, order="C")
    for c in range(nc):
        fd[:, c] = rng.integers(0, nb, n)
    fnb = np.full(nc, nb, dtype=np.int64)
    cand = np.arange(p, dtype=np.int64)
    y_index, z_index = p, p + 1
    return fd, cand, y_index, z_index, fnb


def _cpu_ref(fd, fnb, p, y_index, z_index):
    return np.array([conditional_mi(fd, np.array([c]), np.array([y_index]), np.array([z_index]), None, fnb) for c in range(p)])


@_gpu_only
def test_cache_reuses_one_device_copy_per_fit():
    """Repeated dispatch calls (different candidate subsets -- successive greedy rounds) on the SAME
    factors_data object must upload the device-resident matrix exactly ONCE, not once per round."""
    import mlframe.feature_selection.filters.info_theory._cmi_cuda as cm

    fd, cand, y_index, z_index, fnb = _matrix()

    upload_calls = {"n": 0}
    import cupy as cp

    orig_asarray = cp.asarray

    def _counting_asarray(arr, *a, **kw):
        if getattr(arr, "shape", None) == fd.shape:
            upload_calls["n"] += 1
        return orig_asarray(arr, *a, **kw)

    cp.asarray = _counting_asarray
    try:
        # Round 1: candidates 0..29. Round 2: candidates 30..59 (DISJOINT subset, same factors_data).
        conditional_mi_batched_dispatch(fd, cand[:30], y_index, z_index, fnb, force="cuda")
        conditional_mi_batched_dispatch(fd, cand[30:], y_index, z_index, fnb, force="cuda")
        conditional_mi_batched_dispatch(fd, cand, y_index, z_index, fnb, force="cuda")
    finally:
        cp.asarray = orig_asarray

    assert upload_calls["n"] == 1, f"factors_data-shaped cp.asarray called {upload_calls['n']} times across 3 rounds (expected 1)"
    assert len(cm._FACTORS_DEVICE_CACHE) == 1


@_gpu_only
def test_dispatch_bit_identical_resident_on_vs_off():
    """XC-resident gather ON (default) must be bit-identical to the OLD host-slice-then-upload path."""
    import mlframe.feature_selection.filters.info_theory._cmi_cuda as cm

    fd, cand, y_index, z_index, fnb = _matrix(seed=3)

    import os

    os.environ["MLFRAME_CMI_XC_RESIDENT"] = "0"
    cm.clear_cmi_xc_resident_cache()
    try:
        off = conditional_mi_batched_dispatch(fd, cand, y_index, z_index, fnb, force="cuda")
    finally:
        os.environ["MLFRAME_CMI_XC_RESIDENT"] = "1"
        cm.clear_cmi_xc_resident_cache()
    on = conditional_mi_batched_dispatch(fd, cand, y_index, z_index, fnb, force="cuda")

    assert np.array_equal(on, off), f"maxdiff={np.max(np.abs(on - off))}"
    cpu = _cpu_ref(fd, fnb, len(cand), y_index, z_index)
    assert np.abs(on - cpu).max() < 1e-9


@_gpu_only
def test_weakref_guard_no_stale_gather_on_recycled_id():
    """A fresh factors_data must never be gathered from another array's cached device copy."""
    import mlframe.feature_selection.filters.info_theory._cmi_cuda as cm

    fd1, _cand1, _y1, _z1, fnb1 = _matrix(seed=1)
    dev1 = cm._resident_factors_device(fd1, fnb1)
    key1 = id(fd1)
    del fd1
    gc.collect()

    fd2, _cand2, _y2, _z2, fnb2 = _matrix(seed=2, nb=9)
    dev2 = cm._resident_factors_device(fd2, fnb2)
    assert np.array_equal(np.asarray(dev2.get()), fd2)
    if id(fd2) == key1:
        assert dev2 is not dev1


@_gpu_only
def test_oob_guard_fires_once_for_whole_matrix():
    """A code exceeding its OWN column's factors_nbins entry must still raise, even though the
    per-round Xc-subset check was replaced by the one-time whole-matrix guard."""
    import mlframe.feature_selection.filters.info_theory._cmi_cuda as cm

    fd, _cand, _y_index, _z_index, fnb = _matrix(seed=4, nb=5)
    fnb_bad = fnb.copy()
    fnb_bad[2] = 2  # column 2's actual codes go up to 4, but we claim only 2 bins -> OOB
    with pytest.raises(ValueError, match="out of range"):
        cm._resident_factors_device(fd, fnb_bad)


@_gpu_only
def test_disabled_ab_switch_skips_cache():
    """MLFRAME_CMI_XC_RESIDENT=0 must upload fresh every call (A/B baseline reproduction), never
    populating the resident cache."""
    import os

    import mlframe.feature_selection.filters.info_theory._cmi_cuda as cm

    fd, _cand, _y_index, _z_index, fnb = _matrix(seed=5)
    os.environ["MLFRAME_CMI_XC_RESIDENT"] = "0"
    try:
        cm._resident_factors_device(fd, fnb)
        assert len(cm._FACTORS_DEVICE_CACHE) == 0
    finally:
        os.environ["MLFRAME_CMI_XC_RESIDENT"] = "1"


@_gpu_only
def test_lru_bound_evicts_coldest_entry():
    """More than _FACTORS_DEVICE_MAX_ENTRIES distinct factors_data objects must evict the coldest,
    never grow the cache unboundedly (VRAM safety)."""
    import mlframe.feature_selection.filters.info_theory._cmi_cuda as cm

    mats = [_matrix(seed=100 + i, n=500, p=10)[0] for i in range(cm._FACTORS_DEVICE_MAX_ENTRIES + 2)]
    fnb = _matrix(seed=100, n=500, p=10)[4]
    for m in mats:
        cm._resident_factors_device(m, fnb)
    assert len(cm._FACTORS_DEVICE_CACHE) <= cm._FACTORS_DEVICE_MAX_ENTRIES


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "--tb=short"])
