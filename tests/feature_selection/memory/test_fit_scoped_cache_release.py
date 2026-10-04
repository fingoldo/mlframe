"""Fit-end release of process-global caches: GPU operand tables, F-order CMI copy, memmap dumps."""
from __future__ import annotations

import os

import numpy as np

from mlframe.feature_selection.filters import _gpu_resident_materialise as gm
from mlframe.feature_selection.filters import _joblib_safe as js
from mlframe.feature_selection.filters._fit_scoped_release import release_fit_scoped_caches
from mlframe.feature_selection.filters.info_theory import _cmi_cuda as cc


class _FakeDev:
    """Stand-in for a device array exposing only ``nbytes`` and ``shape``."""

    def __init__(self, nbytes: int) -> None:
        """Record the byte size."""
        self.nbytes = nbytes
        self.shape = (1, 1)


def test_release_clears_operand_tables_forder_and_memmap():
    """One call at fit end empties all three caches and unlinks the memmap backing file."""
    host = np.zeros((4, 3), dtype=np.float32)
    gm._OPERAND_TABLE_CACHE.clear()
    import weakref

    gm._OPERAND_TABLE_CACHE[id(host)] = (weakref.ref(host), _FakeDev(16))
    gm.register_prebuilt_operand_table(host, _FakeDev(16))
    gm._OPERAND_TABLE_CM_CACHE["ref"] = weakref.ref(host)
    gm._OPERAND_TABLE_CM_CACHE["cm"] = _FakeDev(16)
    fac = np.arange(60, dtype=np.int32).reshape(20, 3)
    cc.reset_cmi_forder_cache()
    cc._cmi_forder_view(fac)
    assert len(cc._FORDER_CACHE) == 1
    mm = js.fit_constant_memmap(np.arange(1000, dtype=np.float64))
    path = mm.filename
    assert os.path.exists(path) and len(js._FIT_MEMMAP_CACHE) >= 1

    release_fit_scoped_caches()

    assert not gm._OPERAND_TABLE_CACHE and not gm._PREBUILT_OPERAND_TABLE
    assert gm._OPERAND_TABLE_CM_CACHE["cm"] is None
    assert not cc._FORDER_CACHE
    assert not js._FIT_MEMMAP_CACHE
    assert not os.path.exists(path)


def test_operand_cache_trim_drops_dead_hosts_and_respects_byte_budget(monkeypatch):
    """Dead-host entries are purged on insert and the byte budget evicts oldest-first, keeping the newest."""
    import weakref
    from collections import OrderedDict

    monkeypatch.setenv("MLFRAME_FE_OPERAND_CACHE_MAX_MB", "1")
    cache: OrderedDict = OrderedDict()
    dead = np.zeros(2)
    cache[1] = (weakref.ref(dead), _FakeDev(10))
    del dead
    hosts = [np.zeros(2) for _ in range(3)]
    for i, h in enumerate(hosts):
        cache[100 + i] = (weakref.ref(h), _FakeDev(600 * 1024))
    gm._trim_operand_cache(cache, 8)
    assert 1 not in cache
    assert list(cache) == [102]


def test_release_with_nothing_imported_is_noop():
    """Calling the release twice is idempotent."""
    release_fit_scoped_caches()
    release_fit_scoped_caches()
    assert not js._FIT_MEMMAP_CACHE


def test_mrmr_exit_scope_releases_only_when_last_fit_leaves():
    """The last in-flight fit leaving triggers the release; an earlier exit leaves caches intact."""
    from mlframe.feature_selection.filters.mrmr import _mrmr_class_fit_helpers as fh

    mm = js.fit_constant_memmap(np.arange(500, dtype=np.float64))
    assert mm is not None
    obj = fh._MRMRFitHelpersMixin()
    fh._ACTIVE_FIT_COUNT = 2
    obj._exit_active_fit_scope()
    assert js._FIT_MEMMAP_CACHE
    obj._exit_active_fit_scope()
    assert not js._FIT_MEMMAP_CACHE
