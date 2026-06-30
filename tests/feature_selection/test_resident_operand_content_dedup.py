"""Regression: the FE resident-operand cache must dedup by CONTENT across roles (one upload per content).

``resident_operand(arr, role)`` caches the fit-constant operand's device copy. The label ``y`` and support
``z`` are uploaded under many different role labels (``cmi_y`` / ``card_y`` / ``fixedyz_y`` / ``y_mi_classif``
/ ...; ``cmi_z`` / ``card_z`` / ``fixedyz_z``). The original ``(role, shape, dtype)`` key uploaded the SAME
content once PER ROLE -- an H2D audit of a 1M strict-resident fit measured 615 MB / 90 ops (65% of operand
H2D) of such cross-role re-uploads. The cache now keys PURELY on the content signature
(shape + dtype + ``hash(tobytes())``), so identical content shares one resident device buffer regardless of
role. This pins that contract (identity of the returned device array) plus the correctness guards: different
VALUES never alias, different dtype stays distinct, and LRU eviction keeps the hot entry resident.
"""
from __future__ import annotations

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

import mlframe.feature_selection.filters._fe_resident_operands as RO


def _need_cuda() -> bool:
    try:
        from pyutilz.core.pythonlib import is_cuda_available
        return is_cuda_available()
    except Exception:
        try:
            return cp.cuda.runtime.getDeviceCount() > 0
        except Exception:
            return False


pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not _need_cuda(), reason="no CUDA")]


def _fresh_cache(monkeypatch):
    from collections import OrderedDict
    monkeypatch.setattr(RO, "_FE_RESIDENT_OPERANDS", OrderedDict(), raising=False)
    # ensure the diagnostic A/B disable switch is OFF for this test
    monkeypatch.delenv("MLFRAME_FE_RESIDENT_OPERANDS", raising=False)


def test_same_content_different_roles_share_one_device_buffer(monkeypatch):
    _fresh_cache(monkeypatch)
    y = np.arange(5000, dtype=np.int64) % 7        # the label content, as int64 codes

    g_cmi = RO.resident_operand(y, "cmi_y", dtype=np.int64)
    g_card = RO.resident_operand(y, "card_y", dtype=np.int64)        # different role, SAME content
    g_fixed = RO.resident_operand(y.copy(), "fixedyz_y", dtype=np.int64)  # fresh host object, SAME content

    # Cross-role identical content must return the SAME device array (one upload, shared) -- not three buffers.
    assert g_cmi is g_card, "same content under a different role must reuse the resident buffer (no re-upload)"
    assert g_cmi is g_fixed, "a fresh host object with identical content must hit the content-keyed cache"
    assert len(RO._FE_RESIDENT_OPERANDS) == 1, "three same-content roles must occupy ONE cache entry"
    assert bool((g_cmi == cp.asarray(y)).all())


def test_different_values_never_alias(monkeypatch):
    _fresh_cache(monkeypatch)
    a = np.arange(5000, dtype=np.int64) % 7
    b = a.copy(); b[123] += 1                       # one differing element -> different content
    ga = RO.resident_operand(a, "cmi_y", dtype=np.int64)
    gb = RO.resident_operand(b, "cmi_y", dtype=np.int64)
    assert ga is not gb, "different VALUES must miss (fresh upload), never alias a stale buffer"
    assert int(gb[123].get()) == int(b[123])


def test_dtype_variants_stay_distinct(monkeypatch):
    _fresh_cache(monkeypatch)
    v = np.arange(5000) % 7
    gi = RO.resident_operand(v, "role", dtype=np.int64)
    gf = RO.resident_operand(v, "role", dtype=cp.float64)   # same values, different FINAL dtype -> distinct
    assert gi is not gf
    assert gi.dtype == cp.int64 and gf.dtype == cp.float64


def test_lru_evicts_coldest_not_whole_table(monkeypatch):
    _fresh_cache(monkeypatch)
    monkeypatch.setattr(RO, "_MAX_ENTRIES", 3, raising=False)
    cols = [np.full(2000, i, dtype=np.int64) for i in range(4)]   # 4 distinct contents, bound 3
    g0 = RO.resident_operand(cols[0], "op", dtype=np.int64)
    RO.resident_operand(cols[1], "op", dtype=np.int64)
    RO.resident_operand(cols[2], "op", dtype=np.int64)
    RO.resident_operand(cols[0], "op", dtype=np.int64)           # touch col0 -> now hot (most recent)
    RO.resident_operand(cols[3], "op", dtype=np.int64)           # overflow -> evict the COLDEST (col1), not col0
    assert len(RO._FE_RESIDENT_OPERANDS) == 3, "LRU keeps the bound, not a clear-all (which would drop to 1)"
    # col0 was refreshed before the overflow -> it must still be resident (same buffer), proving single-evict LRU.
    assert RO.resident_operand(cols[0], "op", dtype=np.int64) is g0
