"""GPU resident caches for the conditional-MI kernels: LRU-bounded device uploads of y/z roles and of the factors matrix.

Split out of ``_cmi_cuda`` (its name-mutating state stays in place: these caches are only mutated, never rebound).
"""

from __future__ import annotations

import logging
import os as _os_thr
import threading
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


from collections import OrderedDict as _CmiResidentOrderedDict

_CMI_RESIDENT_CACHE: "_CmiResidentOrderedDict" = _CmiResidentOrderedDict()
_CMI_RESIDENT_CACHE_MAX_ENTRIES = 16
# This cache had no lock at all, unlike its two siblings in
# this same file (_FORDER_LOCK, _FACTORS_DEVICE_LOCK). Content-hash-guarded (see _resident_upload's own
# `sig` fingerprint), so an unlocked race here only cost a redundant upload (self-healing) rather than
# silent corruption - but the 2-of-3-locked inconsistency was a design smell worth closing.
_CMI_RESIDENT_CACHE_LOCK = threading.Lock()


def _resident_upload(arr, key):
    """Return a device int32 copy of host ``arr``, reusing a cached upload keyed on the caller's ``key``.

    The dispatch keys on ``(id(factors_data), column_index, nbins)`` - a STABLE identity across the greedy
    loop (the freshly-sliced host column is re-allocated every call, so keying on the slice's own id would
    never hit; the parent ``factors_data`` + column index is the fit-constant). ``id()`` can be recycled
    after factors_data is GC'd, so shape/dtype are folded into the cached entry and a recycled id with a
    different shape simply misses and re-uploads (never a stale wrong-shape buffer). Bounded; cleared at
    FE-step teardown.
    """
    import cupy as cp

    # Diagnostic A/B switch only (default ON): MLFRAME_CMI_RESIDENT_CACHE=0 forces a fresh upload every call
    # to reproduce the pre-FIX3 H2D churn for the wall A/B. NOT a perf gate - the cache is always on in prod.
    import os as _os
    if _os.environ.get("MLFRAME_CMI_RESIDENT_CACHE", "1").strip().lower() in ("0", "false", "off", "no"):
        return cp.asarray(arr)
    # CONTENT FINGERPRINT (recycled-id guard): id(factors_data) is reused by the allocator after the parent
    # is GC'd, so a recycled id with the SAME column index + shape + dtype but DIFFERENT values would return a
    # STALE device copy (a silent correctness bug - caught by the cmi parity tests). Fold a cheap O(n) host
    # content hash of the (small, 1-D, int) array into the key so different contents never alias. The hash is
    # pure CPU and far cheaper than the H2D it guards; on a real greedy loop the same y/z hash identically ->
    # the cache still hits every candidate batch.
    from mlframe.utils.disk_cache import hash_array_content

    # A 128-bit content digest, not the 64-bit builtin hash: a collision here returns a stale device copy silently.
    sig = (arr.shape, arr.dtype.str, hash_array_content(arr))
    with _CMI_RESIDENT_CACHE_LOCK:
        cached = _CMI_RESIDENT_CACHE.get(key)
        if cached is not None:
            g, csig = cached
            if csig == sig:
                _CMI_RESIDENT_CACHE.move_to_end(key)  # LRU: this content is hot
                return g
        g = cp.asarray(arr)
        _CMI_RESIDENT_CACHE[key] = (g, sig)
        if len(_CMI_RESIDENT_CACHE) > _CMI_RESIDENT_CACHE_MAX_ENTRIES:
            _CMI_RESIDENT_CACHE.popitem(last=False)  # evict ONLY the coldest entry, never the whole table
        return g


def clear_cmi_resident_cache() -> None:
    """Drop the resident y/z device cache (call at FE-step teardown; mirrors the mempool teardown)."""
    with _CMI_RESIDENT_CACHE_LOCK:
        _CMI_RESIDENT_CACHE.clear()


# --------------------------------------------------------------------------------------------------------------------
# XC RESIDENT MATRIX: ``factors_data`` (ALL candidate columns) is a fit-CONSTANT across the whole
# greedy CMI loop - same object for every round - yet the dispatch below re-sliced the round's candidate subset
# via ``factors_data[:, cand_indices].T`` on the HOST every round and handed the fresh host array to
# ``conditional_mi_batched_cuda``, which then re-uploaded it via ``cp.asarray`` - a host slice-copy PLUS an H2D
# transfer, repeated every round, of the single LARGEST operand any GPU kernel in this package touches (bigger
# than y/z by a factor of p, the audit's biggest-magnitude finding: tens of GB of redundant PCIe traffic over a
# 100k x 500 fit). y/z already got the resident treatment above (FIX3) for exactly this reason; Xc did not,
# because it is genuinely PER-CANDIDATE - but every round's candidates are always a SUBSET of the SAME
# fit-constant ``factors_data``, so the fix is to upload ``factors_data`` itself ONCE (keyed on its identity,
# mirroring ``_cmi_forder_view``'s weakref-identity guard a few dozen lines above) and GATHER the round's
# candidate columns ON-DEVICE (cupy fancy-indexing + transpose) - the host never sees the big matrix again
# after the first round. Selection-equivalent by construction: the gathered device values are byte-identical to
# the old host-sliced-then-uploaded ones (same source array, same column selection, same dtype cast).
# OrderedDict (not plain dict) for O(1) LRU: bounded at a handful of entries (each a FULL factors_data
# copy, potentially ~100s of MB of VRAM) so a long-lived process running many fits/CV-folds never grows
# this cache unboundedly - move-to-end on hit, evict only the single coldest entry on overflow (mirrors
# ``_fe_resident_operands.py``'s ``_FE_RESIDENT_OPERANDS`` LRU discipline for the exact same VRAM-safety
# reason). 4 is enough to keep 2-3 concurrently-alive folds/fits resident without ever approaching a 4 GB
# card's budget (at ~200 MB/entry that is <=800 MB worst case, comfortable alongside the other resident
# caches + working buffers documented in ``_fe_resident_operands.py``).
from collections import OrderedDict as _OrderedDict

_FACTORS_DEVICE_CACHE: "_OrderedDict" = _OrderedDict()
_FACTORS_DEVICE_MAX_ENTRIES = 4
_FACTORS_DEVICE_LOCK = threading.Lock()


def _xc_resident_enabled() -> bool:
    """Diagnostic A/B switch only (default ON): MLFRAME_CMI_XC_RESIDENT=0 forces a fresh host slice + upload of
    the candidate matrix every call, reproducing the pre-fix H2D/host-slice churn for the wall A/B."""
    return _os_thr.environ.get("MLFRAME_CMI_XC_RESIDENT", "1").strip().lower() not in ("0", "false", "off", "no")


def _host_codes_for_upload(factors_data: np.ndarray) -> np.ndarray:
    """The code matrix as it goes over PCIe: a signed integer matrix of at most 4 bytes per cell stays at its own width (the fit stores int8 / int16 codes), anything else is cast to int32.
    The widening to the kernels' int32 happens on the device (``_widen_on_device``), so the host never holds a 4-byte copy and the transfer is 1 to 2 bytes per cell."""
    arr = np.asarray(factors_data)
    if arr.dtype.kind == "i" and arr.dtype.itemsize <= 4:
        return np.ascontiguousarray(arr)
    return np.ascontiguousarray(arr, dtype=np.int32)


def _widen_on_device(cp: Any, host: np.ndarray) -> Any:
    """Upload ``host`` and return it as the int32 device matrix the kernels read."""
    dev = cp.asarray(host)
    return dev if dev.dtype == cp.int32 else dev.astype(cp.int32)


def _assert_codes_in_range_2d_per_column(host2d: np.ndarray, factors_nbins: np.ndarray) -> None:
    """One-time, WHOLE-matrix OOB guard: every column's codes must lie in ``[0, factors_nbins[col])``.

    Strictly a SUPERSET of the per-round candidate-subset check the old code ran every round (that check
    only validated the round's own ``cand_indices`` against the round's ``nbins_x = max(factors_nbins[cand])``
    upper bound); checking every column against its OWN nbins entry is at least as strict, so it can only catch
    an invalid code EARLIER, never later. Run ONCE per ``factors_data`` identity (see
    ``_resident_factors_device``) instead of every greedy round - cheaper in total than the old per-round
    re-scan (which repeats across dozens of rounds over overlapping candidate sets) while never weakening the
    safety net the OOB screen exists for (a code indexing outside the device histogram is a hard
    ``cudaErrorIllegalAddress`` crash, not a catchable Python error).
    """
    if host2d.size == 0:
        return
    bins = np.asarray(factors_nbins)
    col_min = host2d.min(axis=0)
    col_max = host2d.max(axis=0)
    if bool((col_min < 0).any()):
        j = int(np.argmax(col_min < 0))
        raise ValueError(f"factors_data column {j} contains a negative integer code (min={int(col_min[j])}); codes must be 0-based.")
    bad = col_max >= bins
    if bool(bad.any()):
        j = int(np.argmax(bad))
        raise ValueError(f"factors_data column {j} code out of range (max={int(col_max[j])} >= nbins={int(bins[j])}).")


def _resident_factors_device(factors_data: np.ndarray, factors_nbins: np.ndarray) -> Any:
    """Return the WHOLE ``factors_data`` matrix resident on-device as int32, uploaded ONCE per fit.

    Keyed on ``id(factors_data)`` with a weakref identity guard - mirrors ``_cmi_forder_view`` a few dozen
    lines above (the SAME object, already trusted with an id-only guard in this exact file): a recycled id
    (allocator reuse after the original ``factors_data`` is GC'd) fails the ``ref() is factors_data`` check and
    re-uploads, never returning a stale buffer. The one-time OOB guard (see
    ``_assert_codes_in_range_2d_per_column``) runs on every fresh upload (cache miss OR the disabled A/B path),
    so every device array this function returns has already been fully validated - callers may pass
    ``codes_trusted=True`` downstream.
    """
    import weakref

    import cupy as cp

    if not _xc_resident_enabled():
        host = _host_codes_for_upload(factors_data)
        _assert_codes_in_range_2d_per_column(host, factors_nbins)
        return _widen_on_device(cp, host)

    # Cache-hit fast path never touches the host array at all (no ascontiguousarray/dtype check on the
    # hot loop) - the identity key alone decides the hit, matching the "gather on-device, never re-touch
    # the host" goal this cache exists for.
    key = id(factors_data)
    with _FACTORS_DEVICE_LOCK:
        ent = _FACTORS_DEVICE_CACHE.get(key)
        if ent is not None and ent[0]() is factors_data:
            _FACTORS_DEVICE_CACHE.move_to_end(key)  # LRU: this factors_data is hot
            return ent[1]
        dead = [k for k, v in _FACTORS_DEVICE_CACHE.items() if v[0]() is None]
        for k in dead:
            del _FACTORS_DEVICE_CACHE[k]
        host = _host_codes_for_upload(factors_data)
        _assert_codes_in_range_2d_per_column(host, factors_nbins)
        dev = _widen_on_device(cp, host)
        _FACTORS_DEVICE_CACHE[key] = (weakref.ref(factors_data), dev)
        if len(_FACTORS_DEVICE_CACHE) > _FACTORS_DEVICE_MAX_ENTRIES:
            _FACTORS_DEVICE_CACHE.popitem(last=False)  # evict ONLY the coldest entry, never the whole table
        return dev


def clear_cmi_xc_resident_cache() -> None:
    """Drop the resident whole-factors_data device cache (call at FE-step teardown alongside the y/z cache)."""
    # Hold the same lock every mutate/iterate of this OrderedDict takes (_resident_factors_device's
    # dead-purge loop), else a teardown clear racing a still-fitting worker raises
    # "RuntimeError: dictionary changed size during iteration".
    with _FACTORS_DEVICE_LOCK:
        _FACTORS_DEVICE_CACHE.clear()
