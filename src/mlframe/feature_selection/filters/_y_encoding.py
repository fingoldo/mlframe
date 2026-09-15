"""Shared target-encoding helper for the classification-MI FE families.

The FE families that score an engineered column's plug-in classification MI against ``y`` must first turn
``y`` into dense integer class codes. The trap the several families independently fell into was casting a
CONTINUOUS float target with ``astype(int64)`` BEFORE ``np.unique`` - which truncates ``0.7 -> 0`` and
collapses the target to a couple of buckets (MI destroyed), OR densifying a genuinely continuous target into
~n singleton classes (classification MI degenerate, every row its own class). Both destroy the signal.

``encode_y_for_classif_mi`` is the single correct path: integer targets densify directly; a continuous
(float/complex) target with more than a handful of distinct values is quantile-binned FIRST (never
int-truncated), then densified. Mirrors the temporal-agg family's continuous-y guard.
"""
from __future__ import annotations

import hashlib
import logging
import threading
from collections import OrderedDict
from typing import Callable

import numpy as np

from mlframe.utils.log_throttle import log_throttle

logger = logging.getLogger(__name__)

# Above this many distinct float values a target is treated as continuous and quantile-binned rather than
# densified as-is (a float target with <=32 levels is already effectively discrete -> densify directly).
_CONTINUOUS_Y_DISTINCT_THRESHOLD = 32
_CONTINUOUS_Y_QCUT_BINS = 10


# Float targets at or above this length go through the content-keyed memo; below it the sort is cheaper than the bookkeeping.
_ENCODE_MEMO_MIN_N = 4096
# One fit discretises one target, so a couple of entries cover it; each entry holds the codes at their narrowest integer dtype.
_ENCODE_MEMO_MAX_ENTRIES = 2
_encode_memo: "OrderedDict[tuple, np.ndarray]" = OrderedDict()
_encode_memo_lock = threading.Lock()


def _resolve_content_hash() -> Callable[[memoryview], int]:
    """``xxhash.xxh3_64_intdigest`` when usable, else a blake2b digest; both hash the raw buffer."""
    try:
        import xxhash as _xxhash

        return _xxhash.xxh3_64_intdigest
    except Exception as e:
        logger.debug("encode_y memo: xxhash unavailable (%s), hashing targets with blake2b", e)

        def _blake(buf: memoryview) -> int:
            """64-bit blake2b digest of ``buf`` as an int."""
            return int.from_bytes(hashlib.blake2b(buf, digest_size=8).digest(), "little")

        return _blake


_content_hash = _resolve_content_hash()


def _clear_encode_cache() -> None:
    """Drop every memoised target encoding."""
    with _encode_memo_lock:
        _encode_memo.clear()


def encode_y_for_classif_mi(y: np.ndarray) -> np.ndarray:
    """Dense int64 class codes for a classification-MI target.

    Integer/bool ``y`` is densified directly. A continuous (float/complex) ``y`` with more than
    ``_CONTINUOUS_Y_DISTINCT_THRESHOLD`` distinct values is quantile-binned (never int-truncated, which would
    collapse ``[0, 1)`` to one class) then densified. Idempotent on already-dense integer codes.

    The FE cascade encodes the same fit target in many stages, so a float target's codes are memoised on a hash of its contents: an equal
    target is served without re-sorting, a target whose values changed is re-encoded, and every call returns an array the caller owns.
    """
    arr = np.asarray(y).ravel()
    if arr.dtype.kind not in "fc" or arr.size < _ENCODE_MEMO_MIN_N:
        return _encode_y_uncached(arr)
    contig = np.ascontiguousarray(arr)
    key = (contig.dtype.str, contig.shape, _content_hash(contig.view(np.uint8).data))
    with _encode_memo_lock:
        hit = _encode_memo.get(key)
        if hit is not None:
            _encode_memo.move_to_end(key)
    if hit is not None:
        return hit.astype(np.int64)
    codes = _encode_y_uncached(contig)
    stored = codes.astype(np.min_scalar_type(int(codes.max())) if codes.size else np.int8)
    with _encode_memo_lock:
        _encode_memo[key] = stored
        while len(_encode_memo) > _ENCODE_MEMO_MAX_ENTRIES:
            _encode_memo.popitem(last=False)
    return codes


def _encode_y_uncached(y: np.ndarray) -> np.ndarray:
    """The encoding itself, without the memo; see ``encode_y_for_classif_mi``."""
    arr = np.asarray(y).ravel()
    if np.issubdtype(arr.dtype, np.integer) or arr.dtype == bool:
        a = arr.astype(np.int64, copy=False)
        # Fast path for an already-dense 0..K-1 code vector (the overwhelmingly common case: this helper is
        # called per-pair x per-modulus x per-permutation inside the FE scans). np.unique would re-SORT the
        # whole column every call; an O(n) bincount occupancy check is far cheaper and returns the identical
        # array when the codes are already dense.
        if a.size:
            mn = int(a.min())
            mx = int(a.max())
            if mn == 0 and mx < 65536 and np.count_nonzero(np.bincount(a, minlength=mx + 1)) == mx + 1:
                return a
        _, inv = np.unique(a, return_inverse=True)
        return inv.astype(np.int64, copy=False)
    if arr.dtype.kind in "fc" and int(np.unique(arr).size) > _CONTINUOUS_Y_DISTINCT_THRESHOLD:
        try:
            import pandas as pd

            # qcut(labels=False) returns an ndarray for ndarray input (Series for Series) - np.asarray covers both.
            arr = np.asarray(pd.qcut(arr, q=_CONTINUOUS_Y_QCUT_BINS, labels=False, duplicates="drop"))
        except Exception as e:
            # The plain-densify fallback turns a genuinely continuous target into roughly n singleton classes, which the
            # module docstring above names as signal-destroying for classification MI. It is kept so the fit still runs,
            # but it must not be silent. Throttled: this helper runs per pair x per modulus x per permutation in the scans.
            log_throttle(
                logger, "encode_y_qcut_failed", logging.WARNING,
                "encode_y_for_classif_mi: pd.qcut binning of a continuous target failed (%s: %s); falling back to plain "
                "densify, which leaves ~one class per distinct value and degrades classification MI.",
                type(e).__name__, e,
            )
    _, inv = np.unique(arr, return_inverse=True)
    return inv.astype(np.int64, copy=False)
