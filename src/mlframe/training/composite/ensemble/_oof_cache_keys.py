"""Content digests and the bounded holdout cache for the composite OOF path.

The digests fingerprint arrays, holdouts, specs and weights into one cache key; the cache holds the
out-of-fold holdout predictions keyed by that key. ``ensemble/__init__`` re-exports every name.
"""
from __future__ import annotations

import threading
from collections import OrderedDict
from typing import Any

import numpy as np

_OOF_HOLDOUT_CACHE: "OrderedDict[tuple, tuple[np.ndarray, np.ndarray, list[str]]]" = OrderedDict()
_OOF_HOLDOUT_CACHE_CAP = 16
# External callers are free to invoke this cache from multiple threads; the lock protects OrderedDict's
# own move_to_end/popitem bookkeeping from concurrent corruption (an unguarded LRU touch racing an
# eviction can leave the dict in an inconsistent order or double-evict). It does NOT make the caller's
# get-compute-put sequence atomic -- two threads racing the same miss both compute (deterministically,
# so idempotent) and the second put simply overwrites the first with an identical value.
_OOF_HOLDOUT_CACHE_LOCK = threading.Lock()


def _oof_cache_get(key: tuple):
    """LRU-touch and return the cached OOF result for ``key``, or ``None`` on a miss."""
    with _OOF_HOLDOUT_CACHE_LOCK:
        if key not in _OOF_HOLDOUT_CACHE:
            return None
        _OOF_HOLDOUT_CACHE.move_to_end(key)
        return _OOF_HOLDOUT_CACHE[key]


def _oof_cache_put(key: tuple, value: tuple) -> None:
    """Store ``value`` under ``key`` in the LRU OOF cache, evicting the oldest entry once over capacity."""
    with _OOF_HOLDOUT_CACHE_LOCK:
        if key in _OOF_HOLDOUT_CACHE:
            _OOF_HOLDOUT_CACHE.move_to_end(key)
            _OOF_HOLDOUT_CACHE[key] = value
            return
        if len(_OOF_HOLDOUT_CACHE) >= _OOF_HOLDOUT_CACHE_CAP:
            # evict-ok: memo; a miss recomputes the value
            _OOF_HOLDOUT_CACHE.popitem(last=False)
        _OOF_HOLDOUT_CACHE[key] = value


def _full_array_digest(a: np.ndarray) -> int:
    """Content digest of a whole array (string / object arrays through their repr), for cache keys that must tell two different arrays apart."""
    import hashlib

    arr = np.ascontiguousarray(a) if a.dtype != object else np.asarray(a.astype(str))
    h = hashlib.blake2b(digest_size=8)
    if arr.dtype.kind == "U":
        # Length-prefix every element: a bare join made ["ab", "c"] and ["a", "bc"] collide (audit 35-F8).
        for item in arr.ravel().tolist():
            encoded = item.encode("utf-8", "surrogatepass")
            h.update(len(encoded).to_bytes(8, "little"))
            h.update(encoded)
    else:
        h.update(arr.tobytes())
    return int.from_bytes(h.digest(), "little")


def _holdout_x_digest(X: Any) -> int:
    """Content digest of the external holdout frame (0 when absent), so equal-length holdouts get distinct keys (audit 35-F9)."""
    if X is None:
        return 0
    if hasattr(X, "hash_rows"):  # polars
        rows = np.asarray(X.hash_rows().to_numpy())
    else:  # pandas or array-like
        import pandas as pd
        rows = pd.util.hash_pandas_object(pd.DataFrame(X), index=False).to_numpy()
    return _full_array_digest(rows) ^ _full_array_digest(np.asarray(np.shape(X), dtype=np.int64))


def _feed_digest(h, value) -> None:
    """Feed ``value`` into the hash ``h``; arrays are digested in full, and dicts, lists and tuples are walked so an array nested
    inside them is not elided by ``repr`` (audit 35-F21)."""
    if isinstance(value, np.ndarray):
        h.update(b"ndarray")
        h.update(repr(_full_array_digest(value)).encode())
    elif isinstance(value, dict):
        h.update(b"dict")
        for k in sorted(value, key=str):
            h.update(repr(k).encode())
            _feed_digest(h, value[k])
    elif isinstance(value, (list, tuple)):
        h.update(b"seq")
        for item in value:
            _feed_digest(h, item)
    else:
        h.update(repr(value).encode())


def _specs_digest(component_specs) -> bytes:
    """Content digest of the component specs."""
    import hashlib

    h = hashlib.blake2b(digest_size=16)
    for spec in component_specs or []:
        _feed_digest(h, spec)
    return h.digest()


def _oof_full_cache_key(cache_key, kfold, random_state, holdout_frac, time_ordering, external_holdout_X, external_holdout_y, group_ids,
                        sample_weight=None, component_specs=None) -> tuple:
    """The OOF cache key: every argument that changes the returned matrix, so two calls differing only in holdout_frac /
    time-mode / external-holdout identity / group_ids / sample weights / component specs cannot serve each other a stale entry."""
    if group_ids is not None:
        g = np.asarray(group_ids)
        group_fp = (g.shape[0], _full_array_digest(g))
    else:
        group_fp = (0, 0)
    # the time order decides which rows each fold sees, so its content (not just its presence) is part of the key
    time_fp = _full_array_digest(np.asarray(time_ordering)) if time_ordering is not None else 0
    return (
        cache_key, int(kfold), int(random_state), round(float(holdout_frac), 6), time_fp,
        _holdout_x_digest(external_holdout_X), (_full_array_digest(np.asarray(external_holdout_y)) if external_holdout_y is not None else -1), group_fp,
        (_full_array_digest(np.asarray(sample_weight)) if sample_weight is not None else -1),
        _specs_digest(component_specs),
    )
