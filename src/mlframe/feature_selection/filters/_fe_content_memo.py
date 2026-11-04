"""Content-keyed memo decorator for the FE scorers a fit calls more than once with the same inputs.

The families of one fit share raw columns, targets and parameters, so the same scorer is sometimes asked the same question twice (the CMIM scorer: one call from the scorer dispatcher and one from the
subsample wrapper). The key hashes every array / frame argument by content and the other arguments by value; a result is returned as a deep copy so a caller that edits it cannot corrupt the stored
one. The store is bounded by entries and bytes and is dropped under memory pressure through the cache-release registry.
"""

from __future__ import annotations

import functools
from ._content_hasher import new_content_hasher
from typing import Any, Callable

import numpy as np
import pandas as pd

from ._cache_release import register_cache_releaser

_STORES: "list[dict]" = []


def _digest(obj: Any, h: Any, depth: int = 0) -> None:
    """Feed ``obj`` into the hash by content (arrays, frames, series, short containers) or by repr."""
    if isinstance(obj, np.ndarray):
        h.update(str(obj.shape).encode() + str(obj.dtype).encode())
        if obj.dtype == object or obj.dtype.kind not in "biufcmM":
            # no byte view of an object / string array (the view raised and the whole call ran unmemoised): hash the values
            h.update(np.ascontiguousarray(pd.util.hash_array(obj)).view(np.uint8).reshape(-1).data)
        else:
            h.update(np.ascontiguousarray(obj).view(np.uint8).reshape(-1).data)
    elif isinstance(obj, pd.DataFrame):
        h.update(("DF" + str(obj.shape) + "|".join(map(str, obj.columns))).encode())
        for c in obj.columns:
            _digest(obj[c].to_numpy(), h, depth + 1)
    elif isinstance(obj, pd.Series):
        _digest(obj.to_numpy(), h, depth + 1)
    elif isinstance(obj, (list, tuple)) and depth < 3 and len(obj) < 500:
        for x in obj:
            _digest(x, h, depth + 1)
    elif isinstance(obj, dict) and depth < 3 and len(obj) < 100:
        for k in sorted(obj, key=str):
            h.update(str(k).encode())
            _digest(obj[k], h, depth + 1)
    else:
        h.update(repr(obj).encode()[:200])


def content_memo(max_entries: int = 4, max_bytes: int = 256 * 1024 * 1024) -> Callable:
    """Decorator: memoise a function returning a DataFrame on the content of its arguments (bounded, copy-on-return)."""

    def deco(fn: Callable) -> Callable:
        """Wrap ``fn`` with its own store."""
        store: dict = {}
        _STORES.append(store)

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            """Return the stored copy for equal inputs, else call ``fn`` and store its DataFrame result."""
            try:
                h = new_content_hasher()
                _digest(args, h)
                _digest(kwargs, h)
                key = h.hexdigest()
            except Exception:  # best-effort: an unhashable input skips the memo; fn computes the same result
                return fn(*args, **kwargs)
            hit = store.get(key)
            if hit is not None:
                return hit.copy(deep=True)
            result = fn(*args, **kwargs)
            if isinstance(result, pd.DataFrame):
                size = int(result.memory_usage(index=True).sum())
                if size <= max_bytes:
                    while store and (len(store) >= max_entries or sum(int(d.memory_usage(index=True).sum()) for d in store.values()) + size > max_bytes):
                        store.pop(next(iter(store)))
                    store[key] = result.copy(deep=True)
            return result

        return wrapper

    return deco


def _release_content_memos() -> int:
    """Drop every memoised result (registered for release under memory pressure); returns the bytes freed."""
    freed = 0
    for store in _STORES:
        freed += sum(int(d.memory_usage(index=True).sum()) for d in store.values())
        store.clear()
    return freed


register_cache_releaser(_release_content_memos)
