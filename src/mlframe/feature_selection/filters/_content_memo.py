"""A small bounded memo of pure results keyed by the content of their array inputs.

The FE step is run again with relaxed thresholds when its first sweep finds nothing, and the retry asks for the same expensive tables over the same pool: the usability |corr| of every pair, the
batched pair MI, the order-2 permutation floor. Each of those is a pure function of its arrays and a few scalars, so the second request can be answered from the first result. The key is the shape,
dtype and content hash (``_fe_resident_operands._content_hash``: xxh3 over the buffer, no copy) of every array plus the scalars that change the result; the newest entries are kept and an entry is
evicted oldest first. A stored value is handed out as a copy so a caller cannot change what a later request receives.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from typing import Any

import numpy as np


def copy_value(value: Any) -> Any:
    """A copy of ``value`` that shares no array or dict with it: arrays are copied, tuples, lists and dicts are copied element-wise, anything else is returned as is."""
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, tuple):
        return tuple(copy_value(v) for v in value)
    if isinstance(value, list):
        return [copy_value(v) for v in value]
    if isinstance(value, dict):
        return {k: copy_value(v) for k, v in value.items()}
    return value


class ContentMemo:
    """Bounded, thread-safe memo keyed by array content; ``get`` and ``put`` copy the value."""

    def __init__(self, max_entries: int):
        self.max_entries = int(max_entries)
        self._entries: "OrderedDict[Any, Any]" = OrderedDict()
        self._lock = threading.Lock()

    def key(self, arrays: "list[np.ndarray]", *extras: Any) -> tuple:
        """The memo key of ``arrays`` (shape, dtype and content hash of each, C-contiguous) and the hashable ``extras`` that also decide the result."""
        from ._fe_resident_operands import _content_hash

        parts = []
        for array in arrays:
            a = np.ascontiguousarray(array)
            parts.append((a.shape, a.dtype.str, _content_hash(a)))
        return (tuple(parts), *extras)

    def get(self, key: tuple) -> Any:
        """A copy of the stored value for ``key``, or ``None`` when there is none."""
        with self._lock:
            hit = self._entries.get(key)
            if hit is None:
                return None
            self._entries.move_to_end(key)
        return copy_value(hit)

    def put(self, key: tuple, value: Any) -> None:
        """Store a copy of ``value`` under ``key`` and evict the oldest entries beyond ``max_entries``."""
        stored = copy_value(value)
        with self._lock:
            self._entries[key] = stored
            self._entries.move_to_end(key)
            while len(self._entries) > self.max_entries:
                self._entries.popitem(last=False)

    def clear(self) -> None:
        """Drop every stored value."""
        with self._lock:
            self._entries.clear()

    def __len__(self) -> int:
        """Number of stored values."""
        return len(self._entries)
