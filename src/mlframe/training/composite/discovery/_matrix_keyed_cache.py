"""A bounded, thread-safe LRU whose entries are tied to a feature matrix by weak reference.

The per-fold caches of the shared-fold fitters (the LightGBM dataset and the ridge factorisation) key an entry on the
matrix's ``id`` plus the fold rows. An ``id`` is reused once its matrix is freed, so each entry also holds a weak
reference to the matrix and hits only while that exact object is alive. One implementation, so the two caches cannot
drift in how they drop dead entries or bound themselves.
"""

from __future__ import annotations

import threading
import weakref
from collections import OrderedDict
from typing import Any, Hashable, Optional


class MatrixKeyedCache:
    """``key -> value`` for values computed from a matrix, valid while that matrix object is alive; LRU-bounded."""

    def __init__(self, max_entries: int) -> None:
        self.max_entries = int(max_entries)
        self._entries: "OrderedDict[Hashable, tuple[Any, Any]]" = OrderedDict()
        self._lock = threading.Lock()

    def __getstate__(self) -> dict:
        """Pickle as an empty cache of the same size: the entries hold weak references and the lock is live runtime state."""
        return {"max_entries": self.max_entries}

    def __setstate__(self, state: dict) -> None:
        self.__init__(state["max_entries"])

    def get(self, key: Hashable, matrix: Any) -> Optional[Any]:
        """The cached value for ``key`` when it was stored for this very ``matrix`` object, else None."""
        with self._lock:
            hit = self._entries.get(key)
            if hit is not None and hit[0]() is matrix:
                self._entries.move_to_end(key)
                return hit[1]
        return None

    def put(self, key: Hashable, matrix: Any, value: Any) -> None:
        """Store ``value`` for ``key`` against ``matrix``, dropping entries whose matrix is gone, then the oldest over the cap."""
        with self._lock:
            # An entry whose matrix is gone can never hit again (a new matrix at the same id fails the weakref check), so it
            # goes now rather than when the LRU reaches it: the rerank gathers per-base matrices on demand and drops them.
            self._drop_dead()
            self._entries[key] = (weakref.ref(matrix), value)
            while len(self._entries) > self.max_entries:
                # evict-ok: a miss recomputes the value from the same rows
                self._entries.popitem(last=False)

    def prune_dead(self) -> None:
        """Drop the entries whose matrix has been freed, so a finished phase leaves nothing of its folds resident."""
        with self._lock:
            self._drop_dead()

    def clear(self) -> None:
        """Drop every entry."""
        with self._lock:
            self._entries.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)

    def _drop_dead(self) -> None:
        """Remove the entries whose matrix has been freed; called with ``self._lock`` held."""
        for dead in [k for k, (ref, _) in self._entries.items() if ref() is None]:
            del self._entries[dead]
