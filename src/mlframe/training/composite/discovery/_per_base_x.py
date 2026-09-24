"""Per-base ``(base_screen, X without the base)`` for the rerank, gathered on demand from one screen matrix.

The rerank used to build a ``np.delete`` copy of the (screen rows x features) matrix for every distinct base up front and
hold them all to the end, so its peak grew with the base count (16 bases x 60k x 199 x 4 B was 760 MB of a 958 MB rerank
peak). The copies are now built when a base is first asked for and at most ``capacity`` stay cached; a caller still holding
an evicted matrix keeps it alive until it is done, so memory follows the bases in flight. A synthetic interaction base
drops its two parents, as the screen does.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from collections.abc import Mapping
from typing import Any, Callable, Iterator, Sequence

import numpy as np

from .._synthetic_bases import dropped_columns


class PerBaseMatrices(Mapping):
    """``base -> (base_screen, x_without_base)``; the base vectors are held, the matrices gathered and LRU-bounded."""

    def __init__(self, x_full: np.ndarray, columns: Sequence[str], base_screens: dict, *, capacity: int = 2):
        self._x_full = x_full
        self._col_index = {c: i for i, c in enumerate(columns)}
        self._base_screens = dict(base_screens)
        self._capacity = max(1, int(capacity))
        self._built: OrderedDict = OrderedDict()
        self._lock = threading.Lock()

    def drop_idx(self, base: str) -> list:
        """Column indices of ``x_full`` that ``base``'s matrix leaves out."""
        return [self._col_index[c] for c in dropped_columns(base, self._col_index)]

    def __getitem__(self, base: str) -> tuple:
        base_screen = self._base_screens[base]
        drop = self.drop_idx(base)
        if not drop:
            return base_screen, self._x_full
        with self._lock:
            x = self._built.get(base)
            if x is None:
                x = np.delete(self._x_full, drop, axis=1)
                self._built[base] = x
                while len(self._built) > self._capacity:
                    self._built.popitem(last=False)
            else:
                self._built.move_to_end(base)
        return base_screen, x

    def __contains__(self, base: object) -> bool:
        return base in self._base_screens

    def __iter__(self) -> Iterator[str]:
        return iter(self._base_screens)

    def __len__(self) -> int:
        return len(self._base_screens)

    def base_screen(self, base: str) -> Any:
        """The base's screen values without gathering its matrix."""
        return self._base_screens.get(base)


class BoundedMemo:
    """A thread-safe ``key -> value`` memo keeping the ``capacity`` most recent values; ``get_or_build(key, build)``."""

    def __init__(self, capacity: int = 2):
        self._capacity = max(1, int(capacity))
        self._values: OrderedDict = OrderedDict()
        self._lock = threading.Lock()

    def get_or_build(self, key: Any, build: Callable[[], Any]) -> Any:
        with self._lock:
            hit = self._values.get(key)
            if hit is None:
                hit = self._values[key] = build()
                while len(self._values) > self._capacity:
                    self._values.popitem(last=False)
            else:
                self._values.move_to_end(key)
        return hit


def base_ordered(indices: Sequence[int], base_of: Callable[[int], str]) -> list:
    """``indices`` grouped by base (first-appearance order, stable within a base), so a bounded cache builds each once."""
    first: dict = {}
    for i in indices:
        first.setdefault(base_of(i), len(first))
    return sorted(indices, key=lambda i: first[base_of(i)])


def release_fold_caches() -> None:
    """Drop the shared LightGBM and Ridge fold entries of matrices that are gone: after the rerank, all of its per-base ones."""
    from ._lgb_shared_fold import prune_dead as prune_lgb
    from ._ridge_shared_fold import prune_dead as prune_ridge

    prune_lgb()
    prune_ridge()


__all__ = ["BoundedMemo", "PerBaseMatrices", "base_ordered", "release_fold_caches"]
