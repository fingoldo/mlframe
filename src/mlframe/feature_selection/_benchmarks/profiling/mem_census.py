"""Census of the live containers that hold large numpy arrays: who keeps the memory of a fit alive.

``census()`` walks the objects the garbage collector tracks (dicts, lists, tuples, deques, DataFrames), sums the bytes of the numpy arrays each holds directly (a view counts for nothing, its base is counted where it lives) and names the
owner: the module attribute a dict is bound to, or the instance whose ``__dict__`` it is. A plain ndarray is not tracked by the collector, so an array held only by a local variable or a C extension is not found; what the census finds are the
caches, memos, registries and fitted-state dicts, which is where a fit leaves its memory behind. Used by ``rss_scale_fit`` with ``RSS_CENSUS_AT_S``.
"""

from __future__ import annotations

import collections
import gc
import sys
from typing import Any

import numpy as np

_MIN_BYTES = 16 * 1024 * 1024


def _owner(a: np.ndarray) -> np.ndarray:
    """The array that owns the buffer ``a`` is a view of."""
    while isinstance(a.base, np.ndarray):
        a = a.base
    return a


def _array_bytes(v: Any, seen: "set[int] | None" = None) -> int:
    """Bytes of the buffers the value holds that were not counted yet (``seen`` holds the ids of the owning arrays counted so far): an ndarray or the arrays directly in a tuple / list of them, a view counted through its owner."""
    seen = set() if seen is None else seen
    arrays: "list[np.ndarray]" = []
    if isinstance(v, np.ndarray):
        arrays = [v]
    elif isinstance(v, (tuple, list)) and 0 < len(v) <= 64:
        arrays = [x for x in v if isinstance(x, np.ndarray)]
    total = 0
    for a in arrays:
        o = _owner(a)
        if id(o) not in seen:
            seen.add(id(o))
            total += int(o.nbytes)
    return total


def _frame_bytes(df: Any, seen: "set[int]") -> int:
    """Bytes of the buffers behind the blocks of a pandas frame that were not counted yet (frames share blocks under copy-on-write)."""
    total = 0
    for blk in df._mgr.blocks:
        v = blk.values
        if isinstance(v, np.ndarray):
            total += _array_bytes(v, seen)
    return total


def _module_labels() -> "dict[int, str]":
    """``id(container) -> "module.name"`` for the dicts, lists and deques bound at module level."""
    labels: "dict[int, str]" = {}
    for mname, mod in list(sys.modules.items()):
        d = getattr(mod, "__dict__", None)
        if not isinstance(d, dict):
            continue
        for k, v in list(d.items()):
            if isinstance(v, (dict, list, collections.deque)):
                labels[id(v)] = f"{mname}.{k}"
    return labels


def census(top: int = 25) -> "list[tuple[int, int, str, str]]":
    """``(bytes, entries, kind, owner)`` of the heaviest containers, largest first."""
    import pandas as pd

    labels = _module_labels()
    rows: "list[tuple[int, int, str, str]]" = []
    seen: "set[int]" = set()
    for o in gc.get_objects():
        try:
            isinstance(o, dict)  # a dead weakref proxy raises here
        except ReferenceError:
            continue
        total = 0
        n = 0
        kind = type(o).__name__
        if isinstance(o, dict):
            for v in list(o.values()):
                b = _array_bytes(v, seen)
                if b:
                    total += b
                    n += 1
        elif isinstance(o, (list, collections.deque)) and len(o) > 0:
            for v in list(o):
                b = _array_bytes(v, seen)
                if b:
                    total += b
                    n += 1
        elif isinstance(o, pd.DataFrame):
            try:
                total = _frame_bytes(o, seen)
            except Exception:  # a frame that cannot report is skipped
                continue
            n = int(o.shape[1])
            kind = "DataFrame"
        else:
            continue
        if total >= _MIN_BYTES:
            rows.append((total, n, kind, labels.get(id(o), "")))
    rows.sort(reverse=True)
    out = []
    for total, n, kind, owner in rows[:top]:
        out.append((total, n, kind, owner or "(unnamed)"))
    return out


def name_owners(rows: "list[tuple[int, int, str, str]]") -> "list[tuple[int, int, str, str]]":
    """Name the unnamed dicts by the instance whose ``__dict__`` they are (found among the referrers)."""
    named = []
    pool = {}
    for o in gc.get_objects():
        try:
            if isinstance(o, dict):
                pool[id(o)] = o
        except ReferenceError:  # a dead weakref proxy
            continue
    by_size = {}
    for o in pool.values():
        t = sum(_array_bytes(v) for v in list(o.values()))
        by_size.setdefault(t, []).append(o)
    for total, n, kind, owner in rows:
        if owner == "(unnamed)" and kind == "dict":
            for cand in by_size.get(total, []):
                for ref in gc.get_referrers(cand):
                    d = getattr(ref, "__dict__", None)
                    if d is cand and not isinstance(ref, type):
                        owner = f"{type(ref).__module__}.{type(ref).__name__}.__dict__ (keys: {', '.join(sorted(map(str, cand))[:4])}...)"
                        break
                if owner != "(unnamed)":
                    break
        named.append((total, n, kind, owner))
    return named


def render(rows: "list[tuple[int, int, str, str]]") -> str:
    """The census as text lines."""
    return "\n".join(f"  {b / 1e6:8.0f} MB in {n:5d} arrays  {kind:12s} {owner}" for b, n, kind, owner in rows)
