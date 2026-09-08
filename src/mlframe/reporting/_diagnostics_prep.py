"""The row/column preparation both error diagnostics do, done once per run instead of twice.

``render_split_error_diagnostics`` and ``render_slice_finder_diagnostic`` are separate entry points that a
report calls one after the other with the SAME frame, targets, task and seed. Each independently computed
the per-row error, drew the same bounded worst-error-preserving sample from it (same seed, so the same
indices), capped the same columns and gathered the same rows -- and then handed its own copy of that
sub-frame to a builder that densified it. The densify alone was measured at 0.29 s on 100k x 200
all-numeric and 2.55 s with twenty object columns, so doing it twice cost 0.6-5.1 s per run.

The prep is memoised on the IDENTITY of its inputs rather than their content: hashing a 100 GB frame to
avoid a 2.5 s densify would be absurd. Identity alone is unsound -- a freed object's id is reused -- so each
entry holds WEAK references and a hit is confirmed against them; weak, not strong, because pinning the
caller's frame is exactly what this package's memory rules forbid. Returning the SAME sub-frame object to
both callers is also what lets the resolver's own identity cache spare the second densify.
"""

from __future__ import annotations

import logging
import threading
import weakref
from typing import Any, Callable, Optional, Tuple

logger = logging.getLogger(__name__)

# One report renders one frame at a time; a couple of entries covers the two diagnostics plus a retry.
_PREP_CACHE_MAX = 4
_PREP_CACHE: dict = {}
_PREP_LOCK = threading.Lock()


def _weak(obj: Any) -> Optional[Callable[[], Any]]:
    """A weak reference to ``obj``, or ``None`` when the object does not support one."""
    try:
        return weakref.ref(obj)
    except TypeError:
        return None


def shared_error_prep(df: Any, y_true: Any, y_pred: Any, task: str, seed: int, build: Callable[[], Tuple]) -> Tuple:
    """``build()``'s result for these exact inputs, computed once and reused by the second diagnostic.

    ``build`` returns whatever the caller needs (loss, sample indices, sub-frame, names); it is only called
    on a miss. Falls back to calling ``build`` directly whenever the inputs cannot be weak-referenced.
    """
    refs = [_weak(o) for o in (df, y_true, y_pred)]
    if any(r is None for r in refs):
        return build()
    key = (id(df), id(y_true), id(y_pred), task, int(seed))
    with _PREP_LOCK:
        entry = _PREP_CACHE.get(key)
    if entry is not None:
        held, result = entry
        if all(r() is o for r, o in zip(held, (df, y_true, y_pred))):
            return tuple(result)
    result = build()
    with _PREP_LOCK:
        if len(_PREP_CACHE) >= _PREP_CACHE_MAX:
            _PREP_CACHE.pop(next(iter(_PREP_CACHE)), None)
        _PREP_CACHE[key] = (refs, result)
    return result


def clear_shared_error_prep() -> None:
    """Drop every memoised prep. For tests, and for a caller that wants the frame released immediately."""
    with _PREP_LOCK:
        _PREP_CACHE.clear()
