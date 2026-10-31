"""Registry of the in-process caches of the FE stages that can be dropped when memory is needed.

Several stages keep results between calls (permutation blocks of the family gate, resample tables, column codes). They pay for themselves while memory is plentiful, but with every family on they
add up to gigabytes, and the memory-budget checks of the later stages (``_can_hoist_shared_buffer``) then refuse the fast batched paths and fall back to per-candidate loops. A stage registers a
function that drops its cache and returns the bytes it freed; the budget check releases the caches and asks again before it declines.
"""

from __future__ import annotations

import logging
import threading
from typing import Callable

logger = logging.getLogger(__name__)

_RELEASERS: "list[Callable[[], int]]" = []
_LOCK = threading.Lock()


def register_cache_releaser(release: "Callable[[], int]") -> "Callable[[], int]":
    """Add ``release`` (drops a cache, returns the bytes freed) to the registry; returns it so it can be used as a decorator."""
    with _LOCK:
        if release not in _RELEASERS:
            _RELEASERS.append(release)
    return release


def _run_releaser(release: "Callable[[], int]") -> int:
    """Bytes freed by one releaser; a failing releaser must not break the memory check that called it."""
    try:
        return int(release() or 0)
    except Exception as exc:
        logger.debug("cache releaser %r failed: %s", release, exc)
        return 0


def release_registered_caches() -> int:
    """Run every registered releaser and return the bytes they report freed (0 when there was nothing to drop)."""
    with _LOCK:
        releasers = list(_RELEASERS)
    freed = sum(_run_releaser(release) for release in releasers)
    if freed:
        logger.debug("released %.0f MB of stage caches under memory pressure", freed / 2**20)
    return freed
