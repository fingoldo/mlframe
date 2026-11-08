"""Registry of the in-process caches of the FE stages that can be dropped when memory is needed.

Several stages keep results between calls (permutation blocks of the family gate, resample tables, column codes). They pay for themselves while memory is plentiful, but with every family on they
add up to gigabytes, and the memory-budget checks of the later stages (``_can_hoist_shared_buffer``) then refuse the fast batched paths and fall back to per-candidate loops. A stage registers a
function that drops its cache and returns the bytes it freed; the budget check releases the caches and asks again before it declines.
"""

from __future__ import annotations

import logging
import threading
from typing import Callable
from mlframe.utils.log_throttle import log_throttle

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
        log_throttle(logger, "_cache_release.except_32", logging.INFO, "cache releaser %r failed: %s", release, exc)
        return 0


def release_registered_caches() -> int:
    """Run every registered releaser and return the bytes they report freed (0 when there was nothing to drop)."""
    with _LOCK:
        releasers = list(_RELEASERS)
    freed = sum(_run_releaser(release) for release in releasers)
    if freed:
        logger.debug("released %.0f MB of stage caches under memory pressure", freed / 2**20)
    return freed


#: Free host RAM below which a family gate drops the registered stage caches before the family starts: the larger of this many bytes and this share of the total.
_LOW_FREE_BYTES = 2 * 1024**3
_LOW_FREE_FRACTION = 0.15
_PROBE_WARNED = False


def release_if_memory_low() -> int:
    """Drop the registered stage caches when the free host RAM is below ``max(2 GB, 15% of the total)``; returns the bytes freed (0 when memory is plentiful or unknown).

    The caches were released only when a budget check refused a fast path, so on a long frame they accumulated through the FE stages (about 1 GB after a 300000-row fit,
    several GB at 10 million rows) until the next large allocation failed; checked at every family start, they are dropped before the family needs the room."""
    try:
        import psutil

        vm = psutil.virtual_memory()
    except Exception as exc:  # no psutil: nothing is known about the free memory, keep the caches
        global _PROBE_WARNED
        if not _PROBE_WARNED:
            _PROBE_WARNED = True
            logger.warning("free-memory probe failed (%r); keeping the caches (reported once per process)", exc)
        return 0
    if vm.available >= max(_LOW_FREE_BYTES, int(_LOW_FREE_FRACTION * vm.total)):
        return 0
    freed = release_registered_caches()
    if freed:
        log_throttle(logger, "_cache_release.low_memory", logging.INFO, "free RAM %.1f GB: released %.0f MB of stage caches before the next FE family", vm.available / 1e9, freed / 2**20)
    return freed
