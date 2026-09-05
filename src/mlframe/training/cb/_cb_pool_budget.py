"""Byte budget for the two module-level CatBoost ``Pool`` caches.

``_CB_POOL_CACHE`` (train) and ``_CB_VAL_POOL_CACHE`` (val) were each bounded by an entry count alone. A
quantised ``Pool`` owns the whole dataset at roughly one byte per (row, feature) cell, so on the 7M-row
frames ``_predict_guards`` cites for its 50-70s rebuild, 500 features put a single Pool near 3.5 GB and the
16-entry cap at ~56 GB per cache -- retained in a module-level dict that outlives every fit, and the two
caps are independent. A realistic suite over four targets and two folds fills eight val entries and holds
~28 GB for the life of the process.

The entry cap stays as a secondary bound. What is added here is an aggregate byte ceiling per cache, plus a
refusal to admit any single Pool larger than that ceiling: caching one entry that alone exhausts the budget
buys nothing, since the next insert evicts it again.

The default is deliberately generous relative to the repo's other byte budgets (512 MiB for a prebin code
matrix, 1 GiB for a resample-index matrix). This cache exists to avoid a minute-long rebuild, so one large
Pool should still fit; sixteen never should.
"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

# Aggregate ceiling per cache, not per entry.
_CB_POOL_CACHE_MAX_BYTES_DEFAULT: int = 8 * 1024 * 1024 * 1024  # 8 GiB

# Byte totals per cache, keyed by the caller's cache label ("train" / "val"), so the two dicts -- which live
# in different modules -- can share one helper without either owning the other's bookkeeping.
_CACHE_SIZES: dict[str, dict[Any, int]] = {}


def cb_pool_cache_max_bytes() -> int:
    """Aggregate byte ceiling for one Pool cache (env-overridable via ``MLFRAME_CB_POOL_CACHE_MAX_BYTES``)."""
    raw = os.environ.get("MLFRAME_CB_POOL_CACHE_MAX_BYTES")
    try:
        return int(raw) if raw and raw.strip() else _CB_POOL_CACHE_MAX_BYTES_DEFAULT
    except (TypeError, ValueError):
        logger.debug("MLFRAME_CB_POOL_CACHE_MAX_BYTES=%r is not an integer; using the default.", raw)
        return _CB_POOL_CACHE_MAX_BYTES_DEFAULT


def estimate_pool_bytes(pool: Any) -> int:
    """Approximate the resident size of a quantised CatBoost ``Pool``, or 0 when it cannot be measured.

    A quantised Pool stores one byte per (row, feature) cell plus border tables, so rows x columns is the
    right order of magnitude. Returning 0 for anything that does not answer ``num_row``/``num_col`` keeps a
    Pool-like object the caller passed in cacheable rather than silently refusing it.
    """
    try:
        return int(pool.num_row()) * int(pool.num_col())
    except Exception as exc:
        logger.debug("estimate_pool_bytes: %s does not report its shape (%s); not counting it.", type(pool).__name__, exc)
        return 0


def admit_pool(cache: dict, label: str, key: Any, pool: Any) -> bool:
    """Evict oldest-first until ``pool`` fits the byte ceiling, then record it; return whether to cache it.

    Called at the INSERT site rather than before the build, because the Pool's size is not known until it
    exists. The existing entry-count eviction stays where it is, ahead of the build, where dropping an old
    Pool before allocating the new one is what keeps the peak down.

    Returns ``False`` when this Pool alone exceeds the ceiling: caching it would evict everything else and
    then be evicted itself by the next insert, so the caller should use it and not store it.
    """
    sizes = _CACHE_SIZES.setdefault(label, {})
    for stale in [k for k in sizes if k not in cache]:
        sizes.pop(stale, None)

    nbytes = estimate_pool_bytes(pool)
    ceiling = cb_pool_cache_max_bytes()
    if nbytes > ceiling:
        logger.info(
            "[cb-pool-cache] not caching a %.2f GiB %s Pool: it alone exceeds the %.2f GiB budget.",
            nbytes / 1024**3, label, ceiling / 1024**3,
        )
        return False

    while cache and sum(sizes.values()) + nbytes > ceiling:
        oldest = next(iter(cache))
        cache.pop(oldest, None)
        sizes.pop(oldest, None)
        logger.info("[cb-pool-cache] evicted the oldest %s Pool to stay under the %.2f GiB budget.", label, ceiling / 1024**3)

    sizes[key] = nbytes
    return True


def forget_pool_bytes(label: str, key: Any) -> None:
    """Drop one entry's recorded size, for a caller that evicted it outside ``evict_for_budget``."""
    _CACHE_SIZES.get(label, {}).pop(key, None)


def cache_bytes(label: str) -> int:
    """Current recorded total for one cache, for tests and diagnostics."""
    return sum(_CACHE_SIZES.get(label, {}).values())


def reset_cache_bytes(label: str | None = None) -> None:
    """Clear the size bookkeeping, for a caller that cleared the cache dict itself."""
    if label is None:
        _CACHE_SIZES.clear()
    else:
        _CACHE_SIZES.pop(label, None)
