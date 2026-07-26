"""Per-call-site log throttling for hot loops."""
from __future__ import annotations

import logging
import threading

_counts: dict[str, int] = {}
_lock = threading.Lock()


def log_throttle(logger: logging.Logger, key: str, level: int, msg: str, *args: object, max_count: int = 5) -> None:
    """Log at most ``max_count`` times per distinct ``key``, then stay silent for that key.

    Guards a per-iteration ``logger.warning(...)``/``logger.error(...)`` call inside a hot loop
    from flooding the log when many iterations hit the same failure condition (e.g. a systemic
    upstream issue): the first few occurrences still surface the signal, the rest are suppressed
    rather than compounding into spam under load. ``key`` identifies the call site (and, where
    useful, the failure shape), not the per-iteration payload -- distinct keys throttle independently.
    """
    with _lock:
        n = _counts.get(key, 0)
        _counts[key] = n + 1
    if n < max_count:
        logger.log(level, msg, *args)
        if n == max_count - 1:
            logger.log(level, "%s: further occurrences suppressed (throttled)", key)
