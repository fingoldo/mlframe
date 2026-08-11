"""Per-call-site log throttling for hot loops."""
from __future__ import annotations

import logging
import threading

_counts: dict[str, int] = {}
_lock = threading.Lock()


def reset_throttle_counts(key: str | None = None) -> None:
    """Clear the per-key emission counter (all keys if ``key`` is None).

    Production call sites use a fixed, hardcoded ``key`` per throttled call site, so a test
    asserting on the *content* of a throttled log message is only reliable for the first
    ``max_count`` times that key fires across the whole process's lifetime -- any earlier test
    in the same worker that hit the same call site silently exhausts the budget. Call this before
    exercising such a call site in a test to reset it to a known (unthrottled) state.
    """
    with _lock:
        if key is None:
            _counts.clear()
        else:
            _counts.pop(key, None)


def log_throttle(logger: logging.Logger, key: str, level: int, msg: str, *args: object, max_count: int = 5, exc_info: bool = False) -> None:
    """Log at most ``max_count`` times per distinct ``key``, then stay silent for that key.

    Guards a per-iteration ``logger.warning(...)``/``logger.error(...)``/``logger.exception(...)``
    call inside a hot loop from flooding the log when many iterations hit the same failure condition
    (e.g. a systemic upstream issue): the first few occurrences still surface the signal, the rest
    are suppressed rather than compounding into spam under load. ``key`` identifies the call site
    (and, where useful, the failure shape), not the per-iteration payload -- distinct keys throttle
    independently. Pass ``exc_info=True`` (as ``logger.exception`` implies) to keep the traceback on
    the surfaced occurrences.
    """
    with _lock:
        n = _counts.get(key, 0)
        _counts[key] = n + 1
    if n < max_count:
        logger.log(level, msg, *args, exc_info=exc_info)
        if n == max_count - 1:
            logger.log(level, "%s: further occurrences suppressed (throttled)", key)
