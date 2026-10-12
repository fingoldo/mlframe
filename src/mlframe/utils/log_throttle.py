"""Per-call-site log throttling for hot loops.

The implementation lives in ``pyutilz.dev.logginglib`` (``log_throttled`` for the count-based form, ``reset_log_throttles`` for the reset), shared with
the time-window ``log_throttle`` there; this module keeps the names mlframe's call sites use. ``log_throttle(logger, key, level, msg, *args,
max_count=5, exc_info=False)`` logs at most ``max_count`` times per ``key``, and ``reset_throttle_counts(key=None)`` forgets the counts of one key or of all.
"""

from __future__ import annotations

from pyutilz.dev.logginglib import log_throttle_count
from pyutilz.dev.logginglib import log_throttled as log_throttle
from pyutilz.dev.logginglib import reset_log_throttles as reset_throttle_counts

__all__ = ["log_throttle", "log_throttle_count", "reset_throttle_counts"]
