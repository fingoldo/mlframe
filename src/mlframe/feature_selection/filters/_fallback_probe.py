"""One shape for a best-effort probe whose failure must fall back to a default without disappearing from the logs.

Many selection stages wrap an estimator call in ``try: ... except Exception: <debug line>; <default>``. The default is usually a deliberate
polarity (retain a possibly-genuine feature, exclude an unscorable candidate), but at debug level a persistent fault silently changes what a
fit selects with nothing an operator can see. ``call_or_default`` keeps the polarity and makes the fallback visible, throttled per call site so
a probe inside a candidate loop cannot flood the log.
"""

from __future__ import annotations

import logging
from typing import Callable, TypeVar

from mlframe.utils.log_throttle import log_throttle

T = TypeVar("T")

_logger = logging.getLogger("mlframe.feature_selection.filters.mrmr")


def call_or_default(fn: Callable[[], T], default: T, *, key: str, message: str, logger: logging.Logger | None = None) -> T:
    """Return ``fn()``; if it raises, log ``message`` with the exception at WARNING (throttled under ``key``) and return ``default``."""
    try:
        return fn()
    except Exception as exc:
        log_throttle(logger or _logger, key, logging.WARNING, "%s (%s: %s)", message, type(exc).__name__, exc)
        return default
