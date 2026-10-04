"""Run a callable on a daemon thread and hand back a ``Future``."""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from concurrent.futures import Future
from typing import Any

logger = logging.getLogger(__name__)


def start_daemon_task(fn: Callable[..., Any], *args: Any) -> Future:
    """Run ``fn(*args)`` on a fresh DAEMON thread and return a ``Future`` for its result.

    Unlike ``ThreadPoolExecutor``, nothing ever joins the thread: a caller that gives up on ``result(timeout=...)``
    really does move on, and a wedged task cannot hold up interpreter exit. The future is always settled, including
    when ``fn`` raises a ``BaseException``; that one is re-raised on the thread after the future carries it."""
    fut: Future = Future()

    def _run() -> None:
        """Run ``fn`` on the daemon thread and settle ``fut`` with its result or exception (skipped if the future was cancelled first)."""
        if not fut.set_running_or_notify_cancel():
            return
        try:
            fut.set_result(fn(*args))
        except BaseException as exc:  # handed to the waiting caller, which classifies it
            logger.debug("daemon task raised %s: %s", type(exc).__name__, exc)
            fut.set_exception(exc)
            if not isinstance(exc, Exception):
                raise

    threading.Thread(target=_run, name="mlframe-daemon-task", daemon=True).start()
    return fut
