"""A shared budget for the large transient host blocks of concurrent worker threads.

Several FE stages run on threads and several of their steps build a column block of the subsample plus float64 working copies (a bootstrap resample: its block of up to 256 MB and the copies of the KSG, copula, xi,
dcor and hsic scorers; the quantile codes of a (30000, 3000) matrix; ...). Each step sizes its block on its own, so the peak of ``p`` workers is ``p`` times a block: at 250000 x 1000 that took the process to 5.8 GB of growth,
and a machine with about 8 GB free stopped the fit. ``host_block_budget().reserve(nbytes)`` admits a step while the bytes in flight stay within the budget and makes the others wait for a release.

A request larger than the whole budget is admitted when nothing else holds a reservation (it would otherwise wait for ever), and a thread that already holds a reservation is admitted at once (no waiting while holding, so
nested use cannot deadlock). The budget is ``min(3 GiB, a third of the available RAM at the first use)``, at least 512 MiB; ``MLFRAME_HOST_BLOCK_BUDGET_MB`` sets it (``0`` switches the limit off).
"""

from __future__ import annotations

import contextlib
import logging
import os
import threading
import time
from collections.abc import Iterator

logger = logging.getLogger(__name__)

_MIN_BUDGET = 512 * 1024 * 1024
_MAX_BUDGET = 3 * 1024 * 1024 * 1024


class HostBlockBudget:
    """Bytes of transient host blocks that may be in flight at once; ``total_bytes <= 0`` admits everything."""

    def __init__(self, total_bytes: int) -> None:
        self.total = int(total_bytes)
        self._used = 0
        self._cond = threading.Condition()
        self._local = threading.local()
        self.waits = 0
        self.waited_s = 0.0

    @contextlib.contextmanager
    def reserve(self, nbytes: int) -> "Iterator[None]":
        """Hold ``nbytes`` of the budget for the duration of the block."""
        nbytes = max(0, int(nbytes))
        if self.total <= 0 or nbytes == 0:
            yield
            return
        held = getattr(self._local, "held", 0)
        with self._cond:
            if held == 0:
                t0 = None
                while self._used > 0 and self._used + nbytes > self.total:
                    if t0 is None:
                        t0 = time.perf_counter()
                        self.waits += 1
                    self._cond.wait(timeout=5.0)
                if t0 is not None:
                    self.waited_s += time.perf_counter() - t0
            self._used += nbytes
        self._local.held = held + nbytes
        try:
            yield
        finally:
            self._local.held = held
            with self._cond:
                self._used -= nbytes
                self._cond.notify_all()

    def in_flight(self) -> int:
        """Bytes reserved right now."""
        with self._cond:
            return self._used


_BUDGET: "HostBlockBudget | None" = None
_BUDGET_LOCK = threading.Lock()


def _default_total() -> int:
    """The budget in bytes: the environment value, else a third of the available RAM within the bounds."""
    raw = os.environ.get("MLFRAME_HOST_BLOCK_BUDGET_MB", "").strip()
    if raw:
        try:
            return int(float(raw) * 1024 * 1024)
        except ValueError:
            logger.warning("MLFRAME_HOST_BLOCK_BUDGET_MB=%r is not a number; using the default", raw)
    try:
        import psutil

        avail = int(psutil.virtual_memory().available)
    except Exception:  # without psutil the cap alone applies
        return _MAX_BUDGET
    return max(_MIN_BUDGET, min(_MAX_BUDGET, avail // 3))


def host_block_budget() -> HostBlockBudget:
    """The process-wide budget, created on first use."""
    global _BUDGET
    with _BUDGET_LOCK:
        if _BUDGET is None:
            _BUDGET = HostBlockBudget(_default_total())
        return _BUDGET


def reset_host_block_budget() -> None:
    """Forget the process-wide budget so the next use sizes it again (tests, and a fit that wants the current free RAM)."""
    global _BUDGET
    with _BUDGET_LOCK:
        _BUDGET = None
