"""A shared budget for the large transient host blocks of concurrent worker threads.

Several FE stages run on threads and several of their steps build a column block of the subsample plus float64 working copies (a bootstrap resample: its block of up to 256 MB and the copies of the KSG, copula, xi,
dcor and hsic scorers; the quantile codes of a (30000, 3000) matrix; ...). Each step sizes its block on its own, so the peak of ``p`` workers is ``p`` times a block: at 250000 x 1000 that took the process to 5.8 GB of growth,
and a machine with about 8 GB free stopped the fit. ``host_block_budget().reserve(nbytes)`` admits a step while the bytes in flight stay within the budget and makes the others wait for a release.

A request larger than the whole budget is admitted when nothing else holds a reservation (it would otherwise wait for ever), and a thread that already holds a reservation is admitted at once (no waiting while holding, so
nested use cannot deadlock). The budget is ``min(3 GiB, a third of the available RAM at the first use)``, at least 512 MiB; ``MLFRAME_HOST_BLOCK_BUDGET_MB`` sets it (``0`` switches the limit off).
"""

from __future__ import annotations

import contextlib
import functools
import logging
import os
import threading
import time
from collections.abc import Callable, Iterator
from typing import Any, TypeVar, cast

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

    def __getstate__(self) -> dict:
        """Pickle the budget figures, not the live condition or thread-local (both are made again on load)."""
        state = self.__dict__.copy()
        state.pop("_cond", None)
        state.pop("_local", None)
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        self._cond = threading.Condition()
        self._local = threading.local()

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


_F = TypeVar("_F", bound=Callable[..., Any])


def reserves_host_block_budget(estimate: "Callable[..., int]") -> "Callable[[_F], _F]":
    """Decorator: the call holds ``estimate(*args, **kwargs)`` bytes of the shared host block budget while it runs, so a family that builds a large frame of candidate columns does not run next to others doing the same.

    ``estimate`` gets the call's own arguments and returns the bytes the call keeps alive at its peak (an upper bound is fine: a request above the whole budget runs alone)."""

    def decorate(fn: _F) -> _F:
        """Wrap ``fn``."""

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            """``fn`` under a reservation of its estimated bytes."""
            try:
                nbytes = int(estimate(*args, **kwargs))
            except Exception as exc:  # a failing estimate must not stop the call
                logger.debug("host block budget: estimate failed (%r); no reservation", exc)
                nbytes = 0
            with host_block_budget().reserve(nbytes):
                return fn(*args, **kwargs)

        return cast(_F, wrapper)

    return decorate
