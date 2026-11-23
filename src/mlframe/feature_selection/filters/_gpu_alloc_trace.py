"""Allocation trace of the device memory pools during a fit: how many allocations, the peak the pool reached, the largest allocation sites and the context of every out-of-memory failure.

The pool-full failures of a 500000 x 1000 fit (a held block of codes starving every later stage) were found with an allocator that logged each large allocation with its call site. This module keeps that as a
permanent, cheap record: the hook costs about 0.4 microseconds an allocation (24482 allocations in a 100000 x 100 fit, 0.01 s), the call stack is read only for allocations of at least ``_BIG_BYTES``, and the
result is stored on the estimator as ``gpu_alloc_trace_``.

The wrapper keeps ``__self__`` pointing at the pool, which ``_device_sort_safe._current_pool`` reads to size device blocks; ``MLFRAME_GPU_ALLOC_TRACE=0`` turns the trace off.
"""

from __future__ import annotations

import contextlib
import logging
import os
import threading
import traceback
from collections.abc import Iterator
from typing import Any

logger = logging.getLogger(__name__)

_BIG_BYTES = 32 * 1024 * 1024
_TOP_SITES = 8
_MAX_OOM_EVENTS = 8


class _TraceState:
    """Counters of one trace scope (empty at the start), shared by the tracers of every pool the fit's threads allocate from."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.n_allocations = 0
        self.total_bytes = 0
        self.peak_pool_used = 0
        self.n_oom = 0
        self.sites: "list[tuple[int, str, str]]" = []
        self.oom_events: "list[dict]" = []

    def __getstate__(self) -> dict:
        """Pickle the counters, not the live lock (a fresh one is made on load)."""
        state = self.__dict__.copy()
        state.pop("lock", None)
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        self.lock = threading.Lock()

    def record_big(self, size: int, pool_used: int, where: str) -> None:
        """Remember a large allocation: the peak the pool reached with it and, among the largest sites, this one."""
        with self.lock:
            self.peak_pool_used = max(self.peak_pool_used, pool_used + size)
            self.sites.append((int(size), where, threading.current_thread().name))
            if len(self.sites) > 4 * _TOP_SITES:
                self.sites = sorted(self.sites, reverse=True)[:_TOP_SITES]

    def record_oom(self, size: int, pool: Any, where: str) -> None:
        """Count an out-of-memory failure and keep the context of the first few."""
        with self.lock:
            self.n_oom += 1
            if len(self.oom_events) < _MAX_OOM_EVENTS:
                try:
                    used, limit = int(pool.used_bytes()), int(pool.get_limit())
                except Exception as exc:  # the pool is only described, never needed
                    logger.debug("gpu alloc trace: pool stats unavailable: %s", exc)
                    used, limit = -1, -1
                self.oom_events.append({"requested_mb": round(size / 1e6, 1), "pool_used_mb": round(used / 1e6, 1), "pool_limit_mb": round(limit / 1e6, 1), "where": where, "thread": threading.current_thread().name})

    def summary(self) -> dict:
        """The record stored on the estimator."""
        with self.lock:
            largest = sorted(self.sites, reverse=True)[:_TOP_SITES]
            return {
                "n_allocations": int(self.n_allocations),
                "total_gb": round(self.total_bytes / 1e9, 2),
                "peak_pool_used_mb": round(self.peak_pool_used / 1e6, 1),
                "n_oom": int(self.n_oom),
                "largest_allocations": [{"mb": round(s / 1e6, 1), "where": w, "thread": t} for s, w, t in largest],
                "oom_events": list(self.oom_events),
            }


_STATE: "_TraceState | None" = None
_STATE_LOCK = threading.Lock()


def _where() -> str:
    """The innermost four mlframe frames of the calling stack as ``module:function:line``, outermost first."""
    frames = [f for f in traceback.extract_stack(limit=18) if "mlframe" in f.filename and "_gpu_alloc_trace" not in f.filename]
    return " < ".join(f"{f.filename.replace(chr(92), '/').rsplit('/', 1)[-1][:-3]}:{f.name}:{f.lineno}" for f in reversed(frames[-4:]))


class _TracingAllocator:
    """An allocator that forwards to ``pool.malloc`` and records what it forwarded; ``__self__`` is the pool, as for a bound ``pool.malloc``. Built around ``pool``, reporting to ``state``."""

    def __init__(self, pool: Any, state: _TraceState) -> None:
        self.__self__ = pool
        self._pool = pool
        self._state = state

    def __call__(self, size: int) -> Any:
        """Allocate ``size`` bytes from the pool, recording the allocation (and, for a large one or a failure, its call site)."""
        state = self._state
        state.n_allocations += 1
        state.total_bytes += int(size)
        try:
            mem = self._pool.malloc(size)
        except Exception as exc:
            if type(exc).__name__ == "OutOfMemoryError":
                state.record_oom(int(size), self._pool, _where())
            raise
        if size >= _BIG_BYTES:
            state.record_big(int(size), int(self._pool.used_bytes()), _where())
        return mem


def traced_allocator(pool: Any) -> Any:
    """The allocator a thread should run with for ``pool``: the tracing wrapper while a trace scope is active, ``pool.malloc`` otherwise."""
    state = _STATE
    return pool.malloc if state is None else _TracingAllocator(pool, state)


@contextlib.contextmanager
def gpu_alloc_trace_scope(estimator: Any = None) -> "Iterator[Any]":
    """Trace the device allocations of the calling thread (default pool) and of the stage threads (``traced_allocator``) inside the block; on exit the record is stored as ``estimator.gpu_alloc_trace_`` and logged.

    A no-op without cupy, with ``MLFRAME_GPU_ALLOC_TRACE=0``, or inside another trace scope (the outer one keeps the record)."""
    global _STATE
    if os.environ.get("MLFRAME_GPU_ALLOC_TRACE", "1").strip().lower() in ("0", "false", "off", "no"):
        yield None
        return
    try:
        import cupy as cp
    except ImportError:
        yield None
        return
    with _STATE_LOCK:
        if _STATE is not None:
            outer = True
        else:
            outer = False
            _STATE = _TraceState()
        state = _STATE
    if outer:
        yield None
        return
    try:
        with cp.cuda.using_allocator(_TracingAllocator(cp.get_default_memory_pool(), state)):
            yield state
    finally:
        with _STATE_LOCK:
            _STATE = None
        summary = state.summary()
        if estimator is not None:
            try:
                estimator.gpu_alloc_trace_ = summary
            except Exception as exc:  # a read-only estimator only loses the attribute
                logger.debug("gpu alloc trace: could not store the record (%r)", exc)
        logger.info(
            "gpu alloc trace: %d allocations (%.1f GB), pool peak %.0f MB, %d out-of-memory; largest: %s",
            summary["n_allocations"], summary["total_gb"], summary["peak_pool_used_mb"], summary["n_oom"],
            "; ".join(f"{a['mb']:.0f} MB at {a['where'].split(' < ')[0]}" for a in summary["largest_allocations"][:3]) or "none",
        )
