"""Run independent per-item work on the edge-jobs thread pool and return the results in item order.

The FE families scan independent columns (or pairs, folds); each wraps the body of one item in a function and hands the list to :func:`map_on_edge_pool`, which publishes the fit's wall-clock deadline to the worker
threads, sizes the pool with ``_adaptive_nbins._resolve_edge_jobs`` and folds nothing itself: the caller folds the returned list in item order, so the result equals the serial loop.
"""

from __future__ import annotations

import contextlib
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Iterator, Sequence

#: FE stage cascades running concurrently right now (``_fe_stage_concurrency``): ``_resolve_edge_jobs`` gives each family's pool its share of the cores, so
#: three cascades do not run 3 x cores threads on the cores
_ACTIVE_STAGES = [1]
_ACTIVE_LOCK = threading.Lock()


def stage_finished() -> None:
    """One of the concurrently running FE stage cascades has ended: the cascades still running divide the cores among themselves (``_resolve_edge_jobs`` reads the live count), instead of keeping the share they
    had when the wave started while the cores of the finished cascade sit idle. The count never goes below 1; ``concurrent_stages`` restores the previous value when the whole wave ends."""
    with _ACTIVE_LOCK:
        _ACTIVE_STAGES[0] = max(1, _ACTIVE_STAGES[0] - 1)


@contextlib.contextmanager
def concurrent_stages(n: int) -> "Iterator[None]":
    """Mark ``n`` FE stage cascades as running together for the duration of the block."""
    with _ACTIVE_LOCK:
        prev = _ACTIVE_STAGES[0]
        _ACTIVE_STAGES[0] = max(1, int(n))
    try:
        yield
    finally:
        with _ACTIVE_LOCK:
            _ACTIVE_STAGES[0] = prev


def _capture_device_scope() -> "Callable[[], Any]":
    """A context-manager factory that re-enters the CUDA device and the memory allocator the CALLING thread uses (both are thread-local: a worker thread starts on device 0 with the default allocator, so a
    stage placed on another device slot, or under a slot's pool, would hand its device arrays to kernels of the wrong device or bypass the slot's budget). A no-op scope without a usable CUDA device."""
    from contextlib import nullcontext

    try:
        import cupy as cp

        device_id = int(cp.cuda.Device().id)
        allocator = cp.cuda.get_allocator()
    except Exception:  # no cupy / no device: the workers are host-only
        return nullcontext

    from contextlib import ExitStack

    def _scope() -> "ExitStack":
        """Enter the captured device and allocator on this (worker) thread."""
        stack = ExitStack()
        stack.enter_context(cp.cuda.Device(device_id))
        stack.enter_context(cp.cuda.using_allocator(allocator))
        return stack

    return _scope


def map_on_edge_pool(fn: Callable[[Any], Any], items: Sequence[Any], *, min_items: int = 4) -> "list[Any]":
    """``[fn(item) for item in items]``, on the edge-jobs pool when there are at least ``min_items`` items and more than one job."""
    from . import _adaptive_nbins
    from ._fe_deadline import _state as _deadline_state, fe_deadline_scope
    from ._fit_context import capture_fit_context, fit_context_scope

    todo = list(items)
    deadline = getattr(_deadline_state, "deadline", None)
    context = capture_fit_context()
    device_scope = _capture_device_scope()

    def _run(item: Any) -> Any:
        """``fn(item)`` with the fit's wall-clock budget, per-fit scoring settings and the caller's CUDA device and allocator published to the worker thread."""
        with fe_deadline_scope(deadline), fit_context_scope(context), device_scope():
            return fn(item)

    jobs = _adaptive_nbins._resolve_edge_jobs(None, len(todo))
    if jobs > 1 and len(todo) >= min_items:
        with ThreadPoolExecutor(max_workers=jobs) as pool:
            return list(pool.map(_run, todo))
    return [_run(item) for item in todo]


def until_first_none(results: Sequence[Any]) -> "list[Any]":
    """The leading results up to (not including) the first ``None``: a worker returns ``None`` once the fit's wall-clock budget has passed, and the serial loop stopped there."""
    out: list = []
    for r in results:
        if r is None:
            break
        out.append(r)
    return out
