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


def map_on_edge_pool(fn: Callable[[Any], Any], items: Sequence[Any], *, min_items: int = 4) -> "list[Any]":
    """``[fn(item) for item in items]``, on the edge-jobs pool when there are at least ``min_items`` items and more than one job."""
    from . import _adaptive_nbins
    from ._fe_deadline import _state as _deadline_state, fe_deadline_scope
    from ._fit_context import capture_fit_context, fit_context_scope

    todo = list(items)
    deadline = getattr(_deadline_state, "deadline", None)
    context = capture_fit_context()

    def _run(item: Any) -> Any:
        """``fn(item)`` with the fit's wall-clock budget and per-fit scoring settings published to the worker thread."""
        with fe_deadline_scope(deadline), fit_context_scope(context):
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
