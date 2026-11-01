"""Run independent per-item work on the edge-jobs thread pool and return the results in item order.

The FE families scan independent columns (or pairs, folds); each wraps the body of one item in a function and hands the list to :func:`map_on_edge_pool`, which publishes the fit's wall-clock deadline to the worker
threads, sizes the pool with ``_adaptive_nbins._resolve_edge_jobs`` and folds nothing itself: the caller folds the returned list in item order, so the result equals the serial loop.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Sequence


def map_on_edge_pool(fn: Callable[[Any], Any], items: Sequence[Any], *, min_items: int = 4) -> "list[Any]":
    """``[fn(item) for item in items]``, on the edge-jobs pool when there are at least ``min_items`` items and more than one job."""
    from . import _adaptive_nbins
    from ._fe_deadline import _state as _deadline_state, fe_deadline_scope

    todo = list(items)
    deadline = getattr(_deadline_state, "deadline", None)

    def _run(item: Any) -> Any:
        """``fn(item)`` with the fit's wall-clock budget published to the worker thread."""
        with fe_deadline_scope(deadline):
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
