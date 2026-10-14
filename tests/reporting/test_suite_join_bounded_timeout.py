"""Suite finalisation must not wait forever on a chart render that never finishes.

The c0023 combo stalled at finalisation with ``join_suite_render_queue`` blocked on a render task that never completed, because the
join had no upper bound. The join is now bounded, and the unfinished work is reported rather than awaited.
"""

from __future__ import annotations

import threading

import pytest

from mlframe.reporting import async_render_hooks
from mlframe.reporting._async_render import ReportRenderQueue


def test_join_suite_render_queue_returns_when_a_render_task_never_finishes(monkeypatch):
    """A stuck render task releases the suite join after the bound instead of blocking it indefinitely."""
    monkeypatch.setattr(async_render_hooks, "_SUITE_JOIN_TIMEOUT_S", 0.5)
    never = threading.Event()
    q = ReportRenderQueue(backend="thread", workers=1)
    try:
        q.submit(lambda: never.wait(), name="stuck")

        # The join returns while the stuck task is still running; a join that blocked on it would hang here,
        # which pytest-timeout reports, so no wall-clock bound is asserted.
        async_render_hooks.join_suite_render_queue(q, {}, final=False)
        assert q.pending() == 1
    finally:
        never.set()
        q.close(wait=False)


def test_join_suite_render_queue_is_quiet_when_every_task_finishes(monkeypatch):
    """A queue whose tasks all finish joins without leaving anything pending."""
    monkeypatch.setattr(async_render_hooks, "_SUITE_JOIN_TIMEOUT_S", 30.0)
    q = ReportRenderQueue(backend="thread", workers=1)
    try:
        q.submit(lambda: None, name="quick")

        async_render_hooks.join_suite_render_queue(q, {}, final=False)

        assert q.pending() == 0
    finally:
        q.close(wait=False)


@pytest.mark.parametrize("final", [True])
def test_final_join_still_closes_the_queue(monkeypatch, final):
    """The suite's final pass keeps its close-and-unlink behaviour regardless of the new bound."""
    monkeypatch.setattr(async_render_hooks, "_SUITE_JOIN_TIMEOUT_S", 0.5)
    q = ReportRenderQueue(backend="thread", workers=1)
    async_render_hooks.join_suite_render_queue(q, {}, final=final)
    assert q.pending() == 0
