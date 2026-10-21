"""A failed warm-up submit is recorded in the queue summary, not only written to the log."""

from __future__ import annotations

from mlframe.reporting._async_render import ReportRenderQueue


class _RefusingExecutor:
    """Executor stand-in whose submit always fails, as a process pool does when it cannot spawn workers."""

    def submit(self, fn, *args, **kwargs):
        """Refuse the task."""
        raise RuntimeError("spawn refused")


def test_warm_up_submit_failure_is_recorded_in_the_summary(monkeypatch):
    """A warm-up submit that raises counts as one failure in the summary, not only a log line."""
    queue = ReportRenderQueue(backend="process", workers=2)
    monkeypatch.setattr(queue, "_get_executor", lambda: _RefusingExecutor())
    try:
        queue.warm()
        summary = queue.summary()
    finally:
        queue.close()
    assert summary.failed == 1
    assert summary.failures[0].name == "render queue warm-up"
    assert "spawn refused" in summary.failures[0].error
