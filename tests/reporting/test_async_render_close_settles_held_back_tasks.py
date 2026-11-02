"""A held-back task whose predecessor finishes after close(wait=False) is settled as failed, not left pending forever (tracker 2026-11-02 wave 8, RES8-2)."""

import threading

import pytest

from mlframe.reporting._async_render import ReportRenderQueue


def test_follower_of_a_running_task_fails_after_abandoning_close():
    """close(wait=False) cancels queued work but not the running task; the follower waiting on it used to hang on result()."""
    release = threading.Event()
    started = threading.Event()

    def blocker():
        """Hold the worker until the test lets go."""
        started.set()
        release.wait(30)
        return 1

    q = ReportRenderQueue(backend="thread", workers=1)
    first = q.submit(blocker)
    follower = q.submit(lambda: 2, after_pending=True)
    assert started.wait(30)
    q.close(wait=False)
    release.set()
    first.result(30)
    with pytest.raises(RuntimeError, match="closed"):
        follower.result(10)
