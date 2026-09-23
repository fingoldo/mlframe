"""Interpreter-exit handlers must neither abandon in-flight work nor hang the exit."""

import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

import mlframe.training.crash_diagnostics as cd
import mlframe.training.feature_handling.registry as reg


@pytest.fixture
def own_executor(monkeypatch):
    """A private executor per test: the handler under test shuts the module one down for good."""
    ex = ThreadPoolExecutor(max_workers=2, thread_name_prefix="test-prewarm")
    monkeypatch.setattr(reg, "_PREWARM_EXECUTOR", ex)
    monkeypatch.setattr(reg, "_PREWARM_FUTURES", {})
    yield ex
    ex.shutdown(wait=False, cancel_futures=True)


def test_the_prewarm_exit_handler_waits_for_a_running_prewarm(own_executor, monkeypatch):
    """A prewarm loading a provider onto the GPU must not be left running into interpreter teardown."""
    finished = threading.Event()

    def _slow_prewarm():
        time.sleep(0.5)
        finished.set()

    reg._PREWARM_FUTURES["sig"] = own_executor.submit(_slow_prewarm)
    reg._shutdown_prewarm_executor_at_exit()
    assert finished.is_set(), "the handler returned while the prewarm was still touching GPU objects"


def test_the_prewarm_exit_handler_gives_up_on_a_stuck_prewarm(own_executor, monkeypatch):
    """Waiting without a bound would let one stuck provider load hang the process exit."""
    release = threading.Event()
    stuck = own_executor.submit(release.wait, 30)
    reg._PREWARM_FUTURES["stuck"] = stuck
    monkeypatch.setattr(reg, "_PREWARM_EXIT_GRACE_S", 0.3)
    try:
        reg._shutdown_prewarm_executor_at_exit()
        # The state, not the clock: the handler returned while the prewarm was still running, which is what "gave up"
        # means here. A wall-clock bound would just measure this machine's scheduler.
        assert not stuck.done(), "the handler waited for a prewarm that never finishes"
    finally:
        release.set()
        stuck.result(timeout=30)


def test_the_heartbeat_is_joined_at_exit(monkeypatch):
    """An un-joined heartbeat can be mid-log when logging's own atexit closes the handlers."""
    stops = []

    class _HB:
        def stop(self, join=True, timeout=None):
            stops.append((join, timeout))

    monkeypatch.setattr(cd, "_HEARTBEAT", _HB())
    cd._atexit_handler()
    assert stops and stops[0][0] is True, "the exit handler must join the heartbeat, not fire and forget"
    assert stops[0][1] and stops[0][1] > 0, "and it must bound that join"
