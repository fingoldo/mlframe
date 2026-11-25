"""The typed.Dict warm-up compiles once even when a fit and the import-time thread ask for it together, and the import-time start can be switched off."""

import threading
import time

from mlframe.feature_selection.filters import _numba_warmup as W


def test_two_callers_compile_once_and_both_wait_for_it(monkeypatch):
    """The second caller blocks until the first finished compiling: after both returned the suite is compiled, by exactly one compile."""
    calls = []

    def slow_compile():
        """A stand-in compile that takes a moment."""
        calls.append(time.perf_counter())
        time.sleep(0.3)

    monkeypatch.setattr(W, "_warmup_done", False)
    monkeypatch.setattr(W, "_compile_typed_dict_suite", slow_compile)
    monkeypatch.delenv("MLFRAME_SKIP_NUMBA_WARMUP", raising=False)
    waited = []

    def caller():
        """One caller: how long it waited."""
        t = time.perf_counter()
        W.warmup_typed_dict()
        waited.append(time.perf_counter() - t)

    threads = [threading.Thread(target=caller) for _ in range(2)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert len(calls) == 1 and W._warmup_done is True
    assert min(waited) >= 0.25, waited  # neither returned while the compile was still running


def test_import_time_start_can_be_switched_off_and_runs_the_warmup_otherwise(monkeypatch):
    """With MLFRAME_IMPORT_PREWARM=0 no thread is started; by default a daemon thread runs the warm-up."""
    ran = []
    monkeypatch.setattr(W, "warmup_typed_dict", lambda: ran.append(threading.current_thread().name))
    monkeypatch.setenv("MLFRAME_IMPORT_PREWARM", "0")
    W.start_import_time_warmup()
    # A started thread is visible to enumerate() as soon as start() returns, so no wait is needed to see that none was started.
    assert not any(t.name == "mlframe-typed-dict-warmup" for t in threading.enumerate())
    assert ran == []
    monkeypatch.setenv("MLFRAME_IMPORT_PREWARM", "1")
    monkeypatch.delenv("MLFRAME_SKIP_NUMBA_WARMUP", raising=False)
    W.start_import_time_warmup()
    deadline = time.perf_counter() + 5
    while not ran and time.perf_counter() < deadline:
        time.sleep(0.02)
    assert ran == ["mlframe-typed-dict-warmup"]
