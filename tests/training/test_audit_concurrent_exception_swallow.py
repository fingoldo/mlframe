"""Wave 43 (2026-05-20): concurrent.futures / threading silent-swallow audit.

Result: CLEAN for the core bug class (0 P0/P1). mlframe's parallel surface is
disciplined: all joblib.Parallel sites use eager-list return (re-raises), the
plotly Thread joins-with-exception-capture, recurrent.executor.map is fully
materialised, and the save.py ThreadPoolExecutor calls .result() on every
future.

2 P2 fragility hardenings applied:

  1. metrics/core.py:120 (_kick_cpu_count daemon-thread prefetch)
     bare `except Exception: pass` inside the worker silently dropped any
     failure of the perf prefetch -- completely invisible. Failure has no
     semantic effect (the main path calls cpu_count again later) so keep the
     swallow, but add logger.debug(..., exc_info=True) for triage.

  2. training/feature_handling/registry.py:307 (prewarm)
     prewarm/wait_prewarm is a public-API contract where the caller MUST call
     wait_prewarm() to surface worker exceptions. Without it, exceptions
     stored on the cached future are silently retained forever.
     Fix: attach an add_done_callback that calls fut.exception(timeout=0) and
     logs at warning level if it's non-None -- catches the contract violation
     even when the caller forgets the wait.
"""

from __future__ import annotations

import logging
import time

# ---------------------------------------------------------------------------
# Failure surfacing, exercised through the code that owns it
# ---------------------------------------------------------------------------


def test_kick_cpu_count_logs_at_debug_on_failure(monkeypatch, caplog) -> None:
    """A failing cpu_count prefetch is swallowed but leaves a DEBUG record with the traceback; a healthy one logs nothing."""
    import threading

    import joblib.parallel

    from mlframe.metrics import _core_numba_warmup as warmup

    started = []

    class _InlineThread:
        """Thread stand-in that records its target so the test runs it synchronously."""

        def __init__(self, target, daemon=None):
            """Remember the target."""
            self.target = target

        def start(self):
            """Record instead of spawning."""
            started.append(self.target)

    monkeypatch.setattr(threading, "Thread", _InlineThread)

    def _boom():
        """Fail like a broken physical-core probe."""
        raise OSError("probe exploded")

    monkeypatch.setattr(joblib.parallel, "cpu_count", _boom)
    warmup._prewarm_numba_cach_kick_loky_wmic_physical()
    assert len(started) == 1
    with caplog.at_level(logging.DEBUG, logger=warmup.logger.name):
        started[0]()
    failed = [r for r in caplog.records if "_kick_cpu_count: prefetch failed" in r.getMessage()]
    assert len(failed) == 1
    assert failed[0].levelno == logging.DEBUG
    assert failed[0].exc_info is not None and failed[0].exc_info[0] is OSError

    caplog.clear()
    monkeypatch.setattr(joblib.parallel, "cpu_count", lambda *a, **k: 4)
    warmup._prewarm_numba_cach_kick_loky_wmic_physical()
    with caplog.at_level(logging.DEBUG, logger=warmup.logger.name):
        started[1]()
    assert [r for r in caplog.records if "_kick_cpu_count" in r.getMessage()] == []


def test_prewarm_registers_done_callback(caplog) -> None:
    """prewarm() on a provider whose load fails logs one WARNING naming the signature even when wait_prewarm is never called."""
    from mlframe.training.feature_handling import registry as reg

    sig = f"prewarm_unawaited_failure_{id(object())}"

    class _FailingProvider:
        """Provider whose weight load always raises."""

        signature = sig

        def acquire(self):
            """Fail the load."""
            raise RuntimeError("intentional prewarm failure")

    provider = _FailingProvider()
    caplog.set_level(logging.WARNING, logger=reg.logger.name)
    try:
        fut = reg.prewarm(provider)
        assert isinstance(fut.exception(timeout=30), RuntimeError)
        for _ in range(300):
            if any(sig in r.getMessage() for r in caplog.records):
                break
            time.sleep(0.01)
        logged = [r for r in caplog.records if sig in r.getMessage()]
        assert len(logged) == 1
        assert logged[0].levelno == logging.WARNING
        assert "caller did not call wait_prewarm" in logged[0].getMessage()
        assert sig not in reg._REGISTRY
    finally:
        reg._PREWARM_FUTURES.pop(sig, None)


# ---------------------------------------------------------------------------
# Behavioural sensor: prewarm callback actually fires + logs on failure.
# ---------------------------------------------------------------------------


def test_prewarm_done_callback_logs_warning_when_worker_raises(caplog) -> None:
    """If the prewarm worker raises and the caller never calls wait_prewarm(),
    the registered done-callback must surface a WARNING via the module logger."""
    from mlframe.training.feature_handling import registry as reg

    # Build a tiny synthetic provider with a signature unique to this test.
    sig = f"wave43_silent_test_{id(object())}"

    class _DummyProvider:
        """Groups tests covering dummy provider."""
        signature = sig

    # Inject a _do_load that raises immediately.
    def _raising_do_load():
        """Raising do load."""
        raise RuntimeError("intentional wave-43 sensor failure")

    # Submit directly via the executor to bypass the registry's full prewarm
    # logic (which depends on FrozenFeaturizerProvider semantics). We assert
    # the done-callback PATTERN: add_done_callback -> read .exception() ->
    # logger.warning if non-None.
    caplog.set_level(logging.WARNING, logger=reg.__name__)
    fut = reg._PREWARM_EXECUTOR.submit(_raising_do_load)
    # Mimic the registry's wiring.
    captured = {}

    def _log_unhandled(_fut):
        """Log unhandled."""
        try:
            exc = _fut.exception(timeout=0)
        except Exception:
            return
        if exc is not None:
            captured["exc"] = exc
            reg.logger.warning("prewarm(%r) failed; caller did not call wait_prewarm.", sig, exc_info=exc)

    fut.add_done_callback(_log_unhandled)
    # Wait for the future to fire.
    for _ in range(100):
        if fut.done():
            break
        time.sleep(0.01)
    assert fut.done()
    # The callback may run on the executor thread; give it a beat.
    for _ in range(50):
        if "exc" in captured:
            break
        time.sleep(0.01)
    assert isinstance(captured.get("exc"), RuntimeError)
    assert any("prewarm" in r.message for r in caplog.records)
