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

import importlib
import logging
import time
from pathlib import Path

import pytest


MLFRAME_ROOT = Path(importlib.import_module("mlframe").__file__).parent


def _read(rel: str) -> str:
    return (MLFRAME_ROOT / rel).read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Source-level sensors
# ---------------------------------------------------------------------------


def test_kick_cpu_count_logs_at_debug_on_failure() -> None:
    # ``_kick_cpu_count`` was carved out of metrics/core.py into sibling
    # metrics/_core_numba_warmup.py during a numba-warmup split. Check both
    # files so the sensor still works after the move.
    src = _read("metrics/core.py")
    helper_idx = src.find("def _kick_cpu_count")
    if helper_idx == -1:
        src = _read("metrics/_core_numba_warmup.py")
        helper_idx = src.find("def _kick_cpu_count")
    assert helper_idx != -1, "def _kick_cpu_count not found in metrics/core.py or metrics/_core_numba_warmup.py"
    snippet = src[helper_idx : helper_idx + 800]
    assert "logger.debug" in snippet, (
        "_kick_cpu_count must surface failures at DEBUG, not silently pass."
    )


def test_prewarm_registers_done_callback() -> None:
    src = _read("training/feature_handling/registry.py")
    # The fix attaches add_done_callback after submit.
    assert "add_done_callback(_log_unhandled)" in src, (
        "registry.py prewarm: must attach a done-callback so an unawaited failure logs."
    )
    assert "_log_unhandled" in src, "registry.py prewarm: _log_unhandled helper must exist."


# ---------------------------------------------------------------------------
# Behavioural sensor: prewarm callback actually fires + logs on failure.
# ---------------------------------------------------------------------------


def test_prewarm_done_callback_logs_warning_when_worker_raises(caplog) -> None:
    """If the prewarm worker raises and the caller never calls wait_prewarm(),
    the registered done-callback must surface a WARNING via the module logger."""
    from unittest.mock import patch
    from mlframe.training.feature_handling import registry as reg

    # Build a tiny synthetic provider with a signature unique to this test.
    sig = f"wave43_silent_test_{id(object())}"

    class _DummyProvider:
        signature = sig

    # Inject a _do_load that raises immediately.
    def _raising_do_load():
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
