"""Cluster 31 regression: a kernel sweep timeout logs what actually happened, not a fallback that never ran (audit 2026-11-20, finding 31-F10)."""

import logging
import subprocess
from types import SimpleNamespace

from mlframe.system.kernel_tuning_cache import _ensure


def test_timeout_is_reported_as_timeout_not_fallback(monkeypatch, caplog):
    """A sweep that times out returns the timeout status and the log says it timed out (31-F10)."""

    def _timeout(*a, **k):
        """Simulate a sweep exceeding its per-kernel limit."""
        raise subprocess.TimeoutExpired(cmd="sweep", timeout=1)

    monkeypatch.setattr(_ensure.subprocess, "run", _timeout)
    spec = SimpleNamespace(kernel_name="dummy_kernel")
    with caplog.at_level(logging.WARNING, logger=_ensure.logger.name):
        status, _detail = _ensure._run_sweep(spec, timeout_s=1)
    assert status == "timeout"
    messages = [r.getMessage() for r in caplog.records]
    assert any("timed out" in m for m in messages)
    assert not any("fell back" in m for m in messages)
