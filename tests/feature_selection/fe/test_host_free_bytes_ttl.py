"""The host free-memory probe is read once per TTL window and re-read after it."""

from __future__ import annotations

import psutil

from mlframe.feature_selection.filters._orthogonal_univariate_fe import _streamed_gpu_mi as mod


def _reset(monkeypatch):
    """Empty the module cache so each test starts with no reading."""
    monkeypatch.setattr(mod, "_host_free_cache", (float("-inf"), 0))


def test_repeated_calls_inside_the_window_probe_once(monkeypatch):
    """Many path choices in one burst cost one ``virtual_memory`` read."""
    _reset(monkeypatch)
    calls = []

    class _Mem:
        """Memory reading stand-in."""

        available = 12345

    monkeypatch.setattr(psutil, "virtual_memory", lambda: calls.append(1) or _Mem())
    assert [mod._host_free_bytes() for _ in range(50)] == [12345] * 50
    assert len(calls) == 1


def test_a_call_after_the_window_probes_again(monkeypatch):
    """Once the TTL has passed the new reading replaces the cached one."""
    _reset(monkeypatch)
    values = iter([100, 200])

    class _Mem:
        """Memory reading stand-in."""

        def __init__(self, v):
            """Hold the available-bytes value."""
            self.available = v

    monkeypatch.setattr(psutil, "virtual_memory", lambda: _Mem(next(values)))
    clock = [0.0]
    monkeypatch.setattr(mod.time, "monotonic", lambda: clock[0])
    assert mod._host_free_bytes() == 100
    clock[0] = mod._HOST_FREE_TTL_SECONDS + 0.1
    assert mod._host_free_bytes() == 200
