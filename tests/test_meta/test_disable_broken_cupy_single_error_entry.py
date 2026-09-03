"""CORE_INFRA_MISC-5: _disable_broken_cupy's module docstring documents gpu_disable_errors as holding
'0 or 1 entries' per process, but the NVRTC-probe-failure branch appended two near-duplicate messages,
violating that invariant. Pins the fix by simulating a broken cupy probe and checking exactly one entry
is appended.
"""

from __future__ import annotations

import sys
import types

import mlframe


def test_disable_broken_cupy_appends_exactly_one_entry_on_probe_failure(monkeypatch):
    """A cupy whose reduction probe raises must add exactly one entry to gpu_disable_errors, not two."""
    fake_cupy = types.ModuleType("cupy")

    class _BrokenArray:
        """Stand-in for a cupy array whose reduction reaches NVRTC and raises."""

        def sum(self):
            """Return self so .item() below is the one that raises."""
            return self

        def item(self):
            """Simulate the NVRTC probe failure the guard is meant to catch."""
            raise RuntimeError("simulated NVRTC probe failure")

    fake_cupy.asarray = lambda *a, **k: _BrokenArray()  # type: ignore[attr-defined]
    fake_cupy.float32 = "float32"  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "cupy", fake_cupy)
    monkeypatch.delenv("MLFRAME_KEEP_BROKEN_CUPY", raising=False)
    monkeypatch.setattr(mlframe, "gpu_disable_errors", [])

    mlframe._disable_broken_cupy()

    assert len(mlframe.gpu_disable_errors) == 1
    assert "NVRTC probe failed" in mlframe.gpu_disable_errors[0]
    assert sys.modules["cupy"] is None
