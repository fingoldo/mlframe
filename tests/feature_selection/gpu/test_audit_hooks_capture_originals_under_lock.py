"""The transfer audits capture the original cupy entry points only once they hold the audit lock, so overlapping audits on two threads restore the true originals."""

from __future__ import annotations

import sys
import threading
import types

import pytest

from mlframe.feature_selection.filters._gpu_strict_fe import duplicate_transfer_audit, residency_audit


def _fake_cupy() -> types.ModuleType:
    """A cupy stand-in exposing exactly the attributes the audits patch."""

    class _Arr:
        """Stand-in for ``cupy.ndarray``."""

        def get(self, *a, **k):
            """Pretend D2H copy."""
            return None

        def item(self):
            """Pretend scalar read."""
            return 0

        def __float__(self):
            """Pretend scalar read."""
            return 0.0

        def __int__(self):
            """Pretend scalar read."""
            return 0

        def __bool__(self):
            """Pretend scalar read."""
            return False

    mod = types.ModuleType("cupy")
    mod.asarray = lambda obj, *a, **k: obj  # type: ignore[attr-defined]
    mod.asnumpy = lambda obj, *a, **k: obj  # type: ignore[attr-defined]
    mod.ndarray = _Arr  # type: ignore[attr-defined]
    return mod


@pytest.mark.parametrize("audit", [residency_audit, duplicate_transfer_audit])
def test_overlapping_audits_on_two_threads_restore_the_original_cupy_functions(monkeypatch, audit):
    """A second audit entering while the first is active must not capture the first one's wrapper as its "original"."""
    fake = _fake_cupy()
    monkeypatch.setitem(sys.modules, "cupy", fake)
    orig_asnumpy, orig_get = fake.asnumpy, fake.ndarray.get

    second_started = threading.Event()

    def _second() -> None:
        """Enter an audit while the main thread's audit is still open."""
        second_started.set()
        with audit():
            pass

    with audit():
        t = threading.Thread(target=_second)
        t.start()
        second_started.wait(5)
        t.join(0.3)
    t.join(10)

    assert not t.is_alive()
    assert fake.asnumpy is orig_asnumpy
    assert fake.ndarray.get is orig_get
