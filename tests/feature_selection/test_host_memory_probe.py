"""``host_memory`` reports what ``psutil.virtual_memory()`` reports and honours a test double installed over it."""

from __future__ import annotations

import psutil

from mlframe.system import _ram_probe


def test_host_memory_agrees_with_psutil() -> None:
    """Total is the same number; available moves between the two reads, so it is compared within 5% of the total."""
    available, total = _ram_probe.host_memory()
    vm = psutil.virtual_memory()
    assert total == int(vm.total)
    assert abs(available - int(vm.available)) <= 0.05 * total


def test_host_memory_honours_a_patched_psutil(monkeypatch) -> None:
    """A fit under a simulated memory level (the low-memory tests replace ``psutil.virtual_memory``) reads the simulated numbers."""
    _ram_probe.host_memory()  # the shipped function is remembered on first use

    class _Vm:
        """A host with 3 of 10 bytes free."""

        available = 3
        total = 10

    monkeypatch.setattr(psutil, "virtual_memory", lambda: _Vm)
    assert _ram_probe.host_memory() == (3, 10)
    assert _ram_probe.available_ram_bytes() == 3
