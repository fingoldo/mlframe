"""``typed_module``: one compiled kernel module per (source, C type) in a process, with the ``{DT}`` placeholder replaced by the type (the cache the shuffle and the family-gather kernels share)."""

from __future__ import annotations

import threading

import pytest

from mlframe.feature_selection.filters import _batch_mi_noise_gate_bits as bits


class _FakeCupy:
    """A stand-in for cupy whose ``RawModule`` records every compilation."""

    def __init__(self) -> None:
        """No compilations yet."""
        self.compiled: "list[str]" = []

    def RawModule(self, code: str) -> "tuple[str]":
        """Record the source and hand back a token that stands for the module."""
        self.compiled.append(code)
        return (code,)


@pytest.fixture(autouse=True)
def _fresh_cache(monkeypatch):
    """Every test starts from an empty cache."""
    monkeypatch.setattr(bits, "_TYPED_MODULES", {})


def test_the_placeholder_is_replaced_by_the_c_type_and_the_module_is_built_once():
    """Two requests for one source and type compile once and return the same module."""
    cp = _FakeCupy()
    first = bits.typed_module(cp, "{DT}* a; {DT}* b;", "short")
    second = bits.typed_module(cp, "{DT}* a; {DT}* b;", "short")
    assert first is second
    assert cp.compiled == ["short* a; short* b;"]


def test_another_type_or_another_source_is_another_module():
    """The key is the pair (source, type): a different type and a different source each compile again."""
    cp = _FakeCupy()
    a = bits.typed_module(cp, "{DT} x;", "int")
    b = bits.typed_module(cp, "{DT} x;", "long long")
    c = bits.typed_module(cp, "{DT} y;", "int")
    assert len({id(a), id(b), id(c)}) == 3
    assert cp.compiled == ["int x;", "long long x;", "int y;"]


def test_concurrent_first_requests_compile_once():
    """Threads asking for the same module at the same time share one compilation."""
    cp = _FakeCupy()
    got: "list[object]" = []
    barrier = threading.Barrier(6)

    def ask() -> None:
        """Wait for the others, then request the module."""
        barrier.wait()
        got.append(bits.typed_module(cp, "{DT} z;", "signed char"))

    threads = [threading.Thread(target=ask) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(got) == 6 and all(m is got[0] for m in got)
    assert cp.compiled == ["signed char z;"]
