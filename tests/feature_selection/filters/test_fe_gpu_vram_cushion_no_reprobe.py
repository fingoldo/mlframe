"""``fe_gpu_has_vram_cushion`` accepts an already-probed ``(free_b, total_b)`` so a hot gate does not call
``memGetInfo`` twice per dispatch (``_cmi_cuda._should_use_cuda`` queries it for its own relative cap immediately
above the call).

Three contracts: (1) the decision is identical whether the pair is supplied or probed internally, (2) a partial
pair falls back to probing rather than mixing a stale value with a live one, (3) with no cupy the function is
permissive, so a non-GPU host is never blocked by a probe it cannot make.

Contract (3) used to be written as a bare ``is True`` on the assumption that the dev box had no cupy. That made
the assertion a statement about the machine rather than about the function: it passed on CI, where cupy is
genuinely absent, and failed on any developer box with a GPU -- where the internal probe correctly returns False
on a near-full card. The absence of cupy is now SIMULATED, so the test asserts the contract on every host.
"""

import sys

import pytest

from mlframe.feature_selection.filters._fe_gpu_vram import _cushion_bytes, fe_gpu_has_vram_cushion


@pytest.fixture
def without_cupy(monkeypatch):
    """Make ``import cupy`` raise ImportError for the duration of one test.

    A ``None`` entry in ``sys.modules`` is the documented way to do this: the import machinery treats it as a
    failed import rather than as a cached module, so the function's own ``except ImportError`` branch runs.
    """
    monkeypatch.setitem(sys.modules, "cupy", None)


def test_cushion_is_permissive_when_cupy_is_absent(without_cupy):
    """A host that cannot probe must never be blocked by the cushion -- the caller's other gates decide."""
    assert fe_gpu_has_vram_cushion(10**9) is True


def test_an_explicit_probe_pair_is_honoured_even_without_cupy(without_cupy):
    """Supplying the pair must not trigger a cupy import at all, so the decision stands on the given numbers."""
    total_b = 4 * 1024**3
    assert fe_gpu_has_vram_cushion(10**9, free_b=2 * 1024**3, total_b=total_b) is True
    assert fe_gpu_has_vram_cushion(10**9, free_b=_cushion_bytes(total_b), total_b=total_b) is False


def test_cushion_decision_matches_manual_formula_when_probe_supplied():
    """When the pair IS supplied the decision must equal the documented formula, on any host.

    Independent of cupy by construction: the supplied-pair branch never probes.
    """
    total_b = 4 * 1024 * 1024 * 1024
    cushion = _cushion_bytes(total_b)
    free_b_ok = cushion + 10**7  # just above the cushion floor
    free_b_bad = max(0, cushion - 10**7)  # just below

    assert fe_gpu_has_vram_cushion(0, free_b=free_b_ok, total_b=total_b) is True
    assert fe_gpu_has_vram_cushion(0, free_b=free_b_bad, total_b=total_b) is False


@pytest.mark.parametrize(
    "partial",
    [
        {"free_b": 2 * 1024**3, "total_b": None},
        {"free_b": None, "total_b": 4 * 1024**3},
    ],
)
def test_a_partial_pair_falls_back_to_probing(partial):
    """Half a probe is worse than none: mixing one caller-supplied number with one live one describes no card.

    Asserted as "same answer as probing" rather than as a fixed True, because what the probe returns depends on
    the host -- which is exactly the assumption that made this file machine-specific before.
    """
    assert fe_gpu_has_vram_cushion(10**9, **partial) == fe_gpu_has_vram_cushion(10**9)


def test_a_partial_pair_ignores_the_supplied_half(without_cupy):
    """Pinned with the probe forced permissive, so a False here could only come from the stale half leaking in."""
    assert fe_gpu_has_vram_cushion(10**12, free_b=0, total_b=None) is True
