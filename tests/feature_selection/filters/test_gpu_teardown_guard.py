"""Regression: the cupy/CUDA teardown-race guard must not crash from within its own hooks.

During interpreter finalization Python clears every global name in a module's own namespace,
including an imported ``sys`` -- if a hook installed onto ``sys.excepthook``/``sys.unraisablehook``
re-references the module-global name ``sys`` internally (rather than a pre-bound reference
captured before finalization began), the hook itself raises ``AttributeError: 'NoneType' object
has no attribute 'is_finalizing'`` instead of quietly swallowing the cosmetic cupy teardown error
it was installed to suppress -- observed live as an "Error in sys.excepthook" spam loop.
"""

from __future__ import annotations

import sys
import types

from mlframe.feature_selection.filters import _gpu_teardown_guard as guard


class _FakeCUDADriverError(Exception):
    """Stand-in for cupy_backends' CUDADriverError (mirrors its class name + message shape)."""


_FakeCUDADriverError.__name__ = "CUDADriverError"


def test_is_cuda_teardown_error_matches_illegal_address():
    """A CUDADriverError/CUDARuntimeError carrying an illegal-address message is recognized."""
    exc = _FakeCUDADriverError("CUDA_ERROR_ILLEGAL_ADDRESS: an illegal memory access was encountered")
    assert guard._is_cuda_teardown_error(exc)


def test_is_cuda_teardown_error_rejects_unrelated_exception():
    """A non-CUDA exception type must never be classified as the teardown race."""
    assert not guard._is_cuda_teardown_error(ValueError("unrelated"))


def test_is_cuda_teardown_error_rejects_none():
    """A missing exc_value (None) must never be classified as the teardown race."""
    assert not guard._is_cuda_teardown_error(None)


def test_excepthook_survives_sys_name_cleared_during_finalization(monkeypatch):
    """The exact bug: with the module-global name ``sys`` set to None (simulating interpreter
    finalization clearing this module's namespace) and the interpreter genuinely finalizing, the
    hook must swallow the known cupy teardown error without raising AttributeError itself."""
    monkeypatch.setattr(guard, "_is_finalizing", lambda: True)
    monkeypatch.setattr(guard, "sys", None)
    exc = _FakeCUDADriverError("CUDA_ERROR_ILLEGAL_ADDRESS: an illegal memory access was encountered")
    guard._prev_excepthook = None
    guard._excepthook(type(exc), exc, None)  # must not raise


def test_unraisablehook_survives_sys_name_cleared_during_finalization(monkeypatch):
    """Same bug/fix as the excepthook test above, for the sys.unraisablehook entry point."""
    monkeypatch.setattr(guard, "_is_finalizing", lambda: True)
    monkeypatch.setattr(guard, "sys", None)
    exc = _FakeCUDADriverError("CUDA_ERROR_ILLEGAL_ADDRESS: an illegal memory access was encountered")
    unraisable = types.SimpleNamespace(exc_value=exc)
    guard._prev_unraisablehook = None
    guard._unraisablehook(unraisable)  # must not raise


def test_excepthook_chains_to_previous_hook_for_unrelated_exception(monkeypatch):
    """A non-teardown exception must still reach the previously-installed excepthook."""
    calls = []
    monkeypatch.setattr(guard, "_is_finalizing", lambda: False)
    monkeypatch.setattr(guard, "_prev_excepthook", lambda *a: calls.append(a))
    exc = ValueError("boom")
    guard._excepthook(type(exc), exc, None)
    assert len(calls) == 1


def test_install_cuda_teardown_guard_is_idempotent():
    """Calling install twice must not double-chain the hooks."""
    before_installed = guard._installed
    before_excepthook = sys.excepthook
    before_unraisablehook = getattr(sys, "unraisablehook", None)
    before_prev_excepthook = guard._prev_excepthook
    before_prev_unraisablehook = guard._prev_unraisablehook
    try:
        guard._installed = False
        guard._prev_excepthook = None
        guard._prev_unraisablehook = None
        guard.install_cuda_teardown_guard()
        first_hook = sys.excepthook
        guard.install_cuda_teardown_guard()
        assert sys.excepthook is first_hook
    finally:
        sys.excepthook = before_excepthook
        if before_unraisablehook is not None:
            sys.unraisablehook = before_unraisablehook
        guard._installed = before_installed
        guard._prev_excepthook = before_prev_excepthook
        guard._prev_unraisablehook = before_prev_unraisablehook
