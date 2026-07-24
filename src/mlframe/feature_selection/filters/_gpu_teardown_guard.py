"""Quiet the intermittent cupy<->numba CUDA teardown race at interpreter shutdown (2026-06-29).

When the GPU FE path runs, cupy ``RawModule`` objects coexist with numba.cuda's context. At interpreter
finalization the unload ordering of the two libraries' C-level destructors is not coordinated, so cupy's
``Module.__dealloc__`` (``cuModuleUnload``) occasionally fires AFTER the CUDA context it belongs to has already
been torn down -> a ``cudaErrorIllegalAddress`` surfaced through ``sys.unraisablehook`` / ``sys.excepthook``.

This is COSMETIC: it happens only during ``sys.is_finalizing()`` (after every fit has returned and every result
has been produced); repeated clean runs show the device context alive throughout the fit and the FE results
correct (selection-equivalent). The existing ``gpu._GpuBufferPool.free`` note already documents that REORDERING
the teardown (freeing pooled cupy buffers at atexit alongside numba) risks heap corruption (0xc0000374), so the
fix here deliberately does NOT touch teardown ordering. It installs a chaining ``sys.unraisablehook`` /
``sys.excepthook`` that swallows ONLY this exact error AND ONLY while the interpreter is finalizing -- any
illegal-access during a real fit (not finalizing) propagates unchanged, so a genuine OOB is never masked.
"""
from __future__ import annotations

import sys
import threading

# GPU_INFRA_C-2 fix (mrmr_audit_2026-07-22): install_cuda_teardown_guard's idempotency check used to be a
# bare unlocked `if _installed: return` -- two threads racing through it before either set _installed=True
# could let the second thread's `_prev_unraisablehook = getattr(sys, "unraisablehook", None)` capture the
# FIRST thread's own `_unraisablehook` wrapper (not the true original), causing infinite self-recursion the
# next time an unrelated exception fires. This lock makes the whole install atomic (still only ever runs once).
_install_lock = threading.Lock()
_installed = False
_prev_unraisablehook = None
_prev_excepthook = None
# Captured now (while ``sys`` is guaranteed live) rather than referencing the module-global name
# ``sys`` inside the hooks below: during interpreter finalization Python clears every global name
# in THIS module's own namespace, including ``sys`` itself, before these hooks necessarily stop
# firing -- a bare ``sys.is_finalizing()`` call from inside the hook then raises
# ``AttributeError: 'NoneType' object has no attribute 'is_finalizing'``. A pre-bound method
# reference has no such dependency on the name resolving again later.
_is_finalizing = sys.is_finalizing


def _is_cuda_teardown_error(exc) -> bool:
    """True only for the cupy/CUDA illegal-address raised from a teardown destructor."""
    if exc is None:
        return False
    if type(exc).__name__ not in ("CUDARuntimeError", "CUDADriverError"):
        return False
    msg = str(exc).lower()
    return "illegal memory access" in msg or "illegaladdress" in msg


def _unraisablehook(unraisable):
    """Chained ``sys.unraisablehook``: swallows the cosmetic cupy/CUDA teardown-race error during interpreter finalization, chains to the previously-installed hook for everything else."""
    exc = getattr(unraisable, "exc_value", None)
    # Swallow the known cupy<->numba teardown race ONLY during interpreter finalization; never mid-fit.
    if _is_finalizing() and _is_cuda_teardown_error(exc):
        return
    if _prev_unraisablehook is not None:
        _prev_unraisablehook(unraisable)


def _excepthook(exc_type, exc_value, exc_tb):
    """Chained ``sys.excepthook`` mirroring ``_unraisablehook`` for the synchronous exception path."""
    if _is_finalizing() and _is_cuda_teardown_error(exc_value):
        return
    if _prev_excepthook is not None:
        _prev_excepthook(exc_type, exc_value, exc_tb)


def install_cuda_teardown_guard() -> None:
    """Idempotently chain the teardown-only illegal-address suppressor onto the unraisable/except hooks.
    Thread-safe: the whole check-then-install sequence runs under a lock (GPU_INFRA_C-2 fix)."""
    global _installed, _prev_unraisablehook, _prev_excepthook
    with _install_lock:
        if _installed:
            return
        _prev_unraisablehook = getattr(sys, "unraisablehook", None)
        _prev_excepthook = getattr(sys, "excepthook", None)
        if _prev_unraisablehook is not None:
            sys.unraisablehook = _unraisablehook
        sys.excepthook = _excepthook
        _installed = True
