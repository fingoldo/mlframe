"""One-time cuBLAS health probe for cupy matmul paths.

A cupy/cuBLAS/driver combination can be installed that imports fine and reports a device, yet fails every
float32 matmul with ``CUBLAS_STATUS_INVALID_VALUE``. Each GPU matmul path asks this probe once per process and
demotes to its numpy implementation when the probe fails, logging the library versions so the cause is visible.
"""

from __future__ import annotations

import logging
import threading

from mlframe.utils.log_throttle import log_throttle

logger = logging.getLogger(__name__)

_LOCK = threading.Lock()
_RESULT: tuple[bool, str] | None = None


def _matmul_demote_errors() -> tuple[type[BaseException], ...]:
    """Exceptions that legitimately mean "this cupy/cuBLAS matmul does not work here": cupy missing, or a cupy/cuBLAS/CUDA error."""
    excs: tuple[type[BaseException], ...] = (ImportError,)
    try:
        import cupy as cp
        from cupy_backends.cuda.libs import cublas

        excs = (*excs, cp.cuda.runtime.CUDARuntimeError, cublas.CUBLASError)
    except Exception as e:  # nosec B110 - a missing cupy is already covered by ImportError above; log and keep the narrow tuple
        logger.debug("suppressed: %s", e)
    return excs


def _library_versions() -> str:
    """cupy and cuBLAS versions for the diagnostic line, or ``unknown`` when cupy is not importable."""
    try:
        import cupy as cp

        from importlib.metadata import version as _pkg_version

        cublas = _pkg_version("nvidia-cublas-cu12") if _has_pkg("nvidia-cublas-cu12") else "not-installed"
        return f"cupy={cp.__version__} cuda_runtime={cp.cuda.runtime.runtimeGetVersion()} nvidia-cublas-cu12={cublas}"
    except Exception as e:  # nosec B110 - diagnostics only; the probe result itself is what callers act on
        logger.debug("suppressed: %s", e)
        return "unknown"


def _has_pkg(name: str) -> bool:
    """Whether a distribution named ``name`` is installed."""
    from importlib.metadata import PackageNotFoundError, version as _pkg_version

    try:
        _pkg_version(name)
        return True
    except PackageNotFoundError:
        return False


def cublas_matmul_ok() -> bool:
    """True when a float32 cupy matmul completes correctly on the current device; probed once per process."""
    global _RESULT
    with _LOCK:
        if _RESULT is None:
            _RESULT = _probe()
    ok, reason = _RESULT
    if not ok:
        log_throttle(
            logger, "cublas_probe_failed", logging.WARNING,
            "cupy matmul probe failed (%s); GPU matmul paths demote to numpy for this process. Libraries: %s",
            reason, _library_versions(),
        )
    return ok


def _probe() -> tuple[bool, str]:
    """Run a 4x3 @ 3x2 float32 matmul on the device and check one known entry."""
    try:
        import cupy as cp

        a = cp.ones((4, 3), dtype=cp.float32)
        b = cp.ones((3, 2), dtype=cp.float32)
        c = a @ b
        cp.cuda.Stream.null.synchronize()
        value = float(c[0, 0])
        if value != 3.0:
            return False, f"matmul returned {value}, expected 3.0"
        return True, "ok"
    except _matmul_demote_errors() as exc:
        return False, f"{type(exc).__name__}: {exc}"


def reset_cublas_probe() -> None:
    """Forget the cached probe result (tests use this to re-probe under a monkeypatched backend)."""
    global _RESULT
    with _LOCK:
        _RESULT = None


__all__ = ["cublas_matmul_ok", "reset_cublas_probe"]
