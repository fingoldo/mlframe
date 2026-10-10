"""Fourier frequency detection for several columns at once: one resident batch on the device when the GPU-resident mode is on, else the single-column detector in a loop."""

from __future__ import annotations

import logging
from typing import Sequence

import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["detect_fourier_freqs_batch"]

MIN_BATCH = 2  # a single job gains nothing from the batch kernels


def _device_errors() -> tuple:
    """Exception types that mean a device or linear-algebra fault (the CPU path then takes over); a logic error is not among them and propagates."""
    errs: list = [np.linalg.LinAlgError]
    try:
        import cupy as cp

        errs += [cp.cuda.runtime.CUDARuntimeError, cp.cuda.memory.OutOfMemoryError]
        from cupy_backends.cuda.libs import cublas, cusolver

        errs += [getattr(cusolver, "CUSOLVERError", None), getattr(cublas, "CUBLASError", None)]
    except Exception as e:  # nosec B110 - optional dependency import guard
        logger.debug("device error-type probe failed: %s", e)
    return tuple(e for e in errs if isinstance(e, type) and issubclass(e, BaseException))


def detect_fourier_freqs_batch(
    jobs: "Sequence[tuple[np.ndarray, np.ndarray, Sequence[float]]]", *, min_val_corr: float, min_rows: int = 800, max_freqs: int = 4
) -> "list[list[float]]":
    """Detected frequencies of each ``(z01, y, f_grid)`` job: the same lists the single-column detector (``_detect_fourier_freqs_for_col``) returns, one list per job."""
    from ._orth_extra_basis_fe import _detect_fourier_freqs_for_col

    def single(z01, y, f_grid) -> list:
        """The single-column detector with this batch's parameters."""
        return _detect_fourier_freqs_for_col(z01, y, f_grid=f_grid, min_val_corr=min_val_corr, min_rows=min_rows, max_freqs=max_freqs)

    jobs = list(jobs)
    if len(jobs) >= MIN_BATCH:
        try:
            from .._gpu_strict_fe._entry import fe_gpu_strict_resident_enabled

            if fe_gpu_strict_resident_enabled():
                from .._fourier_detect_cap import get_fourier_detect_max_n
                from ._fourier_batch_gpu import detect_fourier_freqs_batch_gpu
                from ._fourier_detect_gpu_resident import _fused_enabled

                if _fused_enabled():
                    return detect_fourier_freqs_batch_gpu(
                        jobs, min_val_corr=min_val_corr, min_rows=min_rows, max_freqs=max_freqs, fourier_detect_max_n=get_fourier_detect_max_n(), single=single
                    )
        except _device_errors():
            pass  # a genuine device fault: the single-column path below (its own CPU fallback) takes over
        except ImportError:
            pass
    return [single(z01, y, f_grid) for z01, y, f_grid in jobs]
