"""Fourier frequency detection for several columns at once: one resident batch on the device when the GPU-resident mode is on, else the single-column detector in a loop."""

from __future__ import annotations

import logging
import os
from typing import Any, Sequence

import numpy as np

from .._ktc_fallback import host_device_sweep, rows_fallback

logger = logging.getLogger(__name__)

__all__ = ["detect_fourier_freqs_batch", "device_batch_enabled", "fourier_device_chosen"]

MIN_BATCH = 2  # a single job gains nothing from the batch kernels
DEVICE_FALLBACK_ROWS = 12_000  # measured crossover (GTX 1050 Ti, 4 and 16 columns); the tuner replaces it per host once a sweep has run
_SWEEP_ROWS = [5_000, 20_000, 100_000]
_SWEEP_COLUMNS = 8
_SWEEP_PARAMS: dict[str, Any] = dict(min_val_corr=0.15, min_rows=500, max_freqs=6)
_SALT = 1


def _sweep_inputs(dims: dict) -> tuple:
    """``_SWEEP_COLUMNS`` columns of ``dims["n_samples"]`` rows: a tone of frequency 3, 2 or 4.25 on every column but each fourth (pure noise), the standard 32-point grid."""
    rng = np.random.default_rng(0)
    n = int(dims["n_samples"])
    grid = [round(0.25 * k, 3) for k in range(1, 33)]
    jobs = []
    for i in range(_SWEEP_COLUMNS):
        z = rng.random(n)
        f = (3.0, 2.0, 4.25, 0.0)[i % 4]
        y = (np.sin(2 * np.pi * f * z) if f else 0.0) + 0.5 * rng.standard_normal(n)
        jobs.append((z, y, grid))
    return (jobs,)


def _as_matrix(found: "list[list[float]]") -> np.ndarray:
    """The detected lists as a fixed-width matrix (zero padded), so two backends are compared on count, order and value."""
    out = np.zeros((len(found), int(_SWEEP_PARAMS["max_freqs"])), dtype=np.float64)  # Intentional float64: comparison matrix
    for i, lst in enumerate(found):
        out[i, : len(lst)] = lst
    return out


def _host_single(z01: np.ndarray, y: np.ndarray, f_grid: Sequence[float]) -> list:
    """The host detector of one column (no device attempt)."""
    from ._orth_extra_basis_fe import _detect_fourier_freqs_host

    return _detect_fourier_freqs_host(z01, y, f_grid=f_grid, **_SWEEP_PARAMS)


def _sweep_host(jobs: list) -> np.ndarray:
    """The host loop, for the sweep."""
    return _as_matrix([_host_single(z, y, g) for z, y, g in jobs])


def _sweep_device(jobs: list) -> np.ndarray:
    """The device batch, for the sweep; a failed launch raises so the sweep does not record a host fallback as a device time."""
    from ._fourier_batch_gpu import detect_fourier_freqs_batch_gpu

    return _as_matrix(detect_fourier_freqs_batch_gpu(jobs, fourier_detect_max_n=200_000, single=_host_single, **_SWEEP_PARAMS))


_fallback_choice = rows_fallback(__name__)

try:
    from pyutilz.performance.kernel_tuning.registry import kernel_tuner

    _SPEC: Any = kernel_tuner(
        kernel_name="fe_fourier_batch_backend",
        variant_fns=(),  # the two backends are timed by the sweep callable
        tuner=host_device_sweep(__name__, _sweep_host, _sweep_device, rtol=1e-6, atol=1e-6),
        axes={"n_samples": list(_SWEEP_ROWS)},
        fallback=_fallback_choice,
        gpu_capable=True,
        salt=_SALT,
        cli_label="fe_fourier_batch_backend",
    )
except Exception as e:  # best-effort: without a tuner entry the row-count fallback decides
    logger.warning("kernel_tuner spec build for fe_fourier_batch_backend failed: %s", e)
    _SPEC = None


def fourier_device_chosen(n_rows: int) -> bool:
    """Whether jobs of ``n_rows`` rows run the device batch: forced by ``MLFRAME_CMI_GPU=1``, else the per-host kernel-tuning cache (fallback ``DEVICE_FALLBACK_ROWS``)."""
    if os.environ.get("MLFRAME_CMI_GPU", "") == "1":
        return True
    if _SPEC is None:
        return int(n_rows) >= DEVICE_FALLBACK_ROWS
    try:
        return str(_SPEC.choose(n_samples=int(n_rows))) == "device"
    except Exception as exc:
        logger.debug("fe_fourier_batch_backend: KTC choose() failed, using the row-count fallback: %s", exc)
        return int(n_rows) >= DEVICE_FALLBACK_ROWS


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


def device_batch_enabled() -> bool:
    """Whether :func:`detect_fourier_freqs_batch` would run the resident device batch (the GPU-resident mode and the fused switch are on)."""
    try:
        from .._gpu_strict_fe._entry import fe_gpu_strict_resident_enabled
        from .._fe_env_switch import fused_switch_enabled
        from ._fourier_detect_gpu_resident import _FUSED_ENV

        return bool(fe_gpu_strict_resident_enabled() and fused_switch_enabled(_FUSED_ENV))
    except ImportError:
        return False


def detect_fourier_freqs_batch(
    jobs: "Sequence[tuple[np.ndarray, np.ndarray, Sequence[float]]]", *, min_val_corr: float, min_rows: int = 800, max_freqs: int = 4
) -> "list[list[float]]":
    """Detected frequencies of each ``(z01, y, f_grid)`` job: the same lists the single-column detector (``_detect_fourier_freqs_for_col``) returns, one list per job."""
    from ._orth_extra_basis_fe import _detect_fourier_freqs_for_col

    # The single-column detector with this batch's parameters.
    single = lambda z01, y, f_grid: _detect_fourier_freqs_for_col(  # noqa: E731 - a one-call adapter, not a second implementation
        z01, y, f_grid=f_grid, min_val_corr=min_val_corr, min_rows=min_rows, max_freqs=max_freqs
    )

    jobs = list(jobs)
    if len(jobs) >= MIN_BATCH:
        try:
            from .._gpu_strict_fe._entry import fe_gpu_strict_resident_enabled

            if fe_gpu_strict_resident_enabled() and fourier_device_chosen(int(np.median([np.size(j[0]) for j in jobs]))):
                from .._fourier_detect_cap import get_fourier_detect_max_n
                from ._fourier_batch_gpu import detect_fourier_freqs_batch_gpu
                from .._fe_env_switch import fused_switch_enabled
                from ._fourier_detect_gpu_resident import _FUSED_ENV

                if fused_switch_enabled(_FUSED_ENV):
                    return detect_fourier_freqs_batch_gpu(
                        jobs, min_val_corr=min_val_corr, min_rows=min_rows, max_freqs=max_freqs, fourier_detect_max_n=get_fourier_detect_max_n(), single=single
                    )
        except _device_errors():
            pass  # a genuine device fault: the single-column path below (its own CPU fallback) takes over
        except ImportError:
            pass
    return [single(z01, y, f_grid) for z01, y, f_grid in jobs]
