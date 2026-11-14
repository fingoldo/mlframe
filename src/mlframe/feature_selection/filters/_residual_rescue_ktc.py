"""Host-versus-device choices of the residual rescue: the column scoring (kernel-tuning entry ``residual_rescue_score_backend``) and the bin codes of the pool (``residual_rescue_codes_backend``).

``_residual_rescue_fe._score_columns`` (numba, one column per thread) and ``_residual_rescue_gpu.score_columns_gpu`` (one block per column) compute the same ``(t, gain)`` per pool column
(equal to 1e-14 on the development card). The host loop is the one that runs first; the device pays an upload of the integer codes of the pool, so it wins only when the work ``n * p`` is large
or when a host is much slower than the card. The crossover is swept per host by the work ``n * p``; until a sweep has run on a host the floor below decides.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from ._ktc_fallback import host_device_sweep, rows_fallback, selection_view

logger = logging.getLogger(__name__)

__all__ = ["score_columns_dispatch", "score_device_chosen", "codes_matrix_dispatch", "codes_device_chosen"]

DEVICE_FALLBACK_ROWS = 200_000_000  # column-rows (n * p) from which the device scores the pool before a sweep has run (the host loop won at every size swept, up to 25.6 million)
_SWEEP_N = 25_000
_SWEEP_ROWS = [_SWEEP_N * k for k in (4, 16, 64, 256, 1024)]  # n_samples is the work n * p at 25000 rows
_SWEEP_BINS = 16
_SALT = 1


def _sweep_inputs(dims: dict) -> tuple:
    """``(Q, resid, even)`` for a work of ``dims["n_samples"]`` column-rows: ``p`` integer-coded columns over 25000 rows and a residual that depends on the first column."""
    rng = np.random.default_rng(0)
    p = max(1, int(dims["n_samples"]) // _SWEEP_N)
    Q = rng.integers(0, _SWEEP_BINS, (p, _SWEEP_N)).astype(np.int64)
    resid = rng.standard_normal(_SWEEP_N) + 0.3 * (Q[0] % 3)
    return (Q, resid, np.arange(_SWEEP_N) % 2 == 0)


def _sweep_host(Q: np.ndarray, resid: np.ndarray, even: np.ndarray) -> np.ndarray:
    """The numba loop, as the comparison vector of the sweep: how many columns were scored and which are at the top by the t statistic."""
    from ._residual_rescue_fe import SHRINK, _score_columns

    t, _gain = _score_columns(Q, resid, even, _SWEEP_BINS, SHRINK)
    return selection_view(t)


def _sweep_device(Q: np.ndarray, resid: np.ndarray, even: np.ndarray) -> np.ndarray:
    """The device kernel; a failed launch raises, so the sweep never records a host fallback as a device time."""
    from ._residual_rescue_fe import SHRINK
    from ._residual_rescue_gpu import score_columns_gpu

    t, _gain = score_columns_gpu(Q, resid, even, _SWEEP_BINS, SHRINK)
    return selection_view(t)


_fallback_choice = rows_fallback(__name__)

try:
    from pyutilz.performance.kernel_tuning.registry import kernel_tuner

    _SPEC: Any = kernel_tuner(
        kernel_name="residual_rescue_score_backend",
        variant_fns=(),  # the two backends are timed by the sweep callable
        tuner=host_device_sweep(__name__, _sweep_host, _sweep_device, rtol=1e-6, atol=1e-6),
        axes={"n_samples": list(_SWEEP_ROWS)},
        fallback=_fallback_choice,
        gpu_capable=True,
        salt=_SALT,
        cli_label="residual_rescue_score_backend",
    )
except Exception as e:  # best-effort: without a tuner entry the floor decides
    logger.warning("kernel_tuner spec build for residual_rescue_score_backend failed: %s", e)
    _SPEC = None


def score_device_chosen(n: int, p: int) -> bool:
    """Whether ``p`` columns over ``n`` rows are scored on the device: the per-host kernel-tuning cache by the work ``n * p``, else the ``DEVICE_FALLBACK_ROWS`` floor."""
    work = int(n) * int(p)
    if _SPEC is None:
        return work >= DEVICE_FALLBACK_ROWS
    try:
        return str(_SPEC.choose(n_samples=work)) == "device"
    except Exception as exc:
        logger.debug("residual_rescue_score_backend: KTC choose() failed, using the floor: %s", exc)
        return work >= DEVICE_FALLBACK_ROWS


def score_columns_dispatch(host_score: Any, Q: np.ndarray, resid: np.ndarray, even: np.ndarray, n_bins: int, shrink: float) -> "tuple[np.ndarray, np.ndarray]":
    """``(t, gain)`` of the pool columns ``Q``: on the device when the tuned choice says so and a device is usable, else with ``host_score`` (the numba loop). A device fault falls back to the host."""
    p, n = Q.shape
    if p and score_device_chosen(n, p):
        try:
            from ._gpu_policy import cuda_available_for_run

            if cuda_available_for_run():
                from ._residual_rescue_gpu import score_columns_gpu

                return score_columns_gpu(Q, resid, even, n_bins, shrink)
        except Exception as exc:  # a device fault must not lose the rescue round
            from mlframe.utils.log_throttle import log_throttle

            log_throttle(logger, "residual_rescue.score_device_failed", logging.WARNING, "residual-rescue device scoring failed (%s: %s); host loop", type(exc).__name__, exc)
    host_result: "tuple[np.ndarray, np.ndarray]" = host_score(Q, resid, even, n_bins, shrink)
    return host_result


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Bin codes of the pool (``_residual_rescue_fe._codes_matrix``): kernel-tuning entry ``residual_rescue_codes_backend``, swept by the work ``n * p`` (columns x rows of the matrix)
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
CODES_DEVICE_FALLBACK_ROWS = 400_000  # matrix cells from which the device codes the pool before a sweep has run (swept on the idle card: device from 25000 x 16; 50000 x 300 host 214 ms, device 109 ms; 25000 x 120 56 / 28 ms)
_CODES_SWEEP_N = 25_000
_CODES_SWEEP_ROWS = [_CODES_SWEEP_N * k for k in (1, 4, 16, 64, 256)]
_CODES_SWEEP_BINS = 16
_CODES_SAMPLE = 4096
_CODES_MIN_FINITE = 800
_CODES_FEW_CLASSES = 20


def _codes_sweep_inputs(dims: dict) -> tuple:
    """A ``(p, 25000)`` float32 matrix of ``dims["n_samples"]`` cells with a few NaN and constant columns, the cases the coding must treat as unusable or as non-finite."""
    rng = np.random.default_rng(0)
    p = max(1, int(dims["n_samples"]) // _CODES_SWEEP_N)
    M = rng.standard_normal((p, _CODES_SWEEP_N)).astype(np.float32)
    M[0, ::7] = np.nan
    if p > 2:
        M[2] = 3.0
    return (M,)


def _codes_host(M: np.ndarray) -> np.ndarray:
    """The numba coding, as the comparison vector of the sweep: a checksum of the codes and the number of usable columns, exact for integer codes."""
    from ._residual_rescue_fe import _codes_matrix

    codes, ok = _codes_matrix(M, _CODES_SWEEP_BINS, _CODES_SAMPLE, _CODES_MIN_FINITE, _CODES_FEW_CLASSES)
    return np.array([float(codes.sum()), float(ok.sum()), float(codes.shape[0])])


def _codes_device(M: np.ndarray) -> np.ndarray:
    """The device coding; a failed launch raises, so the sweep never records a host fallback as a device time."""
    from ._residual_rescue_gpu import codes_matrix_gpu

    codes, ok = codes_matrix_gpu(M, _CODES_SWEEP_BINS, _CODES_SAMPLE, _CODES_MIN_FINITE, _CODES_FEW_CLASSES)
    return np.array([float(codes.sum()), float(ok.sum()), float(codes.shape[0])])


def _codes_sweep() -> list:
    """Time the numba and the device coding per work size; the device codes must equal the host's exactly (the comparison vector holds integer sums)."""
    from pyutilz.dev.benchmarking import sweep_backend_grid

    return list(
        sweep_backend_grid(
            {"host": _codes_host, "device": _codes_device},
            {"n_samples": _CODES_SWEEP_ROWS},
            _codes_sweep_inputs,
            reference="host",
            repeats=3,
            equiv_rtol=0.0,
            equiv_atol=0.0,
        )
    )


try:
    from pyutilz.performance.kernel_tuning.registry import kernel_tuner

    _CODES_SPEC: Any = kernel_tuner(
        kernel_name="residual_rescue_codes_backend",
        variant_fns=(),
        tuner=_codes_sweep,
        axes={"n_samples": list(_CODES_SWEEP_ROWS)},
        fallback=rows_fallback(__name__, "CODES_DEVICE_FALLBACK_ROWS"),
        gpu_capable=True,
        salt=_SALT,
        cli_label="residual_rescue_codes_backend",
    )
except Exception as e:  # best-effort: without a tuner entry the floor decides
    logger.warning("kernel_tuner spec build for residual_rescue_codes_backend failed: %s", e)
    _CODES_SPEC = None


def codes_device_chosen(n: int, p: int) -> bool:
    """Whether the ``p x n`` pool matrix is coded on the device: the per-host cache by its cell count, else the ``CODES_DEVICE_FALLBACK_ROWS`` floor."""
    work = int(n) * int(p)
    if _CODES_SPEC is None:
        return work >= CODES_DEVICE_FALLBACK_ROWS
    try:
        return str(_CODES_SPEC.choose(n_samples=work)) == "device"
    except Exception as exc:
        logger.debug("residual_rescue_codes_backend: KTC choose() failed, using the floor: %s", exc)
        return work >= CODES_DEVICE_FALLBACK_ROWS


def codes_matrix_dispatch(host_codes: Any, M: np.ndarray, n_bins: int, sample: int, min_finite: int, few_classes: int) -> "tuple[np.ndarray, np.ndarray]":
    """``(codes, ok)`` of the rows of ``M``: on the device when the tuned choice says so, else with ``host_codes`` (the numba loop). A device fault falls back to the host."""
    p, n = M.shape
    if p and M.dtype in (np.float32, np.float64) and int(n_bins) <= 127 and codes_device_chosen(n, p):
        try:
            from ._gpu_policy import cuda_available_for_run

            if cuda_available_for_run():
                from ._residual_rescue_gpu import codes_matrix_gpu

                return codes_matrix_gpu(M, n_bins, sample, min_finite, few_classes)
        except Exception as exc:  # a device fault must not lose the rescue round
            from mlframe.utils.log_throttle import log_throttle

            log_throttle(logger, "residual_rescue.codes_device_failed", logging.WARNING, "residual-rescue device coding failed (%s: %s); host loop", type(exc).__name__, exc)
    host_result: "tuple[np.ndarray, np.ndarray]" = host_codes(M, n_bins, sample, min_finite, few_classes)
    return host_result
