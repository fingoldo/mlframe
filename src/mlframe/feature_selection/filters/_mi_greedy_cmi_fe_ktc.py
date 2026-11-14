"""Host-versus-device choice of the batched candidate scoring of the CMI-greedy FE construction (kernel-tuning entry ``cmi_candidate_batch_backend``).

A greedy round scores ``p`` candidate columns of ``n`` rows against a fixed target and support. The host scores them one by one with the hoisted (Y,Z) terms
(``cmi_from_binned_fixed_yz``); the device scores them in one ``batched_cmi_gpu`` workload. Both count the same partitions, so the scores agree to rounding and what a round reads
from them, the best candidates, is held to the strict tolerance by ``selection_view``. The crossover depends on the work ``n * p``; until a sweep has run on a host the floors below
decide (they are the former ``_STRICT_MIN_CALL_*`` pair of ``_fe_gpu_strict``, measured on the development card).
"""

from __future__ import annotations

import logging
import os
from typing import Any

import numpy as np

from ._ktc_fallback import host_device_sweep, rows_fallback, selection_view

logger = logging.getLogger(__name__)

__all__ = ["candidate_batch_device_chosen"]

DEVICE_FALLBACK_ROWS = 1_000_000  # candidate-rows (n * p) from which the device scores the batch before a sweep has run
DEVICE_FALLBACK_MIN_P = 64  # candidates per call below which the host scores them
_SWEEP_N = 30_000
_SWEEP_ROWS = [_SWEEP_N * k for k in (1, 4, 16, 64, 256, 1024)]  # n_samples is the work n * p, swept at 30000 rows
_SWEEP_BINS = 10
_SWEEP_SUPPORT_CLASSES = 30
_SALT = 1


def _sweep_inputs(dims: dict) -> tuple:
    """``(candidates, y, z)`` for a work of ``dims["n_samples"]`` candidate-rows: ``p`` equi-frequency-like codes over 30000 rows, a binary target that depends on the first two candidates, a
    30-class support."""
    rng = np.random.default_rng(0)
    p = max(1, int(dims["n_samples"]) // _SWEEP_N)
    X = rng.integers(0, _SWEEP_BINS, (_SWEEP_N, p)).astype(np.int64)
    y = ((X[:, 0] + X[:, min(1, p - 1)] + rng.integers(0, 6, _SWEEP_N)) > 12).astype(np.int64)
    z = rng.integers(0, _SWEEP_SUPPORT_CLASSES, _SWEEP_N).astype(np.int64)
    return (X, y, z)


def _sweep_host(X: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    """The host path of a greedy round (the per-candidate loop with the hoisted terms), with the device routes of the scorer switched off for the timing."""
    from ._mi_greedy_cmi_fe_steps import _renumber_joint, cmi_from_binned_fixed_yz, precompute_cmi_yz_terms

    saved = {k: os.environ.get(k) for k in ("MLFRAME_CMI_GPU", "MLFRAME_FE_GPU_STRICT")}
    os.environ["MLFRAME_CMI_GPU"] = "0"
    os.environ["MLFRAME_FE_GPU_STRICT"] = "0"
    try:
        y_i, z_i, h_yz, h_z, k_yz, k_z, n_f = precompute_cmi_yz_terms(y, z)
        yz_dense, _ = _renumber_joint(y_i, z_i)
        scores = np.array([cmi_from_binned_fixed_yz(X[:, j], y_i, z_i, h_yz, h_z, k_yz, k_z, n_f, yz_i=yz_dense) for j in range(X.shape[1])], dtype=np.float64)
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    return selection_view(scores)


def _sweep_device(X: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    """The device path of a greedy round: one batched CMI workload over all candidates; a failed launch raises, so the sweep never records a host fallback as a device time."""
    from ._fe_batched_mi import batched_cmi_gpu

    return selection_view(np.asarray(batched_cmi_gpu(X, y, z, codes_trusted=True, kx=_SWEEP_BINS), dtype=np.float64))


_fallback_choice = rows_fallback(__name__)

try:
    from pyutilz.performance.kernel_tuning.registry import kernel_tuner

    _SPEC: Any = kernel_tuner(
        kernel_name="cmi_candidate_batch_backend",
        variant_fns=(),  # the two backends are timed by the sweep callable
        tuner=host_device_sweep(__name__, _sweep_host, _sweep_device, rtol=1e-6, atol=1e-6),
        axes={"n_samples": list(_SWEEP_ROWS)},
        fallback=_fallback_choice,
        gpu_capable=True,
        salt=_SALT,
        cli_label="cmi_candidate_batch_backend",
    )
except Exception as e:  # best-effort: without a tuner entry the floors decide
    logger.warning("kernel_tuner spec build for cmi_candidate_batch_backend failed: %s", e)
    _SPEC = None


def candidate_batch_device_chosen(n: int, p: int) -> bool:
    """Whether ``p`` candidates over ``n`` rows are scored by the batched device CMI: the per-host kernel-tuning cache by the work ``n * p``, else the floors (at least
    ``DEVICE_FALLBACK_MIN_P`` candidates and ``DEVICE_FALLBACK_ROWS`` candidate-rows)."""
    work = int(n) * int(p)

    def _floors() -> bool:
        """The pre-sweep rule."""
        return int(p) >= DEVICE_FALLBACK_MIN_P and work >= DEVICE_FALLBACK_ROWS

    if _SPEC is None:
        return _floors()
    try:
        return str(_SPEC.choose(n_samples=work)) == "device"
    except Exception as exc:
        logger.debug("cmi_candidate_batch_backend: KTC choose() failed, using the floors: %s", exc)
        return _floors()
