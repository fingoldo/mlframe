"""Host or device for the categorical pair search, decided by the kernel-tuning cache on the row count.

``run_cat_interaction_step`` scores every candidate pair with the njit ``_pair_search_kernel_njit`` (prange over the pairs) or, on ``backend="auto"``, with the device
``mi_direct_gpu_batched_pairs`` (one launch histograms every pair, one launch reduces them to the joint MI). The device was taken by a work product fixed by hand
(``n_samples * n_cols >= 200 * 500000``); this entry measures it: both backends score the same pairs of 12-bin columns and the device must return the same joint MI.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from ._ktc_fallback import host_device_sweep, rows_fallback

logger = logging.getLogger(__name__)

DEVICE_FALLBACK_ROWS = 500_000  # the hand-set row mark of the old work-product rule; the tuner replaces it per host once a sweep has run
_SWEEP_ROWS = [20_000, 100_000, 500_000, 2_000_000]
_SWEEP_COLS = 60
_SWEEP_BINS = 12
_SALT = 1


def _sweep_inputs(dims: dict) -> tuple:
    """``(codes, pairs_a, pairs_b, nbins, y, freqs_y)``: ``_SWEEP_COLS`` columns of ``dims["n_samples"]`` rows, every pair, a binary label tied to two columns."""
    rng = np.random.default_rng(0)
    n = int(dims["n_samples"])
    codes = rng.integers(0, _SWEEP_BINS, size=(n, _SWEEP_COLS)).astype(np.int32)
    y = (((codes[:, 0] + codes[:, 1]) % 3 == 0) ^ (rng.random(n) < 0.1)).astype(np.int32)
    ia, ib = np.triu_indices(_SWEEP_COLS, k=1)
    nbins = np.full(_SWEEP_COLS, _SWEEP_BINS, dtype=np.int32)
    freqs_y = np.bincount(y, minlength=2).astype(np.float64) / n  # Intentional float64: class frequencies
    return (codes, ia.astype(np.int64), ib.astype(np.int64), nbins, y, freqs_y)


def _sweep_host(codes: np.ndarray, ia: np.ndarray, ib: np.ndarray, nbins: np.ndarray, y: np.ndarray, freqs_y: np.ndarray) -> np.ndarray:
    """Joint MI of every pair by the njit prange kernel."""
    from .cat_interactions import _pair_search_kernel_njit

    marginal = np.zeros(codes.shape[1], dtype=np.float64)  # Intentional float64: the kernel reads it only to form the interaction information
    return np.asarray(_pair_search_kernel_njit(codes, ia, ib, marginal, nbins, y, freqs_y, np.int32)[0])


def _sweep_device(codes: np.ndarray, ia: np.ndarray, ib: np.ndarray, nbins: np.ndarray, y: np.ndarray, freqs_y: np.ndarray) -> np.ndarray:
    """Joint MI of every pair by the batched device kernels."""
    from ._gpu_pairs import mi_direct_gpu_batched_pairs

    return np.asarray(mi_direct_gpu_batched_pairs(codes, ia, ib, nbins, y, freqs_y, dtype=np.int32))


_fallback_choice = rows_fallback(__name__)

try:
    from pyutilz.performance.kernel_tuning.registry import kernel_tuner

    _SPEC: Any = kernel_tuner(
        kernel_name="cat_pair_search_backend",
        variant_fns=(),  # the two backends are timed by the sweep callable
        tuner=host_device_sweep(__name__, _sweep_host, _sweep_device, rtol=1e-9, atol=1e-12),
        axes={"n_samples": list(_SWEEP_ROWS)},
        fallback=_fallback_choice,
        gpu_capable=True,
        salt=_SALT,
        cli_label="cat_pair_search_backend",
    )
except Exception as e:  # best-effort: without a tuner entry the row-count fallback decides
    logger.warning("kernel_tuner spec build for cat_pair_search_backend failed: %s", e)
    _SPEC = None


def cat_pair_search_on_device(n_rows: int) -> bool:
    """Whether the pair search of ``n_rows`` rows takes the device kernels: the per-host kernel-tuning cache, else the ``DEVICE_FALLBACK_ROWS`` mark."""
    if _SPEC is None:
        return int(n_rows) >= DEVICE_FALLBACK_ROWS
    try:
        return str(_SPEC.choose(n_samples=int(n_rows))) == "device"
    except Exception as exc:
        logger.debug("cat_pair_search_backend: KTC choose() failed, using the row-count fallback: %s", exc)
        return int(n_rows) >= DEVICE_FALLBACK_ROWS
