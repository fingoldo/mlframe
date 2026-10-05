"""Analytic large-n noise gate fed from device-resident candidate codes.

At large n the pair search's permutation gate is replaced by the analytic G-test: the gate needs each candidate column's observed plug-in MI and its occupied
bin count. The dispatcher used to get both on the host - it copied the (n, K) codes back from the device, ran the CPU observed-MI kernel over them (about 1.5 s
per call at n=30000, K~2000, and 27 of the 36 s of a 100k-row strict fit) and counted occupied bins with an njit pass. When the producer kept the codes
resident, both reductions are one pass each on the device, and only the (K,) MI vector and the (K,) occupied counts cross the bus.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Codes per device block (n * k elements): the MI kernel widens a block to int64, so this bounds that transient to ~256 MB on a small card.
_BLOCK_ELEMENTS = 1 << 25


def resident_observed_mi_and_bins(device_codes, classes_y: np.ndarray, by: int) -> "tuple[np.ndarray, np.ndarray]":
    """Observed plug-in MI (nats) and occupied-bin count of every column of the resident ``(n, K)`` code matrix, against the target codes ``classes_y``.

    Both are computed on the device block by block; the returned host arrays are ``(K,)`` float64 and ``(K,)`` int64 (the same quantities the CPU
    ``npermutations=0`` kernel and ``_occupied_bins_per_col`` return).
    """
    import cupy as cp

    from mlframe.feature_selection.filters._fe_batched_mi import binned_mi_from_codes_gpu

    yc = np.ascontiguousarray(classes_y, dtype=np.int64).ravel()
    n, k_total = int(device_codes.shape[0]), int(device_codes.shape[1])
    step = max(1, min(k_total, _BLOCK_ELEMENTS // max(1, n)))
    observed = np.empty(k_total, dtype=np.float64)
    bins = np.empty(k_total, dtype=np.int64)
    for start in range(0, k_total, step):
        block = device_codes[:, start : start + step]
        width = int(block.shape[1])
        observed[start : start + width] = binned_mi_from_codes_gpu(block, yc, ky=int(by), codes_trusted=True)
        wide = block.astype(cp.int64, copy=False)
        m = int(wide.max()) + 1 if wide.size else 1
        counts = cp.bincount((wide * width + cp.arange(width, dtype=cp.int64)[None, :]).ravel(), minlength=m * width)
        bins[start : start + width] = (counts.reshape(m, width) > 0).sum(axis=0).get().astype(np.int64)
    return observed, bins


def resident_analytic_gate(device_codes, classes_y: np.ndarray, by: int, n_rows: int, min_nonzero_confidence: float) -> Optional[np.ndarray]:
    """``fe_mi[K]`` of the analytic noise gate computed from resident codes, or ``None`` on any device fault so the caller keeps the host path."""
    try:
        from .._analytic_mi_null import analytic_batch_noise_gate

        observed, bins = resident_observed_mi_and_bins(device_codes, classes_y, by)
        return analytic_batch_noise_gate(None, observed, classes_y, int(n_rows), float(min_nonzero_confidence), bx_per_col=bins, by=int(by))
    except Exception as e:
        logger.debug("resident analytic noise gate failed, using the host path: %s", e)
        return None
