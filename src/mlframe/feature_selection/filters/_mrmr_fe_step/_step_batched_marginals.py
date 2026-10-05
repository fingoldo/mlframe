"""Batched device marginal MI for the FE-step candidate pools (CMI redundancy gate and escalation admitted pool)."""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Candidate float block uploaded per batched call, in elements; keeps one chunk's (n, K) float64 + int64 codes well inside a 4 GB card.
_MAX_BLOCK_ELEMENTS = 16_000_000


def batched_device_marginals(cols: list, y_codes: np.ndarray, nbins: int) -> Optional[list]:
    """Marginal MI of every column in ``cols`` against ``y_codes``, binned and scored on the device in batched chunks.

    Returns one float per column, or ``None`` when any column is not all-finite or the device path fails, so the caller keeps its
    per-candidate path. The batched binner matches the host ``_quantile_bin`` partition and ``batched_cmi_gpu`` matches
    ``_cmi_from_binned`` per column (``bench_step_score_cmi_cands.py`` pins both).
    """
    if not cols:
        return []
    try:
        on_device = all(hasattr(c, "dev") for c in cols)
        n = int(cols[0].shape[0]) if on_device else int(np.asarray(cols[0]).shape[0])
        if on_device:
            import cupy as cp

            # Device-backed candidates: the finite check is one reduction per column and the block is stacked on the device, so the continuous values are
            # never copied to the host just to be uploaded again.
            if any(int(c.shape[0]) != n or not bool(cp.isfinite(c.dev).all()) for c in cols):
                return None
        elif any(int(np.asarray(c).shape[0]) != n or not np.isfinite(c).all() for c in cols):
            return None
        from .._fe_batched_mi import batched_cmi_gpu, batched_quantile_bin_gpu

        y = np.ascontiguousarray(np.asarray(y_codes).ravel(), dtype=np.int64)
        ky = int(y.max()) + 1 if y.size else 1
        per_chunk = max(1, _MAX_BLOCK_ELEMENTS // max(1, n))
        out: list = []
        for start in range(0, len(cols), per_chunk):
            chunk = cols[start : start + per_chunk]
            if on_device:
                block = cp.stack([c.dev.astype(cp.float64, copy=False) for c in chunk], axis=1)
            else:
                block = np.column_stack([np.asarray(c, dtype=np.float64) for c in chunk])
            codes = batched_quantile_bin_gpu(block, int(nbins))
            mi = np.asarray(batched_cmi_gpu(codes, y, None, kx=int(nbins), ky=ky), dtype=np.float64)
            out.extend(float(v) for v in mi)
        return out
    except Exception as e:
        logger.warning("batched device marginal MI failed (%s: %s); scoring the FE candidates one by one", type(e).__name__, e)
        return None
