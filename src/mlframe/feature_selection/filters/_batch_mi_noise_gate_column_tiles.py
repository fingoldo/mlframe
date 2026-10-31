"""Column-tiled resident noise gate for a counts buffer that does not fit free VRAM.

The resident gate allocates one ``(P, total_size)`` int64 counts buffer for every candidate column at once. When that buffer
cannot fit the free-VRAM cushion, the candidate columns are split into contiguous tiles, each tile is gated on the device
with its own counts buffer, and the per-column gated vectors are concatenated. The gate decision of a column depends only on
that column's own original and permuted MI, so the tiled result equals the untiled one.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from mlframe.utils.log_throttle import log_throttle

logger = logging.getLogger(__name__)


def _widest_fitting_tile(cum_bytes: np.ndarray, start: int, K: int) -> int:
    """Largest number of columns from ``start`` whose counts bytes pass the VRAM cushion check; 0 when even one column does not fit.

    ``cum_bytes[i]`` is the cumulative counts bytes of the first ``i`` columns, so the bytes of a tile ``[start, start + w)`` are
    ``cum_bytes[start + w] - cum_bytes[start]``. The cushion predicate is monotone in the byte count, so a binary search finds the
    widest tile.
    """
    from ._fe_gpu_vram import fe_gpu_has_vram_cushion

    lo, hi, best = 1, K - start, 0
    while lo <= hi:
        mid = (lo + hi) // 2
        tile_bytes = int(cum_bytes[start + mid] - cum_bytes[start])
        if fe_gpu_has_vram_cushion(tile_bytes):
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1
    return best


def gate_in_column_tiles(
    gate: Any,
    disc_2d: np.ndarray,
    nbins_arr: np.ndarray,
    classes_y: np.ndarray,
    classes_y_safe: np.ndarray,
    freqs_y: np.ndarray,
    npermutations: int,
    base_seed: np.uint64,
    min_nonzero_confidence: float,
    use_su: bool,
    dtype: type,
    threads_per_block: "int | None",
    K_y: int,
    P: int,
) -> np.ndarray | None:
    """Run ``gate`` (the resident gate itself) over contiguous column tiles and concatenate the gated vectors.

    Returns None when even a single column's counts buffer does not fit, so the caller takes the CPU path.
    """
    K = int(disc_2d.shape[1])
    cum_bytes = np.zeros(K + 1, dtype=np.int64)
    cum_bytes[1:] = np.cumsum(np.asarray(nbins_arr, dtype=np.int64) * K_y * P * 8)
    parts: list[np.ndarray] = []
    start = 0
    while start < K:
        width = _widest_fitting_tile(cum_bytes, start, K)
        if width == 0:
            log_throttle(
                logger,
                "noise_gate.resident_column_too_big",
                logging.WARNING,
                "noise gate: a single column's resident counts buffer does not fit the free VRAM cushion; using the CPU gate",
            )
            return None
        stop = start + width
        part = gate(
            disc_2d=np.ascontiguousarray(disc_2d[:, start:stop]),
            factors_nbins=nbins_arr[start:stop],
            classes_y=classes_y,
            classes_y_safe=classes_y_safe,
            freqs_y=freqs_y,
            npermutations=npermutations,
            base_seed=base_seed,
            min_nonzero_confidence=min_nonzero_confidence,
            use_su=use_su,
            dtype=dtype,
            threads_per_block=threads_per_block,
            d_disc_resident=None,
        )
        if part is None:
            return None
        parts.append(np.asarray(part))
        start = stop
    log_throttle(
        logger,
        "noise_gate.resident_column_tiles",
        logging.INFO,
        "noise gate: resident counts buffer (P=%d, K=%d, K_y=%d) exceeds the free VRAM cushion; gated in %d column tiles",
        P,
        K,
        K_y,
        len(parts),
    )
    out: np.ndarray = np.concatenate(parts)
    return out
