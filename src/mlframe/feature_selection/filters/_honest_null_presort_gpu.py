"""Device twin of the host preparation of the honest null: the stable sort order of every continuous column and the tie flags of the sorted rows.

The host takes ``_presort_columns`` (a stable argsort per column, 2.2 to 2.6 s for 250 columns x 100000 rows) and ``_tie_flags`` (0.3 s). ``cupy.argsort`` along the rows of a column-major copy gives the SAME permutation, ties included (the sort is stable; checked on columns
with heavy ties and on a constant column), and the tie flag of sorted position ``i`` is one comparison of two gathered values. Columns go through in blocks that stay within what the memory pool can still hand out, and the pool's unused cache is returned afterwards, so the temporaries
(an int64 argsort next to the float copy) cannot starve a later stage.

Applies to finite columns only (the host path stays for anything else); a device fault returns ``None`` and the caller sorts on the host.
"""

from __future__ import annotations

import logging
import os

import numpy as np
from mlframe.feature_selection.filters._device_sort_safe import device_argsort, device_sort  # noqa: F401  (thrust aborts the process on a short pool)
from ._fe_gpu_vram import pool_column_block

logger = logging.getLogger(__name__)

#: Elements (columns x rows) from which the device sort is the default choice.
DEVICE_MIN_ELEMENTS = 5_000_000
_BUDGET_FRACTION = 0.25
#: Device bytes one column row holds at the peak: the float copy, the int64 argsort, the sorted copy and the flags.
_BYTES_PER_COLUMN_ROW = 28


def _presort_column_block(cp, n_rows: int) -> int:
    """Columns per launch within ``_BUDGET_FRACTION`` of what the memory pool can still hand out (at least 4)."""
    return pool_column_block(cp, n_rows, bytes_per_column_row=_BYTES_PER_COLUMN_ROW, budget_fraction=_BUDGET_FRACTION, min_block=4, fallback=8, what="device presort")


def use_device_presort(n_rows: int, n_cols: int) -> bool:
    """Whether the sort runs on the device: ``MLFRAME_HONEST_PRESORT_DEVICE=0|1`` forces a path, else from ``DEVICE_MIN_ELEMENTS`` elements."""
    forced = os.environ.get("MLFRAME_HONEST_PRESORT_DEVICE", "").strip()
    if forced in ("0", "1"):
        return forced == "1"
    return int(n_rows) * int(n_cols) >= DEVICE_MIN_ELEMENTS


def presort_and_ties_gpu(raw_cont: np.ndarray) -> "tuple[np.ndarray, np.ndarray] | None":
    """``(order (columns, n) int32, tie (columns, n) uint8)`` of the columns of ``raw_cont`` ``(n, columns)`` as ``_presort_columns`` and ``_tie_flags`` give them, or ``None`` when the device path does not apply or fails."""
    n, k = int(raw_cont.shape[0]), int(raw_cont.shape[1])
    if k == 0 or n < 2 or raw_cont.dtype not in (np.float32, np.float64) or not np.isfinite(raw_cont).all():  # Intentional float64: the dtypes the block sort accepts
        return None
    try:
        import cupy as cp
    except Exception as exc:
        logger.debug("device presort unavailable (%s)", exc)
        return None
    order = np.empty((k, n), dtype=np.int32)
    tie = np.zeros((k, n), dtype=np.uint8)
    try:
        block = _presort_column_block(cp, n)
        c0 = 0
        while c0 < k:
            c1 = min(k, c0 + block)
            cols = idx = sv = tie_dev = None
            try:
                cols = cp.asarray(np.ascontiguousarray(raw_cont[:, c0:c1].T))  # one contiguous row per column
                idx = device_argsort(cols, axis=1)
                sv = cp.take_along_axis(cols, idx, axis=1)
                tie_dev = cp.zeros((c1 - c0, n), dtype=cp.uint8)
                tie_dev[:, : n - 1] = (sv[:, :-1] == sv[:, 1:]).astype(cp.uint8)
                order[c0:c1] = cp.asnumpy(idx.astype(cp.int32))
                tie[c0:c1] = cp.asnumpy(tie_dev)
            except cp.cuda.memory.OutOfMemoryError:
                # concurrent FE stages draw from the same capped pool, so the room measured when the block was sized can be gone when it is allocated: the block is halved and retried
                # before the whole sort is given up to the host
                cols = idx = sv = tie_dev = None  # the failed block's buffers go back to the pool before the retry
                if block <= 4:
                    raise
                block = max(4, block // 2)
                logger.info("device presort: out of memory on %d columns; retrying with blocks of %d", c1 - c0, block)
                continue
            c0 = c1
        cp.get_default_memory_pool().free_all_blocks()
    except Exception as exc:
        logger.warning("device presort of the honest-null columns failed (%s: %s); sorting on the host", type(exc).__name__, exc)
        try:
            import cupy as _cp

            if isinstance(exc, _cp.cuda.memory.OutOfMemoryError):
                from ._fe_gpu_vram import note_gpu_oom

                note_gpu_oom()  # GPU11-13: the block-halving retry above gave up (block already at its floor); arm the shared cooldown for other call sites.
        except Exception:  # nosec B110 - best-effort cooldown note, never let it mask the original failure
            pass
        return None
    return order, tie


__all__ = ["presort_and_ties_gpu", "use_device_presort"]
