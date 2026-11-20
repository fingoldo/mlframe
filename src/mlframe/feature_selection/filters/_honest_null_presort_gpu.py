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
from mlframe.feature_selection.filters._device_sort_safe import device_argsort, device_sort  # thrust aborts the process on a short pool
from ._fe_gpu_vram import pool_column_block
from mlframe.utils.log_throttle import log_throttle
from ._gpu_resident_select_kernels import _transpose_to_cm

logger = logging.getLogger(__name__)

#: Elements (columns x rows) from which the device sort is the default choice. The device sort alone wins from about 200 000 elements on the GTX 1050 Ti (against the
#: eight-thread host presort, upload and both reads included, identical outputs: 100000 rows x 4 / 16 / 40 columns 5 / 19 / 54 ms against 18 / 72 / 220 ms; 30000 x 8 / 40: 4 / 19
#: against 6 / 40 ms; 3000 x 40 equal), but the order and the tie flags come back to the host (5 bytes a cell) and the acceptance kernel uploads the order again: lowering the
#: floor to that crossover added 2.1 MB of device-to-host traffic to a 30000-row strict fit and broke its residency budget. The floor stays until the order is kept on the device.
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


_KEYS_BYTES_PER_COLUMN_ROW = 56  # float copy, ordered bits, the 64-bit key, its sorted copy and the sort's temporary, the order and the flags


def _presort_by_keys(cp, raw_cont: np.ndarray) -> "tuple[np.ndarray, np.ndarray] | None":
    """The order and tie flags of :func:`presort_and_ties_gpu` from ONE 1-D sort of composite keys per block of columns (the sort of a long vector runs one ``argsort`` per column otherwise): the key is
    ``column | ordered bits of the float32 value | row``, so the sorted keys list each column's rows in value order and, among equal values, in row order (the stable argsort's order; -0.0 is mapped to +0.0 so the two
    tie), the row is read from the low bits and a tie is two adjacent keys with the same ``column | value`` prefix. ``None`` for float64 columns and shapes whose key needs more than 64 bits."""
    n, k = int(raw_cont.shape[0]), int(raw_cont.shape[1])
    if raw_cont.dtype != np.float32:  # Intentional float64: only float32 columns fit a 64-bit composite key
        return None
    row_bits = max(1, int(n - 1).bit_length())
    block = max(1, _presort_column_block_bytes(cp, n, _KEYS_BYTES_PER_COLUMN_ROW))
    col_bits = max(1, int(min(block, k) - 1).bit_length())
    if col_bits + 32 + row_bits > 64:
        return None
    order = np.empty((k, n), dtype=np.int32)
    tie = np.zeros((k, n), dtype=np.uint8)
    row_mask = np.uint64((1 << row_bits) - 1)
    rows = cp.arange(n, dtype=cp.uint64)[None, :]
    for c0 in range(0, k, block):
        c1 = min(k, c0 + block)
        kc = c1 - c0
        cm = _transpose_to_cm(cp.asarray(np.ascontiguousarray(raw_cont[:, c0:c1])))  # (kc, n) float32, one row per column
        bits = (cm + np.float32(0.0)).view(cp.uint32)  # -0.0 becomes +0.0
        ordered = bits ^ ((bits >> 31) * np.uint32(0xFFFFFFFF) | np.uint32(0x80000000))  # unsigned order of the float order
        key = (cp.arange(kc, dtype=cp.uint64)[:, None] << np.uint64(32 + row_bits)) | (ordered.astype(cp.uint64) << np.uint64(row_bits)) | rows
        del cm, bits, ordered
        sk = device_sort(key.ravel())
        del key
        order[c0:c1] = cp.asnumpy((sk & row_mask).astype(cp.int32).reshape(kc, n))
        prefix = sk >> np.uint64(row_bits)
        flags = cp.zeros(kc * n, dtype=cp.uint8)
        flags[:-1] = prefix[:-1] == prefix[1:]  # the column is part of the prefix, so a column's last row never ties with the next column's first
        tie[c0:c1] = cp.asnumpy(flags.reshape(kc, n))
        del sk, prefix, flags
    return order, tie


def _presort_column_block_bytes(cp, n_rows: int, bytes_per_column_row: int) -> int:
    """Columns per launch within ``_BUDGET_FRACTION`` of what the memory pool can still hand out, for ``bytes_per_column_row`` bytes a column row."""
    return pool_column_block(cp, n_rows, bytes_per_column_row=bytes_per_column_row, budget_fraction=_BUDGET_FRACTION, min_block=4, fallback=8, what="device presort")


def presort_and_ties_gpu(raw_cont: np.ndarray) -> "tuple[np.ndarray, np.ndarray] | None":
    """``(order (columns, n) int32, tie (columns, n) uint8)`` of the columns of ``raw_cont`` ``(n, columns)`` as ``_presort_columns`` and ``_tie_flags`` give them, or ``None`` when the device path does not apply or fails."""
    n, k = int(raw_cont.shape[0]), int(raw_cont.shape[1])
    if k == 0 or n < 2:
        return None
    if raw_cont.dtype not in (np.float32, np.float64) or not np.isfinite(raw_cont).all():  # Intentional float64: the dtypes the block sort accepts
        log_throttle(logger, "_honest_null_presort_gpu.not_finite", logging.WARNING, "device presort skipped: the columns are not all finite float32 / float64 (dtype %s); sorting on the host", raw_cont.dtype)
        return None
    try:
        import cupy as cp
    except Exception as exc:
        log_throttle(logger, "_honest_null_presort_gpu.cupy_unavailable", logging.WARNING, "device presort unavailable (%s: %s); sorting on the host", type(exc).__name__, exc)
        return None
    try:
        got = _presort_by_keys(cp, raw_cont)
        if got is not None:
            cp.get_default_memory_pool().free_all_blocks()
            return got
    except Exception as exc:  # best-effort: the per-column device sort below gives the same order
        logger.info("device presort by composite keys failed (%s: %s); sorting per column", type(exc).__name__, exc)
    order = np.empty((k, n), dtype=np.int32)
    tie = np.zeros((k, n), dtype=np.uint8)
    try:
        block = _presort_column_block(cp, n)
        c0 = 0
        while c0 < k:
            c1 = min(k, c0 + block)
            cols = idx = sv = tie_dev = None
            try:
                cols = _transpose_to_cm(cp.asarray(np.ascontiguousarray(raw_cont[:, c0:c1])))  # one contiguous row per column, transposed on the device
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

                note_gpu_oom()  # the block-halving retry above gave up (block already at its floor); arm the shared cooldown for other call sites.
        except Exception:  # nosec B110 - best-effort cooldown note, never let it mask the original failure
            logger.debug("GPU out-of-memory cooldown note failed", exc_info=True)
        return None
    return order, tie


__all__ = ["presort_and_ties_gpu", "use_device_presort"]
