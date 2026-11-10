"""Streaming column blocks through a device stage, shrinking the block when the pool runs out of memory.

A stage that puts the whole ``(rows, columns)`` matrix and its per-basis, per-degree intermediates on the device needs a multiple of the matrix that grows with the column count; at a
thousand columns it exceeds the share of the card a concurrent stage owns. The columns are independent, so the stage runs on blocks of columns and the pieces are joined in order. The first
block is sized from the room under the calling thread's pool; a block that still runs out of memory is halved (the pool's cached blocks are released first) down to one column.
"""

from __future__ import annotations

from typing import Any, Callable, List


def first_block(cp: Any, n_rows: int, n_cols: int, itemsize: int, multiple: int) -> int:
    """Columns of the first block: ``multiple`` copies of a ``(rows, block)`` array in ``itemsize`` bytes fill at most a fifth of the room under the calling thread's pool."""
    from ._device_sort_safe import device_room

    per_col = max(1, int(n_rows) * int(itemsize) * int(multiple))
    return int(max(1, min(int(n_cols), (device_room(cp) // 5) // per_col)))


def run_column_blocks(cp: Any, n_cols: int, block: int, work: Callable[[int, int], Any]) -> List[Any]:
    """``[work(lo, hi), ...]`` over consecutive column blocks starting at ``block`` columns; a block that runs out of device memory is retried at half the width."""
    from ._device_sort_safe import _current_pool

    pieces: List[Any] = []
    lo = 0
    width = max(1, min(int(n_cols), int(block)))
    while lo < n_cols:
        hi = min(n_cols, lo + width)
        try:
            piece = work(lo, hi)
        except cp.cuda.memory.OutOfMemoryError:
            if width == 1:
                raise
            for pool in (_current_pool(cp), cp.get_default_memory_pool()):
                if pool is not None:
                    pool.free_all_blocks()
            width = max(1, width // 2)
            continue
        pieces.append(piece)
        lo = hi
    return pieces
