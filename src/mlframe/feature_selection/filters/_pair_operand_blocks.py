"""Splitting the pairs of a pairwise GPU kernel into chunks that touch a bounded set of operand columns.

A pairwise kernel needs the two columns of every pair on the card. When the whole operand matrix does not fit, the pairs are processed in chunks and only the columns a chunk touches are
gathered and uploaded (a column block of the operand matrix per chunk). Used by the usability |corr| warp backend and by the pair-MI permutation kernel.
"""
from __future__ import annotations

import numpy as np


def operand_block_limit(n_used: int, row_bytes: int, out_bytes_per_operand: int) -> int:
    """The most operand columns whose upload (plus the pair output share) the card admits above its VRAM cushion, read once from the device; 0 when not even two fit.

    The caller has usually just been refused the whole matrix by ``fe_gpu_has_vram_cushion``, which starts the OOM cooldown; a size search through that check would be declined at every size, so the
    free memory is read directly (``fe_gpu_bytes_above_cushion``)."""
    from ._fe_gpu_vram import fe_gpu_bytes_above_cushion

    budget = fe_gpu_bytes_above_cushion()
    if budget is None:
        return int(n_used)
    m = min(int(n_used), (budget - (1 << 20)) // max(1, int(row_bytes) + int(out_bytes_per_operand)))
    return m if m >= 2 else 0


def pair_chunks_by_operand_block(pair_a: np.ndarray, pair_b: np.ndarray, max_operands: int, max_pairs: "int | None" = None) -> "list[np.ndarray]":
    """Index arrays splitting the pairs into chunks that touch at most ``max_operands`` distinct operand columns and hold at most ``max_pairs`` pairs each.

    The used columns are ranked and cut into blocks of half the operand limit; a pair belongs to the tile ``(block of a, block of b)``, and a tile touches the columns of two blocks only. An all-pairs table of
    k columns is then covered by about ``(2k / max_operands) ** 2 / 2`` tiles, each filling its launch with ``(max_operands / 2) ** 2`` pairs. Cutting the pairs in ``(a, b)`` order instead (a chunk grows until the
    next pair would exceed the limit) gave a chunk of about ``max_operands`` pairs per row of the table: 1300 launches of 512 columns for the 650000 pairs of a 1141-column table, 900 s of host gathers and uploads."""
    a = np.asarray(pair_a, dtype=np.int64)
    b = np.asarray(pair_b, dtype=np.int64)
    n_pairs = int(a.shape[0])
    if n_pairs == 0:
        return []
    used = np.unique(np.concatenate([a, b]))
    if used.shape[0] <= int(max_operands):
        tile = np.zeros(n_pairs, dtype=np.int64)
    else:
        half = max(1, int(max_operands) // 2)
        block_a, block_b = np.searchsorted(used, a) // half, np.searchsorted(used, b) // half
        tile = np.minimum(block_a, block_b) * (used.shape[0] // half + 1) + np.maximum(block_a, block_b)
    order = np.argsort(tile, kind="stable")
    sorted_tile = tile[order]
    starts = np.flatnonzero(np.concatenate([[True], sorted_tile[1:] != sorted_tile[:-1]]))
    chunks: "list[np.ndarray]" = []
    for first, last in zip(starts.tolist(), np.concatenate([starts[1:], [n_pairs]]).tolist()):
        members = order[first:last]
        step = int(max_pairs) if max_pairs is not None and int(max_pairs) > 0 else int(members.shape[0])
        chunks.extend(members[i : i + step] for i in range(0, int(members.shape[0]), step))
    return [np.asarray(c, dtype=np.int64) for c in chunks]
