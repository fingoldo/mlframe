"""Splitting the pairs of a pairwise GPU kernel into chunks that touch a bounded set of operand columns.

A pairwise kernel needs the two columns of every pair on the card. When the whole operand matrix does not fit, the pairs are processed in chunks and only the columns a chunk touches are
gathered and uploaded (a column block of the operand matrix per chunk). Used by the usability |corr| warp backend and by the pair-MI permutation kernel.
"""
from __future__ import annotations

import numpy as np


def operand_block_limit(n_used: int, row_bytes: int, out_bytes_per_operand: int) -> int:
    """The most operand columns whose upload (plus the pair output share) the VRAM cushion admits, found by halving from ``n_used``; 0 when not even two fit."""
    from ._fe_gpu_vram import fe_gpu_has_vram_cushion

    m = int(n_used)
    while m >= 2 and not fe_gpu_has_vram_cushion(m * (row_bytes + out_bytes_per_operand) + (1 << 20)):
        m //= 2
    return m if m >= 2 else 0


def pair_chunks_by_operand_block(pair_a: np.ndarray, pair_b: np.ndarray, max_operands: int, max_pairs: "int | None" = None) -> "list[np.ndarray]":
    """Index arrays splitting the pairs (ordered by ``(a, b)``) into chunks that touch at most ``max_operands`` distinct operand columns and hold at most ``max_pairs`` pairs each."""
    order = np.lexsort((pair_b, pair_a))
    chunks: "list[list[int]]" = []
    current: "list[int]" = []
    seen: set = set()
    for idx in order.tolist():
        a, b = int(pair_a[idx]), int(pair_b[idx])
        extra = (a not in seen) + (b not in seen and b != a)
        if current and (len(seen) + extra > max_operands or (max_pairs is not None and len(current) >= max_pairs)):
            chunks.append(current)
            current, seen = [idx], {a, b}
        else:
            current.append(idx)
            seen.update((a, b))
    if current:
        chunks.append(current)
    return [np.asarray(c, dtype=np.int64) for c in chunks]
