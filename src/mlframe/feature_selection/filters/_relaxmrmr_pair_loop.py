"""The RelaxMRMR interaction term's pair loop, as one parallel kernel.

The 3-way correction sums a co-information over every pair of already-selected features, so its cost grows as ``|S|^2``, and the score itself
is computed once per candidate per greedy round. Driving that loop from Python meant three interpreter-to-kernel round trips per pair and a
fresh ``n``-length composite array per pair, both repeated ``p * |S| * (|S| - 1) / 2`` times a round.

The pairs are independent given the candidate, the target and the selected columns, so the whole loop runs as one ``prange``. Each thread
keeps one composite buffer and rewrites it per pair instead of allocating, and every pair writes its own slot of the output, which is summed
afterwards rather than accumulated across threads. Each pair's mutual information comes from the same Miller-Madow kernels the serial form
called, so the arithmetic is unchanged.
"""

from __future__ import annotations

import numpy as np
from numba import get_thread_id, njit, prange

from mlframe._numba_parallel_guard import parallel_kernel_entry
from mlframe.feature_selection.filters._relaxmrmr_kernels import _cmi_mm_njit, _composite_codes_njit, _mi_mm_njit

# Pairs times rows below which starting threads costs more than the loop does. Measured on this bench: 10 pairs at n=2k runs 0.56x
# (a loss), 190 pairs at n=2k 5.0x and 10 pairs at n=500k 2.1x, so the loss is confined to the smallest tables.
_MIN_PARALLEL_WORK = 200_000


@njit(parallel=True, nogil=True, cache=True)
def _pair_interaction_terms_njit(x, y, sel_mat, K_sel, K_x, K_y, cmi_y_mm, marg_mm, min_cells, scratch, pair_i, pair_j):
    """Per-pair ``co_cond - co_uncond`` contributions, one slot per (i, j), left at zero where the pair is not estimable."""
    n = sel_mat.shape[1]
    n_pairs = pair_i.shape[0]
    out = np.zeros(n_pairs, dtype=np.float64)
    n_rows = float(n)
    for p in prange(n_pairs):
        # The (i, j) of each flat index are precomputed: decoding them here would reassign a variable derived from the loop index, which
        # numba's parfors rejects, and it would cost an O(|S|) walk per pair anyway.
        i = pair_i[p]
        j = pair_j[p]
        K_i = K_sel[i]
        K_j = K_sel[j]
        if n_rows < min_cells * K_x * K_i * K_j * K_y:
            continue  # undersampled composite table: this pair's interaction term is not estimable
        buf = scratch[get_thread_id()]
        for r in range(n):
            buf[r] = sel_mat[i, r] * K_j + sel_mat[j, r]
        cmi_ij = _cmi_mm_njit(x, buf, y, K_x, K_i * K_j, K_y)
        mi_x_zz = _mi_mm_njit(x, buf, K_x, K_i * K_j)
        out[p] = (cmi_y_mm[i] + cmi_y_mm[j] - cmi_ij) - (marg_mm[i] + marg_mm[j] - mi_x_zz)
    return out


def _serial_pair_sum(x_int, y_int, sel_int, K_sel, K_x, K_y, cmi_y_mm, marg_mm, min_rows_per_cell) -> float:
    """The same sum, pair by pair, for the case where there is not enough work to repay starting threads."""
    n_S = len(sel_int)
    n_rows = float(x_int.shape[0])
    inter = 0.0
    for i in range(n_S):
        for j in range(i + 1, n_S):
            K_i, K_j = K_sel[i], K_sel[j]
            if n_rows < float(min_rows_per_cell) * K_x * K_i * K_j * K_y:
                continue  # undersampled composite table: this pair's interaction term is not estimable
            z_pair = _composite_codes_njit(sel_int[i], sel_int[j], K_j)
            cmi_ij = _cmi_mm_njit(x_int, z_pair, y_int, K_x, K_i * K_j, K_y)
            mi_x_zz = _mi_mm_njit(x_int, z_pair, K_x, K_i * K_j)
            inter += (cmi_y_mm[i] + cmi_y_mm[j] - cmi_ij) - (marg_mm[i] + marg_mm[j] - mi_x_zz)
    return inter


def pair_interaction_sum(x_int: np.ndarray, y_int: np.ndarray, sel_int: np.ndarray, K_sel: np.ndarray, K_x: int, K_y: int, cmi_y_mm: np.ndarray, marg_mm: np.ndarray, min_rows_per_cell: float) -> float:
    """Sum the 3-way interaction term over every pair of selected features, parallel once there is enough work to be worth it."""
    n_S = len(sel_int)
    if n_S < 2:
        return 0.0
    n = int(x_int.shape[0])
    if (n_S * (n_S - 1) // 2) * n < _MIN_PARALLEL_WORK:
        return _serial_pair_sum(x_int, y_int, sel_int, K_sel, K_x, K_y, cmi_y_mm, marg_mm, min_rows_per_cell)
    sel_mat = np.empty((n_S, n), dtype=np.int64)
    for idx, col in enumerate(sel_int):
        sel_mat[idx, :] = col
    import numba

    scratch = np.empty((max(1, int(numba.get_num_threads())), n), dtype=np.int64)
    pair_i, pair_j = np.triu_indices(n_S, k=1)
    with parallel_kernel_entry():
        terms = _pair_interaction_terms_njit(
            x_int,
            y_int,
            sel_mat,
            np.asarray(K_sel, dtype=np.int64),
            np.int64(K_x),
            np.int64(K_y),
            np.asarray(cmi_y_mm, dtype=np.float64),
            np.asarray(marg_mm, dtype=np.float64),
            float(min_rows_per_cell),
            scratch,
            np.ascontiguousarray(pair_i, dtype=np.int64),
            np.ascontiguousarray(pair_j, dtype=np.int64),
        )
    return float(terms.sum())
