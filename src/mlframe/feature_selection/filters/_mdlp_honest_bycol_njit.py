"""Column-outer variant of ``honest_null_columns``: each column is sorted and gathered once, then every permutation reuses it.

``honest_null_columns`` runs prange over permutations and, for every (permutation, column), gathers the column's values in sorted order
from the wide ``raw_cont`` matrix: a random read with a row stride of ``k * itemsize`` bytes, ``n`` times, although the sorted values are the same
for all permutations. Here the parallel loop runs over columns, the sorted values and the fixed codes are built once per column, and the
inner loop over permutations only gathers the permuted labels. The results are identical to ``honest_null_columns`` (same operations per
(permutation, column)); ``honest_null_columns`` stays as the permutation-parallel reference.
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange

from ._mdlp_honest_njit import (
    _mdlp_cuts_presorted,
    _mdlp_dfs,
    _mi_codes,
    _n_unique_codes,
    _presort_columns,
    _supervised_edges,
)


def permuted_labels(y_raw: np.ndarray, y_codes: np.ndarray, perm_rows, count: int) -> "tuple[np.ndarray, np.ndarray]":
    """``(y_raw[perm], y_codes[perm])`` for the first ``count`` index vectors of ``perm_rows``, each ``(count, n)``, in the narrowest signed integer
    type that holds the labels. Rows are filled as they are drawn, so no ``(count, n)`` index matrix is ever held."""
    out = []
    for labels in (y_raw, y_codes):
        top = int(np.asarray(labels).max(initial=0))
        out.append(np.empty((count, labels.shape[0]), dtype=np.int8 if top < 127 else np.int16 if top < 32767 else np.int32))
    for b, rp in zip(range(count), perm_rows):
        out[0][b] = y_raw[rp]
        out[1][b] = y_codes[rp]
    return out[0], out[1]


@njit(cache=True)
def _cuts_tie_free(xs, ys, n_classes_total, min_split_size, max_depth, alpha, n_permutations, seed, analytic_min_n, analytic_min_cell):
    """``_mdlp_cuts_presorted`` for a column without tied values: the sorted points are already the deduplicated (x, y) pairs, so nothing is copied, and
    the cut buffer holds the most cuts a tree of ``max_depth`` can accept instead of one slot per row."""
    out = np.empty(min(xs.shape[0], (1 << max_depth)) + 1, dtype=np.float64)  # Intentional float64: MDLP cut points at the production binning width
    n_out = np.zeros(1, dtype=np.int64)
    _mdlp_dfs(xs, ys, n_classes_total, min_split_size, max_depth, alpha, n_permutations, seed, analytic_min_n, analytic_min_cell, out, n_out)
    cuts = out[: n_out[0]].copy()
    cuts.sort()
    return cuts


@njit(cache=True)
def _codes_at_rows(col, rows, cuts, out):
    """``out[t]`` = number of cuts <= ``col[rows[t]]`` (``searchsorted(side='right')``), filled in place."""
    for t in range(rows.shape[0]):
        v = col[rows[t]]
        lo = 0
        hi = cuts.shape[0]
        while lo < hi:
            mid = (lo + hi) // 2
            if cuts[mid] <= v:
                lo = mid + 1
            else:
                hi = mid
        out[t] = lo


@njit(parallel=True, cache=True)
def honest_null_columns_bycol(
    raw_cont: np.ndarray,
    is_cont: np.ndarray,
    fixed_codes: np.ndarray,
    fixed_nbins: np.ndarray,
    y_raw_perm: np.ndarray,
    y_codes_perm: np.ndarray,
    n_classes_total: int,
    min_split_size: int,
    max_depth: int,
    alpha: float,
    n_permutations: int,
    seed: int,
    analytic_min_n: int,
    analytic_min_cell: float,
    q_fallback: np.ndarray,
    q_fallback_len: np.ndarray,
    q_refine: np.ndarray,
    q_refine_len: np.ndarray,
    eval_rows: np.ndarray,
    col_slot: np.ndarray,
):
    """Same result as ``honest_null_columns`` (``(plain, corrected)``, each ``(permutations, k)``), prange over columns.

    ``y_raw_perm[b]`` / ``y_codes_perm[b]`` are the raw labels and the screen's target codes under permutation ``b`` (``y[perms[b]]``), stored in the
    narrowest integer type that holds them (see ``permuted_labels``): the per-row gathers then read a small, cache-resident row instead of two
    random reads into 8-byte arrays.
    """
    n_perm_rows = y_raw_perm.shape[0]
    n = y_raw_perm.shape[1]
    k = is_cont.shape[0]
    m = eval_rows.shape[0]
    plain = np.zeros((n_perm_rows, k), dtype=np.float64)  # Intentional float64: statistic vector (MI / null max / p-value inputs), exact permutation ties
    corrected = np.zeros((n_perm_rows, k), dtype=np.float64)  # Intentional float64: statistic vector (MI / null max / p-value inputs), exact permutation ties
    order = _presort_columns(raw_cont)
    for j in prange(k):
        ysrt = np.empty(n, dtype=np.int64)
        ys = np.empty(m, dtype=np.int64)
        xs_buf = np.empty(m, dtype=np.int64)
        tie_free = False
        xsrt = np.empty(n, dtype=raw_cont.dtype)
        col = np.empty(n, dtype=raw_cont.dtype)
        if is_cont[j]:
            sj = col_slot[j]
            for i in range(n):
                xsrt[i] = raw_cont[order[sj, i], sj]
                col[i] = raw_cont[i, sj]
            tie_free = True
            for i in range(1, n):
                if xsrt[i] == xsrt[i - 1]:
                    tie_free = False
                    break
        else:
            fs = col_slot[j]
            for t in range(m):
                xs_buf[t] = fixed_codes[eval_rows[t], fs]
        for b in range(n_perm_rows):
            yr_b = y_raw_perm[b]
            yc_b = y_codes_perm[b]
            ny = 0
            for t in range(m):
                v = yc_b[eval_rows[t]]
                ys[t] = v
                if v + 1 > ny:
                    ny = v + 1
            seen_y = np.zeros(ny, dtype=np.bool_)
            for t in range(m):
                seen_y[ys[t]] = True
            ky_eff = 0.0
            for c in range(ny):
                if seen_y[c]:
                    ky_eff += 1.0
            if is_cont[j]:
                sj = col_slot[j]
                for i in range(n):
                    ysrt[i] = yr_b[order[sj, i]]
                if tie_free:
                    cuts = _cuts_tie_free(xsrt, ysrt, n_classes_total, min_split_size, max_depth, alpha, n_permutations, seed, analytic_min_n, analytic_min_cell)
                else:
                    cuts = _mdlp_cuts_presorted(
                        xsrt,
                        ysrt,
                        n_classes_total,
                        min_split_size,
                        max_depth,
                        alpha,
                        n_permutations,
                        seed,
                        analytic_min_n,
                        analytic_min_cell,
                    )
                final = _supervised_edges(col, cuts, q_fallback[j, : q_fallback_len[j]], q_refine[j, : q_refine_len[j]])
                n_x = final.shape[0] + 1
                _codes_at_rows(col, eval_rows, final, xs_buf)
            else:
                n_x = fixed_nbins[j]
            xs = xs_buf
            mi = _mi_codes(xs, n_x, ys, ny)
            kx_eff = float(_n_unique_codes(xs))
            plain[b, j] = mi
            corrected[b, j] = mi - (kx_eff - 1.0) * (ky_eff - 1.0) / (2.0 * m)
    return plain, corrected
