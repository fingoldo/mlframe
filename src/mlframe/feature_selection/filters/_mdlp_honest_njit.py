"""Numba honest supervised null: the validated MDLP cut search re-run on every permuted target, with ``prange`` over permutations.

Used by the screen floor and the empty-support fallback when the binning strategy is supervised (see
``_honest_mdlp_null``). Only continuous candidate columns are re-binned on each permutation; low-cardinality and
other target-free columns keep their observed codes, which is exact because their binning does not read the target.

The recursion is the depth-first form of the production level-order driver (``_mdlp_recurse_validated_bfs``). Each node's
permutation seed depends only on the root seed, the depth and the split index, so traversal order does not change the
decisions. The per-node permutation test is a serial twin of ``_mdlp_permutation_batch_njit``: a nested parallel kernel
inside the outer ``prange`` crashed, so only the outer loop over permutations is parallel.
"""

from __future__ import annotations


import numpy as np
from numba import njit, prange

# The production validated path (_mdlp_validated_split) scores splits with the incremental kernel; gains of the two kernels differ in the
# last bits, which flips the accept decision on exact ties (permutation null equal to the observed gain). Use the same kernel.
from .supervised_binning import _mdlp_best_split_inc_njit as _mdlp_best_split_njit  # noqa: F401 -- re-exported for callers of this module


from ._numba_utils import (  # noqa: F401 -- helpers hosted in _numba_utils_cross_module for the cross-module njit invariant
    _accept_split_serial,
    _analytic_p_value,
    _codes_from_cuts,
    _compact_labels,
    _gammaincc_nb,
    _mdlp_cuts_presorted,
    _mdlp_dfs,
    _mi_codes,
    _n_unique_codes,
    _presorted_dedup,
    _supervised_edges,
)


@njit(cache=True)
def _sorted_dedup(x: np.ndarray, y: np.ndarray):
    """Sort by (x, y) with two stable passes, then drop adjacent exact duplicates (same as ``_dedupe_xy``)."""
    o1 = np.argsort(y, kind="mergesort")
    x1 = x[o1]
    y1 = y[o1]
    o2 = np.argsort(x1, kind="mergesort")
    xs = x1[o2]
    ys = y1[o2]
    n = xs.shape[0]
    if n <= 1:
        return xs, ys
    keep = np.empty(n, dtype=np.bool_)
    keep[0] = True
    cnt = 1
    for i in range(1, n):
        keep[i] = (xs[i] != xs[i - 1]) or (ys[i] != ys[i - 1])
        if keep[i]:
            cnt += 1
    xd = np.empty(cnt, dtype=np.float64)  # Intentional float64: MDLP cut search at the production binning width (cut points must match supervised_binning)
    yd = np.empty(cnt, dtype=np.int64)
    j = 0
    for i in range(n):
        if keep[i]:
            xd[j] = xs[i]
            yd[j] = ys[i]
            j += 1
    return xd, yd


@njit(cache=True)
def mdlp_inner_cuts(
    x: np.ndarray,
    y: np.ndarray,
    n_classes_total: int,
    min_split_size: int,
    max_depth: int,
    alpha: float,
    n_permutations: int,
    seed: int,
    analytic_min_n: int,
    analytic_min_cell: float,
) -> np.ndarray:
    """Inner cut points of one column against one target (the production ``edges_fayyad_irani`` output, sorted)."""
    sorter = np.argsort(x)
    xs = x[sorter]
    ys = y[sorter]
    xd, yd = _sorted_dedup(xs, ys)
    # Intentional float64: MDLP cut search at the production binning width (cut points must match supervised_binning)
    out = np.empty(max(1, xd.shape[0]), dtype=np.float64)
    n_out = np.zeros(1, dtype=np.int64)
    _mdlp_dfs(xd, yd, n_classes_total, min_split_size, max_depth, alpha, n_permutations, seed, analytic_min_n, analytic_min_cell, out, n_out)
    cuts: np.ndarray = out[: n_out[0]].copy()
    cuts.sort()
    return cuts


@njit(parallel=True, cache=True)
def _presort_columns(raw_cont: np.ndarray):
    """Sort order of every column of the (compact, continuous-only) ``raw_cont``, int32, one row per column: 4 bytes per cell, the sorted
    values are gathered per call instead of stored."""
    n = raw_cont.shape[0]
    k = raw_cont.shape[1]
    order = np.zeros((k, n), dtype=np.int32)
    for j in prange(k):
        o = np.argsort(raw_cont[:, j], kind="mergesort")
        for i in range(n):
            order[j, i] = o[i]
    return order


@njit(parallel=True, cache=True)
def honest_null_columns(
    raw_cont: np.ndarray,
    is_cont: np.ndarray,
    fixed_codes: np.ndarray,
    fixed_nbins: np.ndarray,
    y_raw: np.ndarray,
    y_codes: np.ndarray,
    n_classes_total: int,
    perms: np.ndarray,
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
) -> tuple[np.ndarray, np.ndarray]:
    """Per-permutation, per-column null MI over the pool (prange over permutations).

    Returns ``(plain, corrected)``, both ``(perms.shape[0], k)``: ``plain[b, j]`` is the plug-in MI of column ``j`` under permutation ``b``
    and ``corrected[b, j]`` the Miller-Madow corrected value over the occupied bins. A column's values depend only on the column and the
    permutation, so callers can cache them per column and take maxima over any pool (``max(0, max_j value)``).

    Pool column ``j`` is continuous where ``is_cont[j]``: its values are ``raw_cont[:, col_slot[j]]``; otherwise its observed codes are
    ``fixed_codes[:, col_slot[j]]``. Both matrices hold only their own columns (a wide pool does not get two full-width copies). ``y_raw`` are the raw target labels (integers, compact 0..C-1) that the cut search reads;
    ``y_codes`` are the screen's target codes that the MI reads. ``perms[b]`` is a row permutation.
    """
    n_perm_rows = perms.shape[0]
    n = perms.shape[1]
    k = is_cont.shape[0]
    plain = np.zeros((n_perm_rows, k), dtype=np.float64)  # Intentional float64: statistic vector (MI / null max / p-value inputs), exact permutation ties
    corrected = np.zeros((n_perm_rows, k), dtype=np.float64)  # Intentional float64: statistic vector (MI / null max / p-value inputs), exact permutation ties
    # x never changes between permutations: sort every continuous column once, so each (permutation, column) cut search only gathers the
    # permuted labels in that order instead of re-sorting the column three times (argsort + the two stable passes of the dedup).
    order = _presort_columns(raw_cont)
    for b in prange(n_perm_rows):
        rp = perms[b]
        yc_p = np.empty(n, dtype=np.int64)
        yr_p = np.empty(n, dtype=np.int64)
        for i in range(n):
            yc_p[i] = y_codes[rp[i]]
            yr_p[i] = y_raw[rp[i]]
        # The MI is evaluated on ``eval_rows`` (the screen's row subsample, or every row): the cut search above still reads the whole
        # permuted target, exactly as the observed codes were fitted on the whole target and then scored on the subsample.
        m = eval_rows.shape[0]
        ys = np.empty(m, dtype=np.int64)
        for t in range(m):
            ys[t] = yc_p[eval_rows[t]]
        ny = 0
        for t in range(m):
            if ys[t] + 1 > ny:
                ny = ys[t] + 1
        ky_eff = 0.0
        seen_y = np.zeros(ny, dtype=np.bool_)
        for t in range(m):
            seen_y[ys[t]] = True
        for c in range(ny):
            if seen_y[c]:
                ky_eff += 1.0
        for j in range(k):
            if is_cont[j]:
                sj = col_slot[j]
                ysrt = np.empty(n, dtype=np.int64)
                xsrt = np.empty(n, dtype=raw_cont.dtype)
                for i in range(n):
                    ysrt[i] = yr_p[order[sj, i]]
                    xsrt[i] = raw_cont[order[sj, i], sj]
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
                final = _supervised_edges(raw_cont[:, sj], cuts, q_fallback[j, : q_fallback_len[j]], q_refine[j, : q_refine_len[j]])
                xc = _codes_from_cuts(raw_cont[:, sj], final)
                n_x = final.shape[0] + 1
                xs = np.empty(m, dtype=np.int64)
                for t in range(m):
                    xs[t] = xc[eval_rows[t]]
            else:
                # Fixed codes are stored narrow; filled here so the two branches never share one array variable of different widths.
                fs = col_slot[j]
                n_x = fixed_nbins[j]
                xs = np.empty(m, dtype=np.int64)
                for t in range(m):
                    xs[t] = fixed_codes[eval_rows[t], fs]
            mi = _mi_codes(xs, n_x, ys, ny)
            kx_eff = float(_n_unique_codes(xs))
            plain[b, j] = mi
            corrected[b, j] = mi - (kx_eff - 1.0) * (ky_eff - 1.0) / (2.0 * m)
    return plain, corrected
