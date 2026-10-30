"""Column-outer variant of ``honest_null_columns``: each column is sorted and gathered once, then every permutation reuses it.

``honest_null_columns`` runs prange over permutations and, for every (permutation, column), gathers the column's values in sorted order
from the wide ``raw_cont`` matrix: a random read with a row stride of ``k * itemsize`` bytes, ``n`` times, although the sorted values are the same
for all permutations. Here the parallel loop runs over columns, the sorted values and the fixed codes are built once per column, and the
inner loop over permutations only gathers the permuted labels. The results are identical to ``honest_null_columns`` (same operations per
(permutation, column)); ``honest_null_columns`` stays as the permutation-parallel reference.
"""

from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np
from mlframe.feature_selection.filters._mdlp_honest_bycol_stats import _bycol_perm_label_stats
from numba import njit, prange

from .supervised_binning import _mdlp_best_split_inc_njit as _mdlp_best_split_njit
from ._mdlp_honest_njit import (
    _compact_labels,
    _mdlp_cuts_presorted,
    _mi_codes,
    _n_unique_codes,
    _presort_columns,
)


def permuted_labels(y_raw: np.ndarray, y_codes: np.ndarray, perm_rows: Iterable[np.ndarray], count: int) -> "tuple[np.ndarray, np.ndarray]":
    """``(y_raw[perm], y_codes[perm])`` for the first ``count`` index vectors of ``perm_rows``, each ``(count, n)``, in the narrowest signed integer
    type that holds the labels. Rows are filled as they are drawn, so no ``(count, n)`` index matrix is ever held."""
    out = []
    sources = []
    for labels in (y_raw, y_codes):
        top = int(np.asarray(labels).max(initial=0))
        dtype = np.int8 if top < 127 else np.int16 if top < 32767 else np.int32
        out.append(np.empty((count, labels.shape[0]), dtype=dtype))
        sources.append(np.asarray(labels).astype(dtype))  # gathered in the narrow type: no wide temporary per permutation
    for b, rp in zip(range(count), perm_rows):
        out[0][b] = sources[0][rp]
        out[1][b] = sources[1][rp]
    return out[0], out[1]


@njit(cache=True)
def seeded_permutation(n: int, seed: int) -> np.ndarray:
    """The Fisher-Yates shuffle of ``range(n)`` seeded ``seed`` (numba's generator): the permutation ``b`` of the stream of ``permuted_labels_seeded`` is ``seeded_permutation(n, base_seed + b)``."""
    np.random.seed(seed % 4294967295)
    perm = np.arange(n)
    for i in range(n - 1, 0, -1):
        j = np.random.randint(0, i + 1)
        t = perm[i]
        perm[i] = perm[j]
        perm[j] = t
    return perm


@njit(parallel=True, cache=True)
def _permuted_labels_kernel(src_raw: np.ndarray, src_codes: np.ndarray, base_seed: int, out_raw: np.ndarray, out_codes: np.ndarray) -> None:
    """Row ``b`` of ``out_raw`` / ``out_codes`` is the labels read through the Fisher-Yates shuffle seeded ``base_seed + b``, one parallel task per row (independent of the thread schedule)."""
    n = src_raw.shape[0]
    for b in prange(out_raw.shape[0]):
        # the shuffle of ``seeded_permutation(n, base_seed + b)`` (the same generator calls) on int32 indices: half the footprint of the random swaps
        np.random.seed((base_seed + b) % 4294967295)
        perm = np.arange(n).astype(np.int32)
        for i in range(n - 1, 0, -1):
            j = np.random.randint(0, i + 1)
            t = perm[i]
            perm[i] = perm[j]
            perm[j] = t
        for i in range(n):
            out_raw[b, i] = src_raw[perm[i]]
            out_codes[b, i] = src_codes[perm[i]]


def permuted_labels_seeded(y_raw: np.ndarray, y_codes: np.ndarray, count: int, base_seed: int) -> "tuple[np.ndarray, np.ndarray]":
    """``(y_raw[perm], y_codes[perm])`` for ``count`` shuffles seeded ``base_seed + b``, in the narrowest signed integer type that holds the labels (as ``permuted_labels``); the shuffles are drawn in parallel
    and no index matrix is held. The permutations are a function of ``base_seed`` alone."""
    if np.array_equal(y_raw, y_codes):  # the raw labels are their own codes (a 0/1 target): one matrix serves both, half the memory and half the gathers
        top = int(np.asarray(y_raw).max(initial=0))
        dtype = np.int8 if top < 127 else np.int16 if top < 32767 else np.int32
        src = np.asarray(y_raw).astype(dtype)
        out = np.empty((count, src.shape[0]), dtype=dtype)
        _permuted_labels_kernel(src, src, int(base_seed), out, out)
        return out, out
    outs = []
    srcs = []
    for labels in (y_raw, y_codes):
        top = int(np.asarray(labels).max(initial=0))
        dtype = np.int8 if top < 127 else np.int16 if top < 32767 else np.int32
        outs.append(np.empty((count, labels.shape[0]), dtype=dtype))
        srcs.append(np.asarray(labels).astype(dtype))
    _permuted_labels_kernel(srcs[0], srcs[1], int(base_seed), outs[0], outs[1])
    return outs[0], outs[1]


from ._numba_utils import (  # helpers hosted in _numba_utils_cross_module for the cross-module njit invariant
    _cuts_tie_free,
    column_tie_runs,
    _count_ones_bits,
    dedup_label_bits,
    pack_label_rows,
    pack_sorted_labels_from_bits,
    root_decision_packed,
)

ROOT_NULL_DRAWS = 2000


@njit(parallel=True, cache=True)
def root_null_gain_threshold(labels: np.ndarray, n_classes_total: int, min_split_size: int, alpha: float, n_draws: int, seed: int) -> float:
    """The ``(1 - alpha)`` order statistic of the best-split gain of ``n_draws`` random labellings of ``len(labels)`` distinct points with the class counts
    of ``labels``. A tied-free column's root split is accepted when its gain exceeds it; every column and every permutation shares the class counts, so the
    sample is drawn once instead of once per tree. ``inf`` when fewer than two classes are present."""
    n = labels.shape[0]
    _, n_present = _compact_labels(labels, n_classes_total)
    if n_present <= 1:
        return np.inf
    xs = np.arange(n).astype(np.float64)  # Intentional float64: MDLP split search at the production binning width
    gains = np.empty(n_draws, dtype=np.float64)  # Intentional float64: statistic vector (null gains)
    for d in prange(n_draws):
        np.random.seed(seed + d)
        yp = labels.astype(np.int64)
        for i in range(n - 1, 0, -1):
            j = int(np.random.randint(0, i + 1))
            tmp = yp[i]
            yp[i] = yp[j]
            yp[j] = tmp
        y_c, k = _compact_labels(yp, n_classes_total)
        _, g, _, _, _ = _mdlp_best_split_njit(xs, y_c, k, min_split_size)
        gains[d] = g
    gains.sort()
    q_idx = max(0, min(int(np.ceil((1.0 - alpha) * n_draws)) - 1, n_draws - 1))
    return float(gains[q_idx])


@njit(cache=True)
def _codes_at_rows_nan(col, nan_rows, rows, cuts, nb_reg, out):
    """``out[t]`` = number of cuts <= ``col[rows[t]]``, or ``nb_reg`` (the column's NaN code) where the row was NaN before filling."""
    for t in range(rows.shape[0]):
        r = rows[t]
        if nan_rows[r]:
            out[t] = nb_reg
            continue
        v = col[r]
        lo = 0
        hi = cuts.shape[0]
        while lo < hi:
            mid = (lo + hi) // 2
            if cuts[mid] <= v:
                lo = mid + 1
            else:
                hi = mid
        out[t] = lo


@njit(cache=True)
def _prepare_column(raw_cont, order, nan_cont, fixed_codes, eval_rows, is_cont_j, slot, xsrt, col, xs_buf):
    """Fill the per-column buffers of ``honest_null_columns_bycol``: a continuous column's sorted values ``xsrt`` and plain values ``col``, a fixed column's codes at the evaluation rows
    ``xs_buf``. Returns ``(tie_free, has_nan_col)`` (both False for a fixed column)."""
    n = xsrt.shape[0]
    tie_free = False
    has_nan_col = False
    if is_cont_j:
        for i in range(n):
            xsrt[i] = raw_cont[order[slot, i], slot]
            col[i] = raw_cont[i, slot]
        tie_free = True
        for i in range(1, n):
            if xsrt[i] == xsrt[i - 1]:
                tie_free = False
                break
        for i in range(n):
            if nan_cont[i, slot]:
                has_nan_col = True
                break
    else:
        for t in range(eval_rows.shape[0]):
            xs_buf[t] = fixed_codes[eval_rows[t], slot]
    return tie_free, has_nan_col


@njit(cache=True)
def _supervised_edges_sorted(xsrt: np.ndarray, cuts: np.ndarray, q_fallback: np.ndarray, q_refine: np.ndarray) -> np.ndarray:
    """:func:`_supervised_edges` for a column given by its SORTED values: the bin sizes of the cuts are differences of insertion positions in ``xsrt`` (a value belongs to bin ``b`` when
    exactly ``b`` cuts are <= it, so bin ``b`` holds the values from ``cuts[b-1]`` up to, not including, ``cuts[b]``), so no per-row binary search is run. Same edges."""
    if cuts.shape[0] == 0:
        return q_fallback
    n = xsrt.shape[0]
    k = cuts.shape[0]
    cmax = 0
    prev = 0
    for b in range(k + 1):
        if b < k:
            lo = 0
            hi = n
            c = cuts[b]
            while lo < hi:
                mid = (lo + hi) // 2
                if xsrt[mid] < c:
                    lo = mid + 1
                else:
                    hi = mid
            pos = lo
        else:
            pos = n
        if pos - prev > cmax:
            cmax = pos - prev
        prev = pos
    if cmax <= (1.0 - 1.0 / (2.0 * 5)) * n:
        return cuts
    if q_refine.shape[0] == 0:
        return cuts
    merged = np.concatenate((cuts, q_refine))
    merged.sort()
    out = np.empty(merged.shape[0], dtype=np.float64)  # Intentional float64: bin-edge / cut vector, compared against values at full precision
    m = 0
    for i in range(merged.shape[0]):
        if m == 0 or merged[i] != out[m - 1]:
            out[m] = merged[i]
            m += 1
    return out[:m]


@njit(cache=True)
def _joint_from_sorted(xsrt: np.ndarray, ysrt: np.ndarray, edges: np.ndarray, n_x: int, n_y: int, joint: np.ndarray) -> None:
    """``joint[b, c]`` = rows of class ``c`` whose value has exactly ``b`` of ``edges`` at or below it, from one pass over the sorted values: the bin index only grows along the sorted order."""
    for a in range(n_x):
        for c in range(n_y):
            joint[a, c] = 0
    b = 0
    n_e = edges.shape[0]
    for i in range(xsrt.shape[0]):
        v = xsrt[i]
        while b < n_e and edges[b] <= v:
            b += 1
        joint[b, ysrt[i]] += 1


@njit(cache=True)
def _joint_from_bits(xsrt: np.ndarray, raw_bits: np.ndarray, edges: np.ndarray, n_x: int, n_y: int, joint: np.ndarray) -> None:
    """``_joint_from_sorted`` for 0/1 labels packed in ``raw_bits`` (in the sorted order of ``xsrt``): the rows of bin ``b`` are the run of sorted positions between the first value at or above edge ``b - 1`` and
    the first at or above edge ``b``, so each bin's ones are a popcount over a range."""
    n = xsrt.shape[0]
    n_e = edges.shape[0]
    start = 0
    for b in range(n_x):
        stop = n
        if b < n_e:
            lo = start
            hi = n
            while lo < hi:  # first position whose value is at or above edges[b]
                mid = (lo + hi) >> 1
                if xsrt[mid] < edges[b]:
                    lo = mid + 1
                else:
                    hi = mid
            stop = lo
        ones = _count_ones_bits(raw_bits, start, stop)
        for c in range(n_y):
            joint[b, c] = 0
        if n_y > 1:
            joint[b, 0] = (stop - start) - ones
            joint[b, 1] = ones
        else:
            joint[b, 0] = stop - start
        start = stop


@njit(cache=True)
def _mi_from_joint(joint: np.ndarray, n_x: int, n_y: int, n: int) -> float:
    """The arithmetic of ``_mi_codes`` after its joint fill (same cell order), so the value equals ``_mi_codes`` of the same joint."""
    px = joint.sum(axis=1)
    py = joint.sum(axis=0)
    mi = 0.0
    for a in range(n_x):
        if px[a] == 0:
            continue
        for b in range(n_y):
            nab = joint[a, b]
            if nab == 0:
                continue
            mi += (nab / n) * math.log((nab * n) / (px[a] * py[b]))
    return mi


@njit(cache=True)
def _occupied_rows(joint: np.ndarray, n_x: int) -> int:
    """Number of rows of ``joint`` with a nonzero count (the occupied X bins)."""
    occupied = 0
    for a in range(n_x):
        if joint[a].sum() > 0:
            occupied += 1
    return occupied


@njit(cache=True)
def _perm_cuts(
    use_precheck, tie_free, order_row, yr_b, yr_bits, raw_bits, dd_bits, ysrt, xsrt, run_start, run_end, forb, no_cuts, n_classes_total, min_split_size, max_depth, alpha, n_permutations, seed,
    analytic_min_n, analytic_min_cell,
):
    """``(cuts, root_rejected)`` of the tree of one (column, permutation). With the precheck on, the sorted 0/1 labels are packed 64 to a word (``raw_bits``), the root is judged on the packed
    deduplicated stream (``dd_bits``), and a rejected root ends the tree without the int64 label copy (``ysrt`` is then left unfilled and the caller reads the labels from ``raw_bits``); otherwise the
    sorted labels are gathered and the tree is searched."""
    n = ysrt.shape[0]
    if use_precheck:
        pack_sorted_labels_from_bits(yr_bits, order_row, raw_bits)
        m_d, n_forb = dedup_label_bits(raw_bits, n, run_start, run_end, dd_bits, forb)
        verdict = root_decision_packed(dd_bits, m_d, forb, n_forb, min_split_size, alpha, analytic_min_n, analytic_min_cell)
        if verdict == 0:
            return no_cuts, True
    for i in range(n):
        ysrt[i] = yr_b[order_row[i]]
    if tie_free:
        return _cuts_tie_free(xsrt, ysrt, n_classes_total, min_split_size, max_depth, alpha, n_permutations, seed, analytic_min_n, analytic_min_cell, np.nan), False
    return _mdlp_cuts_presorted(xsrt, ysrt, n_classes_total, min_split_size, max_depth, alpha, n_permutations, seed, analytic_min_n, analytic_min_cell), False


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
    nan_cont: np.ndarray,
    precheck: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Same result as ``honest_null_columns`` (``(plain, corrected)``, each ``(permutations, k)``), prange over columns.

    ``precheck``: for binary labels the root test of each tree is first decided on a bit-packed label stream (``root_decision_packed``, the verdict of ``root_decision_binary``); a rejected root, which is the usual outcome on a permuted label,
    ends the tree there. The decision is the one the general route reaches, so the result does not depend on the flag.

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
    # the permuted 0/1 labels of the root test, 64 to a word: the gather through a column's sorted order reads a row of n / 8 bytes
    y_bits = pack_label_rows(y_raw_perm) if (precheck and n_classes_total == 2) else np.zeros((n_perm_rows, 0), dtype=np.uint64)
    # Per permutation (not per column): the number of target codes and of occupied target codes over the evaluation rows.
    ky_eff_perm, ny_perm, same_labels = _bycol_perm_label_stats(n_perm_rows, y_codes_perm, m, eval_rows, n, y_raw_perm)
    # The root node takes its own per-permutation test (``_perm_cuts`` passes no shared root threshold), as the production recursion does: a threshold shared across permutations
    # (root_null_gain_threshold, from one permutation's labels) changes accept decisions.
    for j in prange(k):
        ysrt = np.empty(n, dtype=np.int64)
        ys = np.empty(m, dtype=np.int64)
        xs_buf = np.empty(m, dtype=np.int64)
        xsrt = np.empty(n, dtype=raw_cont.dtype)
        col = np.empty(n, dtype=raw_cont.dtype)
        tie_free, has_nan_col = _prepare_column(raw_cont, order, nan_cont, fixed_codes, eval_rows, is_cont[j], col_slot[j], xsrt, col, xs_buf)
        # A tree without cuts leaves the column on its fallback edges, which do not depend on the permutation: the codes in ``xs_buf`` and their occupied count are then kept
        # from the first such permutation instead of being rebuilt for each one (most permutations of a noise column accept no cut).
        codes_on_fallback = False
        kx_on_fallback = 0.0
        use_precheck = precheck and is_cont[j] and n_classes_total == 2
        run_start, run_end = column_tie_runs(xsrt) if use_precheck else (np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64))
        raw_bits = np.empty(((n + 63) >> 6) if use_precheck else 0, dtype=np.uint64)
        dd_bits = np.empty((((n + 63) >> 6) + 1) if use_precheck else 0, dtype=np.uint64)
        forb = np.empty(run_start.shape[0] + 1, dtype=np.int64)
        no_cuts = np.empty(0, dtype=np.float64)  # Intentional float64: bin-edge / cut vector
        for b in range(n_perm_rows):
            yr_b = y_raw_perm[b]
            yc_b = y_codes_perm[b]
            ny = ny_perm[b]
            ky_eff = ky_eff_perm[b]
            fast = False
            if is_cont[j]:
                sj = col_slot[j]
                cuts, root_rejected = _perm_cuts(
                    use_precheck, tie_free, order[sj], yr_b, y_bits[b], raw_bits, dd_bits, ysrt, xsrt, run_start, run_end, forb, no_cuts, n_classes_total, min_split_size, max_depth, alpha, n_permutations,
                    seed, analytic_min_n, analytic_min_cell,
                )
                final = _supervised_edges_sorted(xsrt, cuts, q_fallback[j, : q_fallback_len[j]], q_refine[j, : q_refine_len[j]])
                nb_reg = final.shape[0] + 1
                fast = same_labels and not has_nan_col
                if fast:
                    # every row is evaluated, the labels of the tree are the screen's codes, no NaN code: the joint is counted in one pass over the sorted values and labels
                    n_x = nb_reg
                    joint = np.zeros((n_x, ny), dtype=np.int64)
                    if root_rejected:
                        _joint_from_bits(xsrt, raw_bits, final, n_x, ny, joint)  # the labels of a rejected root stay in the packed stream; ``ysrt`` was not filled
                    else:
                        _joint_from_sorted(xsrt, ysrt, final, n_x, ny, joint)
                    mi = _mi_from_joint(joint, n_x, ny, m)
                    kx_eff = float(_occupied_rows(joint, n_x))
                    reuse_codes = False
                else:
                    reuse_codes = cuts.shape[0] == 0 and codes_on_fallback
                    if not reuse_codes:
                        _codes_at_rows_nan(col, nan_cont[:, sj], eval_rows, final, nb_reg, xs_buf)
                    n_x = nb_reg + 1 if has_nan_col else nb_reg
            else:
                n_x = fixed_nbins[j]
                reuse_codes = False
            if not fast:
                for t in range(m):
                    ys[t] = yc_b[eval_rows[t]]
                mi = _mi_codes(xs_buf, n_x, ys, ny)
                if reuse_codes:
                    kx_eff = kx_on_fallback
                else:
                    kx_eff = float(_n_unique_codes(xs_buf))
                    if is_cont[j]:
                        codes_on_fallback = cuts.shape[0] == 0
                        kx_on_fallback = kx_eff
            plain[b, j] = mi
            corrected[b, j] = mi - (kx_eff - 1.0) * (ky_eff - 1.0) / (2.0 * m)
    return plain, corrected
