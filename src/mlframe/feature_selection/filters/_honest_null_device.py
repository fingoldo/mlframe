"""Device honest null of the plain-MDLP binning: the root split of every (permutation, column) on the GPU, the rest on the host.

``honest_null_columns_bycol`` re-runs the supervised cut search of every continuous column on every permuted target (250 columns x 999 permutations = 250 000 searches of 50 000 rows in a
conjunction fit, about 95 s). Under a permuted target the root split is accepted only with probability close to the test level, and when it is not accepted the column keeps its target-free
fallback edges: its codes do not depend on the permutation at all, so its MI against the permuted target is a plain count-table MI of fixed codes. The pipeline here is therefore

1. one kernel scans the sorted column against each permuted target (one thread per column x permutation) and returns the best split's gain, its index and the number of candidate cut points,
   exactly the quantities ``_mdlp_best_split_inc_njit`` and ``_count_candidates_njit`` produce;
2. the analytic significance test of the root (``_analytic_p_value`` x candidates < alpha) marks the few (column, permutation) pairs whose root split is accepted;
3. the MI of every pair is taken from the fixed fallback codes with the bit-set count kernel (``_family_maxt_cuda_bits``);
4. the accepted pairs, and only those, are redone with the host cut search.

Applies where the root is decided analytically (``n >= analytic_min_n`` and enough rows per cell), the target has at most 16 classes and the codes at most 32 bins; otherwise the caller uses
the host kernel.
"""

from __future__ import annotations

import logging
import math

import numpy as np
from numba import njit, prange

from ._mdlp_honest_bycol_njit import _cuts_tie_free
from ._numba_utils import _codes_at_rows
from ._mdlp_honest_njit import _analytic_p_value, _mdlp_cuts_presorted, _mi_codes, _n_unique_codes, _supervised_edges
from ._kernel_types import render

logger = logging.getLogger(__name__)

MAX_CLASSES = 16
MAX_BINS = 32
XLOGX_TABLE_SIZE = 32768  # the host kernel's table size: larger counts take k * log(k) directly

_SOURCE = render(r"""
extern "C" __global__ void root_split_stats(const int* __restrict__ order, const unsigned char* __restrict__ tie, const signed char* __restrict__ ylab,
                                            const {ACC}* __restrict__ tab, const long long* __restrict__ ctot, const int K, const int n, const int B,
                                            const int Kc, const int min_split, const {ACC} h_full, {ACC}* __restrict__ out_gain, int* __restrict__ out_ncand,
                                            int* __restrict__ out_idx) {
    const long long idx = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= (long long)B * Kc) return;
    const int j = (int)(idx / B);
    const int b = (int)(idx - (long long)j * B);
    const int* ord = order + (long long)j * n;
    const unsigned char* tj = tie + (long long)j * n;
    long long cl[16], cr[16];
    for (int c = 0; c < K; ++c) { cl[c] = 0; cr[c] = ctot[c]; }
    {ACC} s_left = 0.0, s_right = 0.0;
    for (int c = 0; c < K; ++c) { const long long v = cr[c]; s_right += (v < 32768) ? tab[v] : ({ACC})v * log(({ACC})v); }
    const {ACC} n_f = ({ACC})n;
    {ACC} best_gain = -1e300;
    int best_idx = -1, ncand = 0;
    for (int i = 0; i < n - 1; ++i) {
        const int yi = ylab[(long long)ord[i] * B + b];
        const long long c_l = cl[yi], c_r = cr[yi];
        s_left += ((c_l + 1 < 32768) ? tab[c_l + 1] : ({ACC})(c_l + 1) * log(({ACC})(c_l + 1))) - ((c_l < 32768) ? tab[c_l] : ({ACC})c_l * log(({ACC})c_l));
        s_right += ((c_r - 1 < 32768) ? tab[c_r - 1] : ({ACC})(c_r - 1) * log(({ACC})(c_r - 1))) - ((c_r < 32768) ? tab[c_r] : ({ACC})c_r * log(({ACC})c_r));
        cl[yi] = c_l + 1;
        cr[yi] = c_r - 1;
        const int n_l = i + 1, n_r = n - n_l;
        if (n_l < min_split || n_r < min_split) continue;
        if (yi == (int)ylab[(long long)ord[i + 1] * B + b]) continue;
        if (tj[i]) continue;
        ++ncand;
        const {ACC} xl_l = (n_l < 32768) ? tab[n_l] : ({ACC})n_l * log(({ACC})n_l);
        const {ACC} xl_r = (n_r < 32768) ? tab[n_r] : ({ACC})n_r * log(({ACC})n_r);
        const {ACC} gain = h_full - (xl_l - s_left + xl_r - s_right) / n_f;
        if (gain > best_gain) { best_gain = gain; best_idx = i; }
    }
    out_gain[idx] = best_gain;
    out_ncand[idx] = ncand;
    out_idx[idx] = best_idx;
}
""", real=np.float32)

_SOURCE_V2 = render(r"""
extern "C" __global__ void root_split_stats_v2(const int* __restrict__ order, const unsigned char* __restrict__ tie, const signed char* __restrict__ ylab,
                                            const {ACC}* __restrict__ tab, const long long* __restrict__ ctot, const int K, const int n, const int B,
                                            const int Kc, const int min_split, const {ACC} h_full, {ACC}* __restrict__ out_gain, int* __restrict__ out_ncand,
                                            int* __restrict__ out_idx) {
    const long long idx = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= (long long)B * Kc) return;
    const int j = (int)(idx / B);
    const int b = (int)(idx - (long long)j * B);
    const int* ord = order + (long long)j * n;
    const unsigned char* tj = tie + (long long)j * n;
    long long cl[16], cr[16];
    for (int c = 0; c < K; ++c) { cl[c] = 0; cr[c] = ctot[c]; }
    {ACC} s_left = 0.0, s_right = 0.0;
    for (int c = 0; c < K; ++c) { const long long v = cr[c]; s_right += (v < 32768) ? tab[v] : ({ACC})v * log(({ACC})v); }
    const {ACC} n_f = ({ACC})n;
    {ACC} best_gain = -1e300;
    int best_idx = -1, ncand = 0;
    // The label of row i + 1 is both the boundary test of step i and the label of step i + 1: it is read once, one step ahead, so the dependent
    // ord -> ylab load chain is off the critical path of the count / table updates. The arithmetic is unchanged.
    int ycur = ylab[(long long)ord[0] * B + b];
    for (int i = 0; i < n - 1; ++i) {
        const int yi = ycur;
        ycur = ylab[(long long)ord[i + 1] * B + b];
        const long long c_l = cl[yi], c_r = cr[yi];
        s_left += ((c_l + 1 < 32768) ? tab[c_l + 1] : ({ACC})(c_l + 1) * log(({ACC})(c_l + 1))) - ((c_l < 32768) ? tab[c_l] : ({ACC})c_l * log(({ACC})c_l));
        s_right += ((c_r - 1 < 32768) ? tab[c_r - 1] : ({ACC})(c_r - 1) * log(({ACC})(c_r - 1))) - ((c_r < 32768) ? tab[c_r] : ({ACC})c_r * log(({ACC})c_r));
        cl[yi] = c_l + 1;
        cr[yi] = c_r - 1;
        const int n_l = i + 1, n_r = n - n_l;
        if (n_l < min_split || n_r < min_split) continue;
        if (yi == ycur) continue;
        if (tj[i]) continue;
        ++ncand;
        const {ACC} xl_l = (n_l < 32768) ? tab[n_l] : ({ACC})n_l * log(({ACC})n_l);
        const {ACC} xl_r = (n_r < 32768) ? tab[n_r] : ({ACC})n_r * log(({ACC})n_r);
        const {ACC} gain = h_full - (xl_l - s_left + xl_r - s_right) / n_f;
        if (gain > best_gain) { best_gain = gain; best_idx = i; }
    }
    out_gain[idx] = best_gain;
    out_ncand[idx] = ncand;
    out_idx[idx] = best_idx;
}
""", real=np.float32)


_kernel = None


def _get_kernel(cp):
    """Compile the root-split kernel once per process (no fused multiply-add: the gains must round as the host's).

    The default is the look-ahead version (``_SOURCE_V2``, same arithmetic, the next row's label read one step early); ``MLFRAME_HONEST_ROOT_SPLIT_V2=0`` selects the original ``_SOURCE``.
    """
    global _kernel
    if _kernel is None:
        import os

        if os.environ.get("MLFRAME_HONEST_ROOT_SPLIT_V2", "1") == "0":
            _kernel = cp.RawKernel(_SOURCE, "root_split_stats", options=("--fmad=false",))
        else:
            _kernel = cp.RawKernel(_SOURCE_V2, "root_split_stats_v2", options=("--fmad=false",))
    return _kernel


_SOURCE_K2 = render(r"""
#define XL(v) (((v) < 32768) ? tab[(v)] : ({ACC})(v) * log(({ACC})(v)))
// n * binary entropy of a side with m rows of which c are class 1: m ln m - c ln c - (m - c) ln(m - c), evaluated as m * h(c / m) so every term stays below n ln 2
#define ENT2(m, c) (((c) > 0.f && (c) < (m)) ? -((c) * __logf((c) / (m)) + ((m) - (c)) * __logf(((m) - (c)) / (m))) : 0.f)
// Two passes over the sorted rows. Pass 1 scores every candidate cut with the float32 entropy sums (no {ACC} arithmetic: the device has 1/32 {ACC} throughput, and the scan was bound
// by the dependency chain of the {ACC} additions) and keeps the smallest. Pass 2 evaluates the exact {ACC} gain only for the candidates whose {REAL} score is within the {REAL} error
// bound (0.5 nats of row-weighted entropy; the {REAL} error of one score is about 0.02 at 50 000 rows) of that minimum, with the host kernel's tie rule (the first strictly greater gain wins).
extern "C" __global__ void root_split_stats_k2(const int* __restrict__ order, const unsigned char* __restrict__ tie, const unsigned char* __restrict__ ybits,
                                               const {ACC}* __restrict__ tab, const long long* __restrict__ ctot, const int n, const int B,
                                               const int Kc, const int min_split, const {ACC} h_full, const int words, {ACC}* __restrict__ out_gain,
                                               int* __restrict__ out_ncand, int* __restrict__ out_idx) {
    const long long idx = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= (long long)B * Kc) return;
    const int j = (int)(idx / B);
    const int b = (int)(idx - (long long)j * B);
    const int* ord = order + (long long)j * n;
    const unsigned char* tj = tie + (long long)j * n;
    const unsigned int* yb = (const unsigned int*)ybits;
    const {ACC} n_f = ({ACC})n;
    const int TILE = 8;
    int ybuf[TILE + 1];
    {REAL} w_min = 3.0e38f;
    int ncand = 0;
    for (int pass = 0; pass < 2; ++pass) {
        long long l0 = 0, l1 = 0, r0 = ctot[0], r1 = ctot[1];
        {ACC} best_gain = -1e300;
        int best_idx = -1;
        for (int i0 = 0; i0 < n - 1; i0 += TILE) {
            const int cnt = min(TILE + 1, n - i0);
#pragma unroll
            for (int t = 0; t < TILE + 1; ++t) {
                if (t < cnt) { const int pos = ord[i0 + t]; ybuf[t] = (yb[(long long)(pos >> 5) * B + b] >> (pos & 31)) & 1; }
            }
            for (int t = 0; t < TILE && i0 + t < n - 1; ++t) {
                const int i = i0 + t;
                const int yi = ybuf[t];
                if (yi) { ++l1; --r1; } else { ++l0; --r0; }
                const int n_l = i + 1, n_r = n - n_l;
                if (n_l >= min_split && n_r >= min_split && yi != ybuf[t + 1] && !tj[i]) {
                    const {REAL} w_f = ENT2(({REAL})n_l, ({REAL})l1) + ENT2(({REAL})n_r, ({REAL})r1);
                    if (pass == 0) {
                        ++ncand;
                        if (w_f < w_min) w_min = w_f;
                    } else if (w_f <= w_min + 0.5f) {
                        const {ACC} gain = h_full - (XL(n_l) - (XL(l0) + XL(l1)) + XL(n_r) - (XL(r0) + XL(r1))) / n_f;
                        if (gain > best_gain) { best_gain = gain; best_idx = i; }
                    }
                }
            }
        }
        if (pass == 1) {
            out_gain[idx] = best_gain;
            out_ncand[idx] = ncand;
            out_idx[idx] = best_idx;
        }
    }
}
""", real=np.float32)

# bench-attempt-rejected (2026-11-01, GTX 1050 Ti, 60 columns x 200 permutations x 100000 rows, 559 ms): tile sizes 4 / 16 / 32 (570 / 581 / 730 ms) and block sizes 64 / 256 / 512 (562 / 550 / 553 ms), outputs
# identical. The kernel does ~2.1 G scan steps per second whatever the tiling: it is bound by the random gather of label bits (the (words, permutations) table is 2.5 MB against a 1 MB L2), not by
# occupancy or the prefetch depth. A leaner version needs the labels in sorted order without a per-step gather, which costs the same gather once per (column, permutation).
# Cost split of the 568 ms (variants with the term removed, outputs wrong, timing only): entropy arithmetic ~100 ms (18%), label-bit gather ~295 ms (52%), loop skeleton ~172 ms (30%). Running the permutations in
# chunks so each launch's label table fits the 1 MB L2 did not help (chunks of 200 / 100 / 50 / 25 permutations: 594 / 571 / 810 / 1461 ms): the cost is the 37.5 M random line fetches, not L2 capacity.
_kernel_k2 = None


def _get_kernel_k2(cp):
    """Binary-target variant: class counts in registers, labels read as bits."""
    global _kernel_k2
    if _kernel_k2 is None:
        _kernel_k2 = cp.RawKernel(_SOURCE_K2, "root_split_stats_k2", options=("--fmad=false",))
    return _kernel_k2


def _pack_bits(y_perm: np.ndarray) -> np.ndarray:
    """``(permutations, ceil(n / 32))`` uint32 rows of the 0 / 1 labels, bit ``i % 32`` of word ``i // 32``."""
    b_count, n = int(y_perm.shape[0]), int(y_perm.shape[1])
    words = (n + 31) // 32
    out = np.empty((b_count, words), dtype=np.uint32)
    # the rows are packed a block at a time: the boolean mask, its uint8 copy and the padded copy of the whole matrix were three (permutations, n) arrays at once
    # (1.5 GB for 500 permutations of a million rows)
    step = max(1, (64 << 20) // max(1, words * 32))
    for r in range(0, b_count, step):
        block = y_perm[r : r + step]
        buf = np.zeros((block.shape[0], words * 32), dtype=np.uint8)
        buf[:, :n] = block == 1
        out[r : r + step] = np.packbits(buf, axis=1, bitorder="little").view(np.uint32)
    return out  # (permutations, words)


@njit(cache=True, parallel=True)
def _tie_flags(raw_cont: np.ndarray, order: np.ndarray) -> np.ndarray:
    """``(k, n)`` uint8: 1 where the value at sorted position ``i`` equals the one at ``i + 1``."""
    k = order.shape[0]
    n = order.shape[1]
    out = np.zeros((k, n), dtype=np.uint8)
    for j in prange(k):
        for i in range(n - 1):
            if raw_cont[order[j, i], j] == raw_cont[order[j, i + 1], j]:
                out[j, i] = 1
    return out


@njit(cache=True, parallel=True)
def _accepted(gain: np.ndarray, ncand: np.ndarray, idx: np.ndarray, n: int, n_classes: int, alpha: float) -> np.ndarray:
    """``(permutations, columns)`` bool: the root split is accepted by the analytic test (best gain positive, p-value x candidates below alpha)."""
    b_count = gain.shape[0]
    k = gain.shape[1]
    out = np.zeros((b_count, k), dtype=np.bool_)
    for b in prange(b_count):
        for j in range(k):
            g = gain[b, j]
            if idx[b, j] < 0 or g <= 0.0:
                continue
            p = min(1.0, _analytic_p_value(g, n, n_classes) * max(1, ncand[b, j]))
            out[b, j] = p < alpha
    return out


@njit(cache=True, parallel=True)
def _redo_accepted(
    raw_cont,
    order,
    tie_free,
    y_raw_perm,
    y_codes_perm,
    items_b,
    items_j,
    col_slot,
    n_classes_total,
    min_split_size,
    max_depth,
    alpha,
    n_permutations,
    seed,
    analytic_min_n,
    analytic_min_cell,
    q_fallback,
    q_fallback_len,
    q_refine,
    q_refine_len,
    eval_rows,
    plain,
    corrected,
    nan_cont,
):
    """Host cut search and MI for the listed (permutation, column) pairs, written into ``plain`` / ``corrected`` (the body of ``honest_null_columns_bycol`` for one pair)."""
    n = raw_cont.shape[0]
    m = eval_rows.shape[0]
    for t in prange(items_b.shape[0]):
        b = items_b[t]
        j = items_j[t]
        sj = col_slot[j]
        xsrt = np.empty(n, dtype=raw_cont.dtype)
        ysrt = np.empty(n, dtype=np.int64)
        col = np.empty(n, dtype=raw_cont.dtype)
        for i in range(n):
            xsrt[i] = raw_cont[order[sj, i], sj]
            ysrt[i] = y_raw_perm[b, order[sj, i]]
            col[i] = raw_cont[i, sj]
        if tie_free[sj]:
            cuts = _cuts_tie_free(xsrt, ysrt, n_classes_total, min_split_size, max_depth, alpha, n_permutations, seed, analytic_min_n, analytic_min_cell, np.nan)
        else:
            cuts = _mdlp_cuts_presorted(xsrt, ysrt, n_classes_total, min_split_size, max_depth, alpha, n_permutations, seed, analytic_min_n, analytic_min_cell)
        final = _supervised_edges(col, cuts, q_fallback[j, : q_fallback_len[j]], q_refine[j, : q_refine_len[j]])
        xs = np.empty(m, dtype=np.int64)
        _codes_at_rows(col, eval_rows, final, xs)
        nb_reg = final.shape[0] + 1
        has_nan = False
        for i in range(n):
            if nan_cont[i, sj]:
                has_nan = True
                break
        if has_nan:
            for r in range(m):
                if nan_cont[eval_rows[r], sj]:
                    xs[r] = nb_reg
        ys = np.empty(m, dtype=np.int64)
        ny = 0
        for r in range(m):
            v = y_codes_perm[b, eval_rows[r]]
            ys[r] = v
            if v + 1 > ny:
                ny = v + 1
        seen = np.zeros(ny, dtype=np.bool_)
        for r in range(m):
            seen[ys[r]] = True
        ky = 0.0
        for c in range(ny):
            if seen[c]:
                ky += 1.0
        mi = _mi_codes(xs, nb_reg + 1 if has_nan else nb_reg, ys, ny)
        plain[b, j] = mi
        corrected[b, j] = mi - (float(_n_unique_codes(xs)) - 1.0) * (ky - 1.0) / (2.0 * m)


@njit(cache=True, parallel=True)
def _fallback_codes(raw_cont, is_cont, fixed_codes, fixed_nbins, col_slot, q_fallback, q_fallback_len, eval_rows, nan_cont):
    """``(codes, bins, occupied bins)``: the codes of every column at the evaluation rows when no cut is accepted (``(columns, m)`` int16).

    A row that was NaN before the median fill keeps the column's NaN code ``nb_reg`` (one past the regular bins), and a column with any NaN has ``nb_reg + 1`` bins, as in the host kernel."""
    k = is_cont.shape[0]
    m = eval_rows.shape[0]
    codes = np.zeros((k, m), dtype=np.int16)
    nx = np.ones(k, dtype=np.int64)
    kx = np.ones(k, dtype=np.int64)
    for j in prange(k):
        xs = np.empty(m, dtype=np.int64)
        if is_cont[j]:
            sj = col_slot[j]
            col = np.empty(raw_cont.shape[0], dtype=raw_cont.dtype)
            for i in range(raw_cont.shape[0]):
                col[i] = raw_cont[i, sj]
            edges = q_fallback[j, : q_fallback_len[j]]
            _codes_at_rows(col, eval_rows, edges, xs)
            nb_reg = edges.shape[0] + 1
            has_nan = False
            for i in range(raw_cont.shape[0]):
                if nan_cont[i, sj]:
                    has_nan = True
                    break
            if has_nan:
                for r in range(m):
                    if nan_cont[eval_rows[r], sj]:
                        xs[r] = nb_reg
            nx[j] = nb_reg + 1 if has_nan else nb_reg
        else:
            fs = col_slot[j]
            for r in range(m):
                xs[r] = fixed_codes[eval_rows[r], fs]
            nx[j] = fixed_nbins[j]
        for r in range(m):
            codes[j, r] = xs[r]
        kx[j] = _n_unique_codes(xs)
    return codes, nx, kx


def eligible(n: int, n_classes_total: int, analytic_min_n: int, analytic_min_cell: float) -> bool:
    """The root split is decided analytically (enough rows, enough rows per cell) and the target has few enough classes for the kernel."""
    return bool(n >= int(analytic_min_n) and (float(n) / max(1.0, 2.0 * n_classes_total)) >= float(analytic_min_cell) and 2 <= n_classes_total <= MAX_CLASSES)


def _entropy(counts: np.ndarray, n: int) -> float:
    """Shannon entropy in nats of the class counts, in the host kernel's arithmetic order."""
    h = 0.0
    n_f = float(n)
    for c in counts:
        if c > 0:
            p = c / n_f
            h -= p * math.log(p)
    return h


def honest_null_columns_device(
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
) -> tuple[np.ndarray, np.ndarray] | None:
    """``(plain, corrected)`` as ``honest_null_columns_bycol`` returns them, or ``None`` when the device path does not apply (see the module docstring)."""
    try:
        import cupy as cp

        from ._family_maxt_cuda_bits import DeviceBitsets, mi_matrix_bits_device
        from ._family_maxt_null_v2_njit import bitset_groups, label_bitsets_batch, xlogx_table
    except Exception as e:
        logger.warning("Device honest-null imports failed (%s: %s); using the CPU null path", type(e).__name__, e)
        return None
    n_perm, n = y_raw_perm.shape
    k = is_cont.shape[0]
    m = eval_rows.shape[0]
    kc = raw_cont.shape[1] if bool(is_cont.any()) else 0
    if not eligible(n, n_classes_total, analytic_min_n, analytic_min_cell):
        return None
    codes, nx, kx = _fallback_codes(raw_cont, is_cont, fixed_codes, fixed_nbins, col_slot, q_fallback, q_fallback_len, eval_rows, nan_cont)
    if int(nx.max()) > MAX_BINS:
        return None
    # the permuted codes keep their own narrow integer type (int8 for a binary target): widening them to int64 made an 8x copy of the (permutations, rows) matrix
    ys = y_codes_perm if (eval_rows.shape[0] == y_codes_perm.shape[1] and bool(np.array_equal(eval_rows, np.arange(eval_rows.shape[0])))) else np.ascontiguousarray(y_codes_perm[:, eval_rows])
    ny = int(ys.max()) + 1
    if ny > MAX_CLASSES:
        return None
    order = np.zeros((0, n), dtype=np.int32)
    tie = np.zeros((0, n), dtype=np.uint8)
    accepted_c = np.zeros((n_perm, 0), dtype=np.bool_)
    try:
        from mlframe.feature_selection.filters._honest_null_accept import _continuous_accept_block
        accepted_c, order, tie = _continuous_accept_block(kc, raw_cont, y_raw_perm, n_classes_total, n, xlogx_table, cp, n_perm, min_split_size, alpha, accepted_c, order, tie)
        words = (m + 63) // 64
        cls = label_bitsets_batch(ys, ny, words)
        prep = bitset_groups(np.ascontiguousarray(codes), nx)
        if ys is y_codes_perm:
            # every row of the full permuted-label matrix is a permutation of the same labels: one class-count vector serves all (256 serial bincounts of 2000000 labels were 2.4 s of a 14 s call)
            counts_b = np.broadcast_to(np.bincount(ys[0], minlength=ny), (n_perm, ny))
        else:
            counts_b = np.stack([np.bincount(ys[b], minlength=ny) for b in range(n_perm)])
        tab_m = xlogx_table(m)
        const_b = tab_m[m] - tab_m[counts_b].sum(axis=1)
        mi = mi_matrix_bits_device(DeviceBitsets(cp, prep, nx, m), cls, const_b)
    except Exception as exc:
        logger.warning("device honest-null failed (%s: %s); falling back to the host path", type(exc).__name__, exc)
        return None
    ky = (counts_b > 0).sum(axis=1).astype(np.float64)  # Intentional float64: effective class counts of the Miller-Madow correction
    plain = np.ascontiguousarray(mi, dtype=np.float64)  # Intentional float64: statistic vector (MI / null max / p-value inputs), exact permutation ties
    corrected = plain - (kx.astype(np.float64)[None, :] - 1.0) * (ky[:, None] - 1.0) / (2.0 * m)  # Intentional float64: Miller-Madow corrected MI
    if kc:
        accepted = np.zeros((n_perm, k), dtype=np.bool_)
        cont_cols = np.flatnonzero(is_cont)
        accepted[:, cont_cols] = accepted_c[:, col_slot[cont_cols]]
        items_b, items_j = np.nonzero(accepted)
        if items_b.size:
            tie_free = np.array([not bool(tie[s].any()) for s in range(kc)], dtype=np.bool_)
            _redo_accepted(
                raw_cont,
                order,
                tie_free,
                y_raw_perm,
                y_codes_perm,
                items_b.astype(np.int64),
                items_j.astype(np.int64),
                col_slot,
                n_classes_total,
                min_split_size,
                max_depth,
                alpha,
                n_permutations,
                seed,
                analytic_min_n,
                analytic_min_cell,
                q_fallback,
                q_fallback_len,
                q_refine,
                q_refine_len,
                eval_rows,
                plain,
                corrected,
                nan_cont,
            )
    return plain, corrected
