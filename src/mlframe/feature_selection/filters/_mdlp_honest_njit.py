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

import math

import numpy as np
from numba import njit, prange

# The production validated path (_mdlp_validated_split) scores splits with the incremental kernel; gains of the two kernels differ in the
# last bits, which flips the accept decision on exact ties (permutation null equal to the observed gain). Use the same kernel.
from .supervised_binning import _mdlp_best_split_inc_njit as _mdlp_best_split_njit
from ._mdlp_validated_split import _count_candidates_njit


@njit(cache=True)
def _gammaincc_nb(a: float, x: float) -> float:
    """Regularised upper incomplete gamma Q(a, x); the chi-square survival is Q(df/2, g/2). Matches scipy to ~3e-14."""
    if x <= 0.0:
        return 1.0
    gln = math.lgamma(a)
    if x < a + 1.0:
        ap = a
        s = 1.0 / a
        d = s
        for _ in range(10000):
            ap += 1.0
            d *= x / ap
            s += d
            if abs(d) < abs(s) * 1e-16:
                break
        return 1.0 - s * math.exp(-x + a * math.log(x) - gln)
    b = x + 1.0 - a
    c = 1.0 / 1e-300
    d = 1.0 / b
    h = d
    for i in range(1, 10000):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < 1e-300:
            d = 1e-300
        c = b + an / c
        if abs(c) < 1e-300:
            c = 1e-300
        d = 1.0 / d
        de = d * c
        h *= de
        if abs(de - 1.0) < 1e-16:
            break
    return math.exp(-x + a * math.log(x) - gln)


@njit(cache=True)
def _analytic_p_value(mi: float, n: int, n_classes: int) -> float:
    """G-test tail of a 2 x n_classes table (df = n_classes - 1), as ``analytic_mi_null`` computes it."""
    df = n_classes - 1
    if df <= 0 or n <= 0 or mi <= 0.0:
        return 1.0
    p = _gammaincc_nb(0.5 * df, 0.5 * 2.0 * float(n) * mi)
    return min(1.0, max(0.0, float(p)))


@njit(cache=True)
def _compact_labels(y: np.ndarray, n_classes_total: int):
    """Rank-compact labels among the classes present. Returns (compact labels, number of present classes)."""
    present = np.zeros(n_classes_total, dtype=np.bool_)
    for i in range(y.shape[0]):
        present[y[i]] = True
    rank = np.full(n_classes_total, -1, dtype=np.int64)
    k = 0
    for c in range(n_classes_total):
        if present[c]:
            rank[c] = k
            k += 1
    out = np.empty(y.shape[0], dtype=np.int64)
    for i in range(y.shape[0]):
        out[i] = rank[y[i]]
    return out, k


@njit(cache=True)
def _accept_split_serial(
    x_i: np.ndarray, y_i: np.ndarray, n_classes: int, gain: float, min_split_size: int, n_permutations: int, base_seed: int, alpha: float
) -> bool:
    """Per-node permutation test (serial twin of ``_mdlp_permutation_batch_njit``): accept when the observed gain exceeds the
    ``ceil((1 - alpha) * B) - 1`` order statistic of the null gains, seeded per permutation."""
    ni = x_i.shape[0]
    q_idx = min(math.ceil((1.0 - alpha) * n_permutations) - 1, n_permutations - 1)
    q_idx = max(0, q_idx)
    # gain > sorted_null[q_idx] holds exactly when at most (n_permutations - 1 - q_idx) draws reach the gain, so the scan stops as soon as
    # that many draws have reached it: the decision is the same as sorting all draws, and each draw is seeded on its own.
    max_reach = n_permutations - 1 - q_idx
    reach = 0
    y_perm = np.empty(ni, dtype=np.int64)
    for p in range(n_permutations):
        np.random.seed(base_seed + p)
        for i in range(ni):
            y_perm[i] = y_i[i]
        for i in range(ni - 1, 0, -1):
            j = int(np.random.randint(0, i + 1))
            tmp = y_perm[i]
            y_perm[i] = y_perm[j]
            y_perm[j] = tmp
        _, g, _, _, _ = _mdlp_best_split_njit(x_i, y_perm, n_classes, min_split_size)
        if g >= gain:
            reach += 1
            if reach > max_reach:
                return False
    return True


@njit(cache=True)
def _mdlp_dfs(
    xs: np.ndarray,
    ys: np.ndarray,
    n_classes_total: int,
    min_split_size: int,
    max_depth: int,
    alpha: float,
    n_permutations: int,
    seed: int,
    analytic_min_n: int,
    analytic_min_cell: float,
    out: np.ndarray,
    n_out: np.ndarray,
) -> None:
    """Validated split search on sorted, deduplicated (x, y), iterative (explicit stack of index ranges): numba cannot cache a recursive
    function, and every process would otherwise recompile the search. Node decisions are those of the depth-first recursion: a node's
    permutation seed depends only on the root seed, its depth and its split index, and the child labels are rank-compacted within the
    node exactly as before."""
    n_total = xs.shape[0]
    cap = 2 * max_depth + 8  # DFS pops one range and pushes two children one level deeper: the stack never holds more than max_depth + 2 ranges
    st_s = np.empty(cap, dtype=np.int64)
    st_e = np.empty(cap, dtype=np.int64)
    st_d = np.empty(cap, dtype=np.int64)
    top = 0
    st_s[0] = 0
    st_e[0] = n_total
    st_d[0] = 0
    top = 1
    while top > 0:
        top -= 1
        s = st_s[top]
        e = st_e[top]
        depth = st_d[top]
        x = xs[s:e]
        n = e - s
        if n < 2 * min_split_size or depth >= max_depth:
            continue
        y_c, n_classes_full = _compact_labels(ys[s:e], n_classes_total)
        if n_classes_full <= 1:
            continue
        best_idx, best_gain, _h, _nl, _nr = _mdlp_best_split_njit(x, y_c, n_classes_full, min_split_size)
        if best_idx < 0 or best_gain <= 0.0:
            continue
        best_split = 0.5 * (x[best_idx] + x[best_idx + 1])
        node_seed = seed + depth * 7919 + best_idx
        if n >= analytic_min_n and (float(n) / max(1.0, 2.0 * n_classes_full)) >= analytic_min_cell:
            n_candidates = _count_candidates_njit(x, y_c, min_split_size)
            p_value = min(1.0, _analytic_p_value(best_gain, n, n_classes_full) * max(1, n_candidates))
            accepted = p_value < alpha
        else:
            accepted = _accept_split_serial(x, y_c, n_classes_full, best_gain, min_split_size, n_permutations, node_seed, alpha)
        if not accepted:
            continue
        out[n_out[0]] = best_split
        n_out[0] += 1
        left_end = best_idx + 1
        st_s[top] = s
        st_e[top] = s + left_end
        st_d[top] = depth + 1
        top += 1
        st_s[top] = s + left_end
        st_e[top] = e
        st_d[top] = depth + 1
        top += 1


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


@njit(cache=True)
def _presorted_dedup(xs: np.ndarray, ys: np.ndarray, n_classes_total: int):
    """``_sorted_dedup`` for a column already sorted by x, ``ys`` the labels in that sorted order. Within each run of equal x the distinct
    labels are emitted ascending, which is the (x, y) order and duplicate removal of the two stable passes, without sorting: only the
    labels change between permutations, the order of x does not. Runs are found by comparing neighbours, so nothing per column is stored."""
    n = xs.shape[0]
    xd = np.empty(n, dtype=xs.dtype)
    yd = np.empty(n, dtype=np.int64)
    seen = np.zeros(n_classes_total, dtype=np.bool_)
    m = 0
    a = 0
    while a < n:
        e = a + 1
        while e < n and xs[e] == xs[a]:
            e += 1
        if e - a == 1:
            xd[m] = xs[a]
            yd[m] = ys[a]
            m += 1
        else:
            for i in range(a, e):
                seen[ys[i]] = True
            for c in range(n_classes_total):
                if seen[c]:
                    xd[m] = xs[a]
                    yd[m] = c
                    m += 1
                    seen[c] = False
        a = e
    return xd[:m], yd[:m]


@njit(cache=True)
def _mdlp_cuts_presorted(
    xs: np.ndarray,
    ys: np.ndarray,
    n_classes_total: int,
    min_split_size: int,
    max_depth: int,
    alpha: float,
    n_permutations: int,
    seed: int,
    analytic_min_n: int,
    analytic_min_cell: float,
) -> np.ndarray:
    """``mdlp_inner_cuts`` on a column presorted by x (see ``_presorted_dedup``); same cuts."""
    xd, yd = _presorted_dedup(xs, ys, n_classes_total)
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


@njit(cache=True)
def _codes_from_cuts(col: np.ndarray, cuts: np.ndarray) -> np.ndarray:
    """Bin code of each value: number of inner cut points <= value (``searchsorted(side='right')``)."""
    out = np.empty(col.shape[0], dtype=np.int64)
    for i in range(col.shape[0]):
        v = col[i]
        lo = 0
        hi = cuts.shape[0]
        while lo < hi:
            mid = (lo + hi) // 2
            if cuts[mid] <= v:
                lo = mid + 1
            else:
                hi = mid
        out[i] = lo
    return out


@njit(cache=True)
def _mi_codes(x: np.ndarray, n_x: int, y: np.ndarray, n_y: int) -> float:
    """Plug-in MI (nats) of two integer-coded vectors; the same contingency formula as ``_parity_fe._mi_codes_njit``."""
    n = x.shape[0]
    joint = np.zeros((n_x, n_y), dtype=np.int64)
    for i in range(n):
        joint[x[i], y[i]] += 1
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
def _n_unique_codes(x: np.ndarray) -> int:
    """Number of distinct codes (occupied bins)."""
    seen = np.zeros(int(x.max()) + 1, dtype=np.bool_)
    cnt = 0
    for i in range(x.shape[0]):
        if not seen[x[i]]:
            seen[x[i]] = True
            cnt += 1
    return cnt


@njit(cache=True)
def _supervised_edges(v: np.ndarray, cuts: np.ndarray, q_fallback: np.ndarray, q_refine: np.ndarray) -> np.ndarray:
    """Final inner edges of a supervised column, as ``compute_col_edges`` produces them: the MDLP cuts; when none were accepted, the
    unsupervised collapse fallback; otherwise the refine step (adds the fallback cuts when one bin holds more than
    ``1 - 1/(2 * fallback_nbins)`` of the rows). ``q_fallback`` / ``q_refine`` are the target-free fallback cuts, precomputed per column."""
    if cuts.shape[0] == 0:
        return q_fallback
    n = v.shape[0]
    counts = np.zeros(cuts.shape[0] + 1, dtype=np.int64)
    for i in range(n):
        lo = 0
        hi = cuts.shape[0]
        while lo < hi:
            mid = (lo + hi) // 2
            if cuts[mid] <= v[i]:
                lo = mid + 1
            else:
                hi = mid
        counts[lo] += 1
    cmax = 0
    for c in range(counts.shape[0]):
        if counts[c] > cmax:
            cmax = counts[c]
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
