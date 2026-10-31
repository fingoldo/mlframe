"""@njit helpers formerly defined in sibling filters modules and imported across them.

Hosted here so every cross-module @njit import goes through ``_numba_utils``. Re-exported by ``_numba_utils``.
"""

from __future__ import annotations

import math
import numpy as np
from numba import njit, prange

from ._internals import _entropy_from_counts_njit


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
def _analytic_p_value(mi: float, n: int, n_classes: int) -> float:
    """G-test tail of a 2 x n_classes table (df = n_classes - 1), as ``analytic_mi_null`` computes it."""
    df = n_classes - 1
    if df <= 0 or n <= 0 or mi <= 0.0:
        return 1.0
    p = _gammaincc_nb(0.5 * df, 0.5 * 2.0 * float(n) * mi)
    return min(1.0, max(0.0, float(p)))


@njit(cache=True)
def _cmi_from_counts(counts: np.ndarray, n: int, min_rows: int) -> float:
    """Sum over strata with at least ``min_rows`` rows of (m / n) * plug-in MI (nats) of that stratum's x-y table, clipped at zero as scikit-learn does."""
    n_s, n_x, n_y = counts.shape
    total = 0.0
    px = np.zeros(n_x, dtype=np.int64)
    py = np.zeros(n_y, dtype=np.int64)
    for sv in range(n_s):
        m = 0
        for a in range(n_x):
            px[a] = 0
        for b in range(n_y):
            py[b] = 0
        for a in range(n_x):
            for b in range(n_y):
                c = counts[sv, a, b]
                px[a] += c
                py[b] += c
                m += c
        if m < min_rows:
            continue
        mi = 0.0
        for a in range(n_x):
            if px[a] == 0:
                continue
            for b in range(n_y):
                c = counts[sv, a, b]
                if c == 0:
                    continue
                mi += (c / m) * math.log((c * m) / (px[a] * py[b]))
        if mi > 0.0:
            total += (m / n) * mi
    return total


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
def _conditional_shuffle_within_strata(
    classes_x2_safe: np.ndarray,
    classes_y: np.ndarray,
    n_y_classes: int,
    base_seed: int,
) -> None:
    """Conditional permutation: shuffle ``classes_x2_safe`` IN PLACE, restricting the shuffle to within each stratum of ``classes_y``.

    This is the correct null distribution for testing ``H0: X1 ⊥ X2 | Y`` (no synergy beyond marginals). Plain shuffle-Y tests ``H0: Y ⊥ (X1, X2)`` (joint independence) instead.

    Implementation: for each Y-stratum, collect the indices where ``classes_y[i] == c``, Fisher-Yates shuffle the corresponding slice of ``classes_x2_safe`` in place.
    Per Anderson & ter Braak 2003 ("Permutation tests for multi-factorial analysis of variance").

    Preserves ``P(X2 | Y)`` - so each marginal ``I(X2; Y)`` is
    unchanged under the shuffle, but the conditional ``I(X1; X2 | Y)``
    is broken. The orchestrator combines this with three-MI calls
    above to compute II_perm under the conditional null.

    The Fisher-Yates RNG is a seeded inline LCG (same PCG-style step as ``_count_nfailed_joint_indep_prange``) so the permutation is reproducible at a fixed
    ``base_seed`` and never touches numpy's process-global RNG.
    """
    n = len(classes_y)
    state = np.uint64(base_seed) * np.uint64(2654435761) + np.uint64(1)
    # For each Y class, gather positions and Fisher-Yates within the slice.
    for c in range(n_y_classes):
        # Collect indices manually (numba doesn't support boolean masks
        # for in-place writes the same way numpy does).
        positions = np.empty(n, dtype=np.int64)
        pos_count = 0
        for i in range(n):
            if classes_y[i] == c:
                positions[pos_count] = i
                pos_count += 1
        # Fisher-Yates shuffle within the stratum
        for idx in range(pos_count - 1, 0, -1):
            state = state * np.uint64(6364136223846793005) + np.uint64(1442695040888963407)
            j = int(state >> np.uint64(33)) % (idx + 1)
            a = positions[idx]
            b = positions[j]
            tmp = classes_x2_safe[a]
            classes_x2_safe[a] = classes_x2_safe[b]
            classes_x2_safe[b] = tmp


@njit(nogil=True, cache=True)
def _count_candidates_njit(x_sorted: np.ndarray, y_sorted: np.ndarray, min_split_size: int) -> int:
    """Count class-boundary candidate cut points a node's scan actually considers (mirrors the skip
    conditions inside ``_mdlp_best_split_njit`` exactly, without doing the entropy work) - needed for
    the Bonferroni multiplicity correction on the analytic significance path."""
    n = x_sorted.shape[0]
    count = 0
    for i in range(n - 1):
        n_l = i + 1
        n_r = n - n_l
        if n_l < min_split_size or n_r < min_split_size:
            continue
        if y_sorted[i] == y_sorted[i + 1]:
            continue
        if x_sorted[i] == x_sorted[i + 1]:
            continue
        count += 1
    return count


@njit(cache=True)
def _cuts_tie_free(xs, ys, n_classes_total, min_split_size, max_depth, alpha, n_permutations, seed, analytic_min_n, analytic_min_cell, root_crit):
    """``_mdlp_cuts_presorted`` for a column without tied values: the sorted points are already the deduplicated (x, y) pairs, so nothing is copied, and
    the cut buffer holds the most cuts a tree of ``max_depth`` can accept instead of one slot per row."""
    out = np.empty(min(xs.shape[0], (1 << max_depth)) + 1, dtype=np.float64)  # Intentional float64: MDLP cut points at the production binning width
    n_out = np.zeros(1, dtype=np.int64)
    _mdlp_dfs(xs, ys, n_classes_total, min_split_size, max_depth, alpha, n_permutations, seed, analytic_min_n, analytic_min_cell, out, n_out, root_crit)
    cuts = out[: n_out[0]].copy()
    cuts.sort()
    return cuts


@njit(cache=True)
def _edge_value(s: np.ndarray, q: float) -> float:
    """``np.quantile(s_unsorted, q)`` (method ``linear``) read off the sorted copy ``s``, in numpy's arithmetic order."""
    n = s.shape[0]
    virtual = (n - 1) * q
    lo = np.floor(virtual)
    if virtual >= n - 1:
        i0 = n - 1
        i1 = n - 1
    elif virtual < 0:
        i0 = 0
        i1 = 0
    else:
        i0 = int(lo)
        i1 = i0 + 1
    gamma = virtual - i0
    a = float(s[i0])
    b = float(s[i1])
    diff = b - a
    out = a + diff * gamma
    if gamma >= 0.5:
        out = b - diff * (1.0 - gamma)
    return float(out)


@njit(cache=True)
def _full_conditional_shuffle_ipf(
    classes_x2_safe: np.ndarray,
    classes_x1: np.ndarray,
    classes_y: np.ndarray,
    n_x1_classes: int,
    n_y_classes: int,
    base_seed: int,
) -> None:
    """Full conditional permutation via IPF-style double-stratification. Shuffle X2 within strata of (X1, Y) so that P(X2 | X1, Y) is preserved on average across
    permutations - the strictest synergy null that holds both marginals fixed while breaking X1-X2 conditional dependence.

    Reference: Patefield 1981 (R x C contingency tables with fixed marginals); Anderson & ter Braak 2003 (multi-factorial permutation).

    Implementation: for each unique (X1, Y) stratum, Fisher-Yates shuffle classes_x2_safe restricted to rows in that stratum.
    The Fisher-Yates RNG is a seeded inline LCG (same PCG-style step as ``_count_nfailed_joint_indep_prange``) so the permutation is reproducible at a fixed
    ``base_seed`` and never touches numpy's process-global RNG.
    """
    n = len(classes_y)
    state = np.uint64(base_seed) * np.uint64(2654435761) + np.uint64(1)
    for cx1 in range(n_x1_classes):
        for cy in range(n_y_classes):
            # Gather indices in this (X1, Y) stratum
            positions = np.empty(n, dtype=np.int64)
            pos_count = 0
            for i in range(n):
                if classes_x1[i] == cx1 and classes_y[i] == cy:
                    positions[pos_count] = i
                    pos_count += 1
            # Fisher-Yates within stratum
            for idx in range(pos_count - 1, 0, -1):
                state = state * np.uint64(6364136223846793005) + np.uint64(1442695040888963407)
                j = int(state >> np.uint64(33)) % (idx + 1)
                a = positions[idx]
                b = positions[j]
                tmp = classes_x2_safe[a]
                classes_x2_safe[a] = classes_x2_safe[b]
                classes_x2_safe[b] = tmp


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
    root_crit: float = np.nan,
) -> None:
    """Validated split search on sorted, deduplicated (x, y), iterative (explicit stack of index ranges): numba cannot cache a recursive
    function, and every process would otherwise recompile the search. Node decisions are those of the depth-first recursion: a node's
    permutation seed depends only on the root seed, its depth and its split index, and the child labels are rank-compacted within the
    node exactly as before.

    ``root_crit`` (finite) replaces the permutation test of the root node by ``best_gain > root_crit``: for a column without tied values the null
    gain of a random labelling depends only on the labels' class counts, so the caller draws one large null sample for all columns."""
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
        if n_classes_total == 2:
            # Two classes: the labels are already 0 / 1, a node holding one class is the split kernel's h_full <= 0 exit, and the kernel counts the candidates the analytic path needs,
            # so neither the label compaction nor the separate candidate pass is run.
            y_c = ys[s:e]
            n_classes_full = 2
            best_idx, best_gain, _h, _nl, _nr, n_cand_fused = _mdlp_best_split_binary_counted_njit(x, y_c, min_split_size)
        else:
            y_c, n_classes_full = _compact_labels(ys[s:e], n_classes_total)
            if n_classes_full <= 1:
                continue
            best_idx, best_gain, _h, _nl, _nr = _mdlp_best_split_njit(x, y_c, n_classes_full, min_split_size)
            n_cand_fused = -1
        if best_idx < 0 or best_gain <= 0.0:
            continue
        best_split = 0.5 * (x[best_idx] + x[best_idx + 1])
        node_seed = seed + depth * 7919 + best_idx
        if n >= analytic_min_n and (float(n) / max(1.0, 2.0 * n_classes_full)) >= analytic_min_cell:
            n_candidates = n_cand_fused if n_cand_fused >= 0 else _count_candidates_njit(x, y_c, min_split_size)
            p_value = min(1.0, _analytic_p_value(best_gain, n, n_classes_full) * max(1, n_candidates))
            accepted = p_value < alpha
        elif depth == 0 and root_crit == root_crit:
            accepted = best_gain > root_crit
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
def _mi_codes_njit(x_codes: np.ndarray, n_x: int, y_codes: np.ndarray, n_y: int) -> float:
    """Plug-in MI (nats) of two integer-coded vectors from one contingency pass; equals ``sklearn.metrics.mutual_info_score``."""
    n = x_codes.shape[0]
    joint = np.zeros((n_x, n_y), dtype=np.int64)
    for i in range(n):
        joint[x_codes[i], y_codes[i]] += 1
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
            mi += (nab / n) * np.log((nab * n) / (px[a] * py[b]))
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


@njit(cache=True, parallel=True)
def corrected_mi_best(
    codes_p: np.ndarray, nb_p: np.ndarray, fixed_codes: np.ndarray, fixed_nbins: np.ndarray, cols: np.ndarray, yc: np.ndarray, ny: int, ky_eff: float, n_eff: int, rows: np.ndarray,
) -> float:
    """Largest Miller-Madow corrected plug-in MI with ``yc`` over the columns ``cols``, parallel over columns: a column below ``codes_p.shape[1]`` is read from ``codes_p`` (the codes re-binned on a
    permuted target), any other from ``fixed_codes`` (columns binned without the target); both are read at ``rows`` (``yc`` is the target at those rows); ``0`` when none is positive."""
    k = cols.shape[0]
    corr = np.zeros(k, dtype=np.float64)  # Intentional float64: statistic vector (MI / null max / p-value inputs), exact permutation ties
    for t in prange(k):
        c = cols[t]
        x = np.empty(rows.shape[0], dtype=np.int64)
        if c < codes_p.shape[1]:
            for i in range(rows.shape[0]):
                x[i] = codes_p[rows[i], c]
            kx = nb_p[c]
        else:
            for i in range(rows.shape[0]):
                x[i] = fixed_codes[rows[i], c]
            kx = fixed_nbins[c]
        mi = _mi_codes_njit(x, kx, yc, ny)
        kx_eff = float(_n_unique_codes(x))
        corr[t] = mi - (kx_eff - 1.0) * (ky_eff - 1.0) / (2.0 * n_eff)
    best = 0.0
    for t in range(k):
        if corr[t] > best:
            best = corr[t]
    return best


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


@njit(nogil=True, cache=True)
def _relevance_mi_1var_fused(
    factors_data: np.ndarray,
    ix: int,
    nb_x: int,
    classes_y: np.ndarray,
    freqs_y: np.ndarray,
) -> tuple:
    """Fused single-variable raw MI I(X_ix; Y) + occupied-x-bin count in ONE O(n) pass.

    Returns ``(mi, bx)`` where ``mi`` is BIT-IDENTICAL to
    ``compute_mi_from_classes(*merge_vars(factors_data, [ix], ...)[:2], classes_y, freqs_y)``
    and ``bx`` equals the pruned ``len(freqs_x)`` (occupied x-bins) that
    ``merge_vars`` returns - exactly the two quantities the ANALYTIC-null branch of
    ``mi_direct`` consumes (``_ax_freqs.shape[0]`` and the MI). The legacy path built the
    length-n ``classes_x`` relabel array via ``merge_vars`` (one O(n) accumulate pass plus a
    second O(n) lookup-remap pass when any x-bin is empty) and then walked all n samples AGAIN
    inside ``compute_mi_from_classes`` to build the joint histogram - and the analytic branch
    DISCARDS ``classes_x`` entirely (it needs only ``len(freqs_x)`` and the MI scalar). This
    kernel builds the ``(nb_x, K_y)`` joint histogram directly from ``factors_data[:, ix]`` and
    ``classes_y`` in a SINGLE row pass, then derives ``freqs_x`` from the row sums - eliminating
    the separate ``merge_vars(x)`` pass + its ``final_classes``/lookup allocations. Mirrors the
    ``joint_freqs_2var`` / ``joint_entropy_2var`` pruned-fast-path wins on the DCD pairwise path.

    BIT-IDENTITY to ``compute_mi_from_classes(merge_vars(...))`` (raw MI, no SU/MM):
      * ``merge_vars`` renumbers OCCUPIED x-bins densely in ASCENDING original-bin order, so
        ``compute_mi_from_classes`` visits x-classes ``i in range(len(freqs_x))`` in that same
        ascending order. Iterating ``i in range(nb_x)`` here and SKIPPING empty rows
        (``rowcount == 0``) visits the identical occupied bins in the identical order.
      * ``prob_x = rowcount / n`` reproduces ``merge_vars``'s ``freqs / n_rows`` (int64 count,
        true division); ``prob_y = freqs_y[j]`` is passed through unchanged; and
        ``jf = jc * inv_n`` with ``inv_n = 1.0 / n`` matches ``compute_mi_from_classes`` exactly
        - same operands, same order, so the ``jf*log(jf/(prob_x*prob_y))`` terms accumulate
        bit-for-bit.
      * Empty x-bins (and empty joint cells) contribute nothing to MI in EITHER path, so pruning
        vs skipping is numerically inert.
    """
    n = factors_data.shape[0]
    K_y = len(freqs_y)
    if n == 0:
        return 0.0, 0
    joint = np.zeros((nb_x, K_y), dtype=np.int64)
    for k in range(n):
        joint[factors_data[k, ix], classes_y[k]] += 1
    inv_n = 1.0 / n
    total = 0.0
    bx = 0
    for i in range(nb_x):
        rowcount = 0
        for j in range(K_y):
            rowcount += joint[i, j]
        if rowcount == 0:
            continue
        bx += 1
        prob_x = rowcount / n
        for j in range(K_y):
            jc = joint[i, j]
            if jc != 0:
                prob_y = freqs_y[j]
                jf = jc * inv_n
                total += jf * math.log(jf / (prob_x * prob_y))
    return total, bx


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


@njit(cache=True)
def _select_in_place(a: np.ndarray, lo: int, hi: int, k: int) -> None:
    """Reorder ``a[lo:hi + 1]`` so that ``a[k]`` holds the value a full sort would put there (smaller or equal values before it, larger or equal after)."""
    while hi > lo:
        mid = (lo + hi) // 2
        x, y, z = a[lo], a[mid], a[hi]
        pivot = max(min(x, y), min(max(x, y), z))
        i = lo
        j = hi
        while i <= j:
            while a[i] < pivot:
                i += 1
            while a[j] > pivot:
                j -= 1
            if i <= j:
                t = a[i]
                a[i] = a[j]
                a[j] = t
                i += 1
                j -= 1
        if k <= j:
            hi = j
        elif k >= i:
            lo = i
        else:
            return


@njit(cache=True)
def _multi_select_in_place(a: np.ndarray, ks: np.ndarray) -> None:
    """Put the order statistics at the sorted distinct positions ``ks`` in place, by selecting the middle position and then each side on its own range."""
    stack = np.empty((128, 4), dtype=np.int64)
    stack[0, 0] = 0
    stack[0, 1] = a.shape[0] - 1
    stack[0, 2] = 0
    stack[0, 3] = ks.shape[0] - 1
    top = 1
    while top > 0:
        top -= 1
        lo, hi, kl, kh = stack[top, 0], stack[top, 1], stack[top, 2], stack[top, 3]
        if kl > kh or hi <= lo:
            continue
        km = (kl + kh) // 2
        pos = ks[km]
        _select_in_place(a, lo, hi, pos)
        stack[top, 0], stack[top, 1], stack[top, 2], stack[top, 3] = lo, pos - 1, kl, km - 1
        stack[top + 1, 0], stack[top + 1, 1], stack[top + 1, 2], stack[top + 1, 3] = pos + 1, hi, km + 1, kh
        top += 2


@njit(cache=True)
def _few_distinct_levels(x: np.ndarray, limit: int) -> np.ndarray:
    """The sorted distinct values of ``x`` when there are at most ``limit`` of them, else an empty array; stops at the first value that exceeds the limit."""
    levels = np.empty(limit, dtype=x.dtype)
    d = 0
    for i in range(x.shape[0]):
        v = x[i]
        found = False
        for t in range(d):
            if levels[t] == v:
                found = True
                break
        if not found:
            if d == limit:
                return levels[:0]
            levels[d] = v
            d += 1
    out: np.ndarray = levels[:d].copy()
    out.sort()
    return out


@njit(cache=True, parallel=True)
def quantile_codes_block(v: np.ndarray, n_bins: int, codes: np.ndarray, nx: np.ndarray) -> None:
    """Fill ``codes[j]`` (int16) and ``nx[j]`` for every row ``j`` of ``v`` ``(columns, n)`` exactly as ``_quantile_codes`` does for a finite column."""
    k = v.shape[0]
    n = v.shape[1]
    step = 1.0 / n_bins
    for j in prange(k):
        levels = _few_distinct_levels(v[j], n_bins)
        d = levels.shape[0]
        if d > 0:
            for i in range(n):
                lo = 0
                hi = d
                x = v[j, i]
                while lo < hi:
                    mid = (lo + hi) // 2
                    if levels[mid] < x:
                        lo = mid + 1
                    else:
                        hi = mid
                codes[j, i] = lo
            nx[j] = d
            continue
        # only the order statistics the edges read are needed, not the whole sort
        want = np.empty(2 * (n_bins + 1), dtype=np.int64)
        for b in range(n_bins + 1):
            q = 1.0 if b == n_bins else b * step
            virtual = (n - 1) * q
            i0 = n - 1 if virtual >= n - 1 else int(np.floor(virtual))
            want[2 * b] = i0
            want[2 * b + 1] = min(i0 + 1, n - 1)
        want.sort()
        nw = 1
        for b in range(1, want.shape[0]):
            if want[b] != want[nw - 1]:
                want[nw] = want[b]
                nw += 1
        s = v[j].copy()
        _multi_select_in_place(s, want[:nw])
        edges = np.empty(n_bins + 1, dtype=np.float64)  # Intentional float64: quantile edges in numpy's arithmetic (equal codes to the numpy path)
        m = 0
        for b in range(n_bins + 1):
            q = 1.0 if b == n_bins else b * step
            e = _edge_value(s, q)
            if m == 0 or e != edges[m - 1]:
                edges[m] = e
                m += 1
        hi_code = max(m - 2, 0)
        top = 0
        for i in range(n):
            x = v[j, i]
            lo = 0
            hi = m
            while lo < hi:
                mid = (lo + hi) // 2
                if edges[mid] <= x:
                    lo = mid + 1
                else:
                    hi = mid
            c = lo - 1
            if c < 0:
                c = 0
            elif c > hi_code:
                c = hi_code
            codes[j, i] = c
            if c > top:
                top = c
        nx[j] = top + 1


@njit(cache=True)
def stratified_cmi(x: np.ndarray, y: np.ndarray, s: np.ndarray, min_rows: int) -> float:
    """Plug-in I(x; y | s) summed over the strata of ``s`` (non-negative integer codes), skipping strata below ``min_rows`` rows."""
    n = x.shape[0]
    counts = np.zeros((int(s.max()) + 1, int(x.max()) + 1, int(y.max()) + 1), dtype=np.int64)
    for i in range(n):
        counts[s[i], x[i], y[i]] += 1
    cmi: float = _cmi_from_counts(counts, n, min_rows)
    return cmi


@njit(cache=True, parallel=True)
def stratified_cmi_null(x: np.ndarray, y: np.ndarray, s: np.ndarray, sources: np.ndarray, min_rows: int) -> np.ndarray:
    """I(x_p; y | s) for every row ``p`` of ``sources``, where ``x_p[i] = x[sources[p, i]]`` is x permuted inside each stratum of ``s``."""
    n_perm = sources.shape[0]
    n = x.shape[0]
    n_s = int(s.max()) + 1
    n_x = int(x.max()) + 1
    n_y = int(y.max()) + 1
    out = np.zeros(n_perm, dtype=np.float64)  # Intentional float64: statistic vector (MI / null max / p-value inputs), exact permutation ties
    for p in prange(n_perm):
        counts = np.zeros((n_s, n_x, n_y), dtype=np.int64)
        for i in range(n):
            counts[s[i], x[sources[p, i]], y[i]] += 1
        out[p] = _cmi_from_counts(counts, n, min_rows)
    return out


from .supervised_binning import _mdlp_best_split_binary_counted_njit, _xlogx
from .supervised_binning import _mdlp_best_split_inc_njit as _mdlp_best_split_njit

# --- Root decision of the MDLP tree for binary labels, read off the int8 label stream (``honest_null_columns_bycol``) -------------------------------------------------------------------
#
# A tree is built for every (column, permutation) of the honest null. On a permuted label the root split is almost never accepted, and the general route pays for that decision with a
# deduplicated copy of the sorted values (8 bytes per row) and labels (8 bytes per row) before it scans them. The decision needs only the labels: the candidate cut points are the class
# boundaries of the sorted points, and the sorted values enter only through the runs of tied values, which are few and fixed per column. ``root_decision_binary`` rebuilds the deduplicated label
# stream in an int8 buffer (the working set shrinks from megabytes to the size of an L2 cache) and applies the root test of ``_mdlp_dfs`` to it with the same arithmetic. It answers only when
# the root is judged by the analytic test; otherwise it returns UNDECIDED and the caller takes the general route, which is also the route for an accepted root (it needs the cut and the children).

REJECTED = 0
ACCEPTED = 1
UNDECIDED = -1


@njit(cache=True)
def column_tie_runs(xsrt: np.ndarray) -> "tuple[np.ndarray, np.ndarray]":
    """``(starts, ends)`` of the maximal runs of equal values (two or more points) in the sorted column ``xsrt``; ``ends`` is exclusive."""
    n = xsrt.shape[0]
    n_runs = 0
    i = 0
    while i < n - 1:
        if xsrt[i] == xsrt[i + 1]:
            n_runs += 1
            j = i + 1
            while j < n and xsrt[j] == xsrt[i]:
                j += 1
            i = j
        else:
            i += 1
    starts = np.empty(n_runs, dtype=np.int64)
    ends = np.empty(n_runs, dtype=np.int64)
    r = 0
    i = 0
    while i < n - 1:
        if xsrt[i] == xsrt[i + 1]:
            j = i + 1
            while j < n and xsrt[j] == xsrt[i]:
                j += 1
            starts[r] = i
            ends[r] = j
            r += 1
            i = j
        else:
            i += 1
    return starts, ends


@njit(cache=True)
def _dedup_binary_stream(ys: np.ndarray, run_start: np.ndarray, run_end: np.ndarray, yd: np.ndarray, forb: np.ndarray) -> "tuple[int, int]":
    """Write the labels of the deduplicated points into ``yd`` and return ``(points, forbidden)``.

    Within a run of tied values the distinct labels are emitted ascending, as ``_presorted_dedup`` does. ``forb[:forbidden]`` holds the stream positions ``i`` whose boundary to ``i + 1`` lies inside
    a run (the ``0`` and ``1`` point of one value): equal values never bound a candidate cut."""
    m = 0
    nf = 0
    p = 0
    for r in range(run_start.shape[0]):
        a = run_start[r]
        e = run_end[r]
        yd[m : m + (a - p)] = ys[p:a]
        m += a - p
        ones = 0
        for i in range(a, e):
            ones += ys[i]
        zeros = (e - a) - ones
        if zeros > 0:
            yd[m] = 0
            m += 1
        if ones > 0:
            if zeros > 0:
                forb[nf] = m - 1
                nf += 1
            yd[m] = 1
            m += 1
        p = e
    n = ys.shape[0]
    yd[m : m + (n - p)] = ys[p:n]
    m += n - p
    return m, nf


@njit(cache=True)
def _tail_threshold(p_star: float) -> float:
    """The largest ``x`` (to bisection accuracy) with ``erfc(sqrt(x)) > p_star``, the chi-square(1) tail, so the analytic test of a gain ``g`` passes only if ``n * g`` exceeds it."""
    if p_star >= 1.0:
        return 0.0
    lo = 0.0
    hi = 1.0
    while math.erfc(math.sqrt(hi)) > p_star and hi < 1e7:
        hi *= 2.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if math.erfc(math.sqrt(mid)) > p_star:
            lo = mid
        else:
            hi = mid
    return lo


@njit(cache=True)
def _count_stream_candidates(yd: np.ndarray, forb: np.ndarray, n_forb: int, lo: int, hi: int) -> int:
    """Candidate cut positions in ``lo..hi``: label changes of the stream, minus the boundaries that lie inside a run of equal values."""
    n_cand = 0
    for i in range(lo, hi + 1):
        if yd[i] != yd[i + 1]:
            n_cand += 1
    for k in range(n_forb):
        if lo <= forb[k] <= hi:
            n_cand -= 1
    return n_cand


@njit(cache=True)
def _best_gain_over_bound(yd: np.ndarray, forb: np.ndarray, n_forb: int, lo: int, hi: int, n: int, tot1: int, h_full: float, scale: float) -> float:
    """Largest exact split gain among the candidates whose bound ``ln(1 + phi^2)`` reaches the acceptance threshold (``-1e300`` when none does).

    A candidate's gain is the mutual information of its 2 x 2 table, bounded by ln(1 + phi^2) with phi^2 = d^2 / (n_l n_r n_1 n_0), d = c1l * n - n_l * n_1 (KL <= ln(1 + chi^2)); ``scale`` is
    phi_min^2 * n_1 * n_0, so a candidate is skipped when d^2 < scale * n_l * n_r."""
    c1l = 0
    for i in range(lo):
        c1l += yd[i]
    # Usual outcome on a permuted label: no position reaches the bound. A branch-light scan over all positions (a superset of the candidates) settles that before the per-candidate work.
    run1 = c1l
    reached = False
    fn = float(n)
    ft1 = float(tot1)
    for i in range(lo, hi + 1):
        run1 += yd[i]
        fl = float(i + 1)
        dd = float(run1) * fn - fl * ft1
        if dd * dd >= scale * fl * (fn - fl):
            reached = True
            break
    if not reached:
        return -1e300
    best_gain = -1e300
    inv_n = 1.0 / n
    fb = 0
    for i in range(lo, hi + 1):
        y_i = yd[i]
        c1l += y_i
        n_l = i + 1
        n_r = n - n_l
        d = float(c1l * n - n_l * tot1)
        if d * d < scale * float(n_l) * float(n_r):
            continue
        if y_i == yd[i + 1]:
            continue
        while fb < n_forb and forb[fb] < i:
            fb += 1
        if fb < n_forb and forb[fb] == i:
            continue
        c0l = n_l - c1l
        c1r = tot1 - c1l
        c0r = n_r - c1r
        h_split = (_xlogx(n_l) - _xlogx(c0l) - _xlogx(c1l) + _xlogx(n_r) - _xlogx(c0r) - _xlogx(c1r)) * inv_n
        gain = h_full - h_split
        if gain > best_gain:
            best_gain = gain
    return best_gain


@njit(cache=True)
def root_decision_binary(
    ys: np.ndarray,
    run_start: np.ndarray,
    run_end: np.ndarray,
    yd: np.ndarray,
    forb: np.ndarray,
    min_split_size: int,
    alpha: float,
    analytic_min_n: int,
    analytic_min_cell: float,
) -> int:
    """Whether the root split of the sorted column with 0/1 labels ``ys`` is accepted, as ``_mdlp_dfs`` decides it: ACCEPTED, REJECTED, or UNDECIDED when the root is not judged by the analytic test.

    ``yd`` (int8, ``len(ys)``) and ``forb`` (int64, one slot per tie run) are scratch buffers. The root is accepted when p(best gain) * n_cand < alpha, i.e. when some candidate's gain exceeds the
    threshold g*; only the positions whose bound reaches g* get the exact gain."""
    n, n_forb = _dedup_binary_stream(ys, run_start, run_end, yd, forb)
    if n < 2 * min_split_size:
        return REJECTED
    if not (n >= analytic_min_n and float(n) / max(1.0, 2.0 * 2) >= analytic_min_cell):
        return UNDECIDED
    tot1 = 0
    for i in range(n):
        tot1 += yd[i]
    lo = min_split_size - 1  # first scanned position: n_l = lo + 1 >= min_split_size
    hi = n - min_split_size - 1  # last scanned position: n_r = n - hi - 1 >= min_split_size
    n_cand = _count_stream_candidates(yd, forb, n_forb, lo, hi)
    verdict: int = _root_verdict(yd, forb, n_forb, n, tot1, n_cand, lo, hi, alpha)
    return verdict


@njit(cache=True)
def _root_verdict(yd: np.ndarray, forb: np.ndarray, n_forb: int, n: int, tot1: int, n_cand: int, lo: int, hi: int, alpha: float) -> int:
    """ACCEPTED or REJECTED for a stream of ``n`` points with ``tot1`` ones and ``n_cand`` candidate cuts among positions ``lo..hi``."""
    counts_total = np.empty(2, dtype=np.int64)
    counts_total[0] = n - tot1
    counts_total[1] = tot1
    h_full = _entropy_from_counts_njit(counts_total, n)
    if h_full <= 0.0:
        return REJECTED
    x_star = _tail_threshold(alpha / max(1, n_cand)) * (1.0 - 1e-6)
    scale = math.expm1(x_star / n) * float(tot1) * float(n - tot1)
    best_gain = _best_gain_over_bound(yd, forb, n_forb, lo, hi, n, tot1, h_full, scale)
    if best_gain <= 0.0:
        return REJECTED
    p_value = min(1.0, _analytic_p_value(best_gain, n, 2) * max(1, n_cand))
    return ACCEPTED if p_value < alpha else REJECTED


@njit(cache=True)
def gather_binary_stream(yr_b: np.ndarray, order_row: np.ndarray, ys: np.ndarray) -> "tuple[int, int]":
    """``ys[i] = yr_b[order_row[i]]`` in one pass that also counts the ones and the label changes between neighbours (``i`` and ``i + 1``)."""
    ones = 0
    trans = 0
    prev = yr_b[order_row[0]]
    ys[0] = prev
    ones += prev
    for i in range(1, ys.shape[0]):
        v = yr_b[order_row[i]]
        ys[i] = v
        ones += v
        if v != prev:
            trans += 1
        prev = v
    return ones, trans


@njit(cache=True)
def root_decision_tie_free(ys: np.ndarray, ones: int, trans: int, min_split_size: int, alpha: float, analytic_min_n: int, analytic_min_cell: float) -> int:
    """``root_decision_binary`` for a column without tied values, from the label stream ``ys`` and the ``ones`` / ``trans`` counted by ``gather_binary_stream``: no tie run, so the stream is the
    deduplicated one and no boundary is forbidden."""
    n = ys.shape[0]
    if n < 2 * min_split_size:
        return REJECTED
    if not (n >= analytic_min_n and float(n) / max(1.0, 2.0 * 2) >= analytic_min_cell):
        return UNDECIDED
    lo = min_split_size - 1
    hi = n - min_split_size - 1
    n_cand = trans
    for i in range(lo):
        if ys[i] != ys[i + 1]:
            n_cand -= 1
    for i in range(hi + 1, n - 1):
        if ys[i] != ys[i + 1]:
            n_cand -= 1
    verdict: int = _root_verdict(ys, np.empty(0, dtype=np.int64), 0, n, ones, n_cand, lo, hi, alpha)
    return verdict


@njit(cache=True)
def scrub_nonfinite_1d(a: np.ndarray) -> None:
    """Set every NaN and infinity of ``a`` to 0 in place (one pass; ``np.nan_to_num(a, nan=0, posinf=0, neginf=0)``)."""
    for i in range(a.shape[0]):
        v = a[i]
        if not (v - v == 0):  # NaN and +-inf: the difference is not 0
            a[i] = 0


@njit(cache=True)
def scrub_nonfinite_2d(a: np.ndarray) -> None:
    """Set every NaN and infinity of the 2-D ``a`` to 0 in place, row by row."""
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            v = a[i, j]
            if not (v - v == 0):
                a[i, j] = 0


# ---- bit-packed binary label streams: the root test of the honest null on 64 labels per word ----


@njit(cache=True)
def _pc64(x):
    """Population count of a uint64 (SWAR)."""
    x = x - ((x >> np.uint64(1)) & np.uint64(0x5555555555555555))
    x = (x & np.uint64(0x3333333333333333)) + ((x >> np.uint64(2)) & np.uint64(0x3333333333333333))
    x = (x + (x >> np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
    return np.int64((x * np.uint64(0x0101010101010101)) >> np.uint64(56))


@njit(cache=True)
def pack_sorted_labels(yr_b: np.ndarray, order_row: np.ndarray, raw_bits: np.ndarray) -> int:
    """Pack the 0/1 labels read through ``order_row`` (``yr_b[order_row[i]]``, the labels in the column's sorted order) into ``raw_bits`` (bit ``i`` of the stream is bit ``i & 63`` of word ``i >> 6``) and return the
    number of ones. ``raw_bits`` is overwritten, bits past the end of the stream are zero."""
    n = order_row.shape[0]
    words = (n + 63) >> 6
    ones = 0
    for w in range(words):
        base = w << 6
        count = min(64, n - base)
        acc = np.uint64(0)
        for k in range(count):
            acc |= np.uint64(yr_b[order_row[base + k]]) << np.uint64(k)
        raw_bits[w] = acc
        ones += _pc64(acc)
    return ones


@njit(cache=True)
def _copy_bits(src: np.ndarray, start: int, stop: int, dst: np.ndarray, offset: int) -> int:
    """Copy bits ``[start, stop)`` of the stream ``src`` to ``dst`` from bit ``offset`` on (``dst`` bits there must be zero); returns the offset after the copy."""
    s = start
    off = offset
    while s < stop:
        w = s >> 6
        sh = s & 63
        take = min(64 - sh, stop - s)
        val = (src[w] >> np.uint64(sh)) & (np.uint64(0xFFFFFFFFFFFFFFFF) >> np.uint64(64 - take))
        dw = off >> 6
        dsh = off & 63
        dst[dw] |= val << np.uint64(dsh)
        if dsh + take > 64:
            dst[dw + 1] |= val >> np.uint64(64 - dsh)
        s += take
        off += take
    return off


@njit(cache=True)
def dedup_label_bits(raw_bits: np.ndarray, n: int, run_start: np.ndarray, run_end: np.ndarray, out_bits: np.ndarray, forb: np.ndarray) -> "tuple[int, int]":
    """The deduplicated stream (see ``_dedup_binary_stream``) of the packed stream ``raw_bits`` of length ``n``, packed into ``out_bits`` (overwritten): rows outside tie runs are copied, a run of tied values
    emits its distinct labels ascending. Returns ``(points, forbidden)``; ``forb[:forbidden]`` are the stream positions whose boundary to the next point lies inside a run."""
    for w in range(out_bits.shape[0]):
        out_bits[w] = np.uint64(0)
    m = 0
    nf = 0
    p = 0
    for r in range(run_start.shape[0]):
        a = run_start[r]
        e = run_end[r]
        m = _copy_bits(raw_bits, p, a, out_bits, m)
        ones = 0
        for i in range(a, e):
            ones += int((raw_bits[i >> 6] >> np.uint64(i & 63)) & np.uint64(1))
        zeros = (e - a) - ones
        if zeros > 0:
            m += 1  # a 0 point: the bit stays zero
        if ones > 0:
            if zeros > 0:
                forb[nf] = m - 1
                nf += 1
            out_bits[m >> 6] |= np.uint64(1) << np.uint64(m & 63)
            m += 1
        p = e
    m = _copy_bits(raw_bits, p, n, out_bits, m)
    return m, nf


@njit(cache=True)
def _count_ones_bits(bits: np.ndarray, start: int, stop: int) -> int:
    """Ones among stream positions ``[start, stop)``."""
    if stop <= start:
        return 0
    total = 0
    w0 = start >> 6
    w1 = (stop - 1) >> 6
    for w in range(w0, w1 + 1):
        word = bits[w]
        lo_bit = start - (w << 6) if w == w0 else 0
        hi_bit = stop - (w << 6) if w == w1 else 64  # exclusive
        if lo_bit > 0:
            word = word >> np.uint64(lo_bit) << np.uint64(lo_bit)
        if hi_bit < 64:
            word = word & ((np.uint64(1) << np.uint64(hi_bit)) - np.uint64(1))
        total += _pc64(word)
    return total


@njit(cache=True)
def _candidate_count_bits(bits: np.ndarray, forb: np.ndarray, n_forb: int, lo: int, hi: int) -> int:
    """``_count_stream_candidates`` on the packed stream: label changes between positions ``i`` and ``i + 1`` for ``i`` in ``lo..hi``, minus the boundaries that lie inside a run of equal values."""
    n_words = bits.shape[0]
    n_cand = 0
    for w in range(lo >> 6, (hi >> 6) + 1):
        cur = bits[w]
        nxt = bits[w + 1] if w + 1 < n_words else np.uint64(0)
        change = cur ^ ((cur >> np.uint64(1)) | (nxt << np.uint64(63)))
        base = w << 6
        k0 = max(lo - base, 0)
        k1 = min(hi - base, 63)
        if k0 > 0:
            change = change >> np.uint64(k0) << np.uint64(k0)
        if k1 < 63:
            change = change & ((np.uint64(1) << np.uint64(k1 + 1)) - np.uint64(1))
        n_cand += _pc64(change)
    for k in range(n_forb):
        if lo <= forb[k] <= hi:
            n_cand -= 1
    return n_cand


@njit(cache=True)
def root_decision_packed(bits: np.ndarray, m: int, forb: np.ndarray, n_forb: int, min_split_size: int, alpha: float, analytic_min_n: int, analytic_min_cell: float) -> int:
    """``root_decision_binary`` for the packed deduplicated stream ``bits`` of ``m`` points (see ``dedup_label_bits``): the same verdict.

    The candidate count is a popcount of the label changes. The scan for a position whose bound reaches the acceptance threshold skips every word of 64 positions that cannot reach it: from the
    value ``d`` at the start of the word, ``|d|`` moves by at most ``max(n_1, n_0)`` per position, so the word is skipped when even that largest value stays below the smallest threshold in the word.
    A word that may reach it is scanned position by position with the arithmetic of the unpacked scan, and when a position does reach it the unpacked stream goes to ``_best_gain_over_bound``."""
    n = m
    if n < 2 * min_split_size:
        return REJECTED
    if not (n >= analytic_min_n and float(n) / max(1.0, 2.0 * 2) >= analytic_min_cell):
        return UNDECIDED
    tot1 = _count_ones_bits(bits, 0, n)
    counts_total = np.empty(2, dtype=np.int64)
    counts_total[0] = n - tot1
    counts_total[1] = tot1
    h_full = _entropy_from_counts_njit(counts_total, n)
    if h_full <= 0.0:
        return REJECTED
    lo = min_split_size - 1
    hi = n - min_split_size - 1
    n_cand = _candidate_count_bits(bits, forb, n_forb, lo, hi)
    x_star = _tail_threshold(alpha / max(1, n_cand)) * (1.0 - 1e-6)
    scale = math.expm1(x_star / n) * float(tot1) * float(n - tot1)
    fn = float(n)
    ft1 = float(tot1)
    step = float(max(tot1, n - tot1))
    reached = False
    c1_before = 0
    for w in range(0, (hi >> 6) + 1):
        base = w << 6
        word = bits[w]
        first = max(base, lo)
        last = min(base + 63, hi)
        if last >= first:
            d_prev = float(c1_before) * fn - float(base) * ft1
            bound = (abs(d_prev) + 64.0 * step) ** 2
            f_first = scale * float(first + 1) * (fn - float(first + 1))
            f_last = scale * float(last + 1) * (fn - float(last + 1))
            if bound >= 0.999999 * min(f_first, f_last):
                c1 = c1_before
                for i in range(base, last + 1):
                    c1 += int((word >> np.uint64(i - base)) & np.uint64(1))
                    if i >= first:
                        fl = float(i + 1)
                        dd = float(c1) * fn - fl * ft1
                        if dd * dd >= scale * fl * (fn - fl):
                            reached = True
                            break
                if reached:
                    break
        c1_before += _pc64(word)
    if not reached:
        return REJECTED
    yd = np.empty(n, dtype=np.int8)
    for i in range(n):
        yd[i] = np.int8((bits[i >> 6] >> np.uint64(i & 63)) & np.uint64(1))
    best_gain = _best_gain_over_bound(yd, forb, n_forb, lo, hi, n, tot1, h_full, scale)
    if best_gain <= 0.0:
        return REJECTED
    p_value = min(1.0, _analytic_p_value(best_gain, n, 2) * max(1, n_cand))
    return ACCEPTED if p_value < alpha else REJECTED


@njit(cache=True, parallel=True)
def pack_label_rows(rows: np.ndarray) -> np.ndarray:
    """``(rows, words)`` uint64: every row of the 0/1 matrix ``rows`` packed 64 labels to a word (bit ``i`` of a row is bit ``i & 63`` of word ``i >> 6``)."""
    n_rows = rows.shape[0]
    n = rows.shape[1]
    words = (n + 63) >> 6
    out = np.zeros((n_rows, words), dtype=np.uint64)
    for b in prange(n_rows):
        for w in range(words):
            base = w << 6
            count = min(64, n - base)
            acc = np.uint64(0)
            for k in range(count):
                acc |= np.uint64(rows[b, base + k]) << np.uint64(k)
            out[b, w] = acc
    return out


@njit(cache=True)
def pack_sorted_labels_from_bits(label_bits: np.ndarray, order_row: np.ndarray, raw_bits: np.ndarray) -> int:
    """``pack_sorted_labels`` for labels held packed in ``label_bits``: the random reads of the gather hit a row of n / 8 bytes (cache resident) instead of n."""
    n = order_row.shape[0]
    words = (n + 63) >> 6
    ones = 0
    for w in range(words):
        base = w << 6
        count = min(64, n - base)
        acc = np.uint64(0)
        for k in range(count):
            r = order_row[base + k]
            acc |= ((label_bits[r >> 6] >> np.uint64(r & 63)) & np.uint64(1)) << np.uint64(k)
        raw_bits[w] = acc
        ones += _pc64(acc)
    return ones


@njit(cache=True, parallel=True)
def within_stratum_shuffles(keys: np.ndarray, z_rank: np.ndarray, order: np.ndarray, x_sorted: np.ndarray, out: np.ndarray) -> None:
    """Row ``i`` of ``out`` is the candidate column shuffled within the strata of the support: ``out[i][order[j]] = x_sorted[w[j]]`` with ``w`` the stable argsort of ``z_rank + keys[i]``.

    ``order`` sorts the rows by stratum, ``x_sorted`` is the candidate in that order and ``z_rank`` the dense stratum number of each sorted row, so adding a key in ``[0, 1)`` sorts the rows of a stratum
    by their key: a uniform shuffle inside each stratum. The rows are independent, hence one parallel task per shuffle; the result is that of ``np.argsort(z_rank + keys[i], kind="stable")`` row by row."""
    for i in prange(keys.shape[0]):
        within = np.argsort(z_rank + keys[i], kind="mergesort")
        for j in range(within.shape[0]):
            out[i, order[j]] = x_sorted[within[j]]


@njit(cache=True)
def rows_to_columns(rows: np.ndarray) -> np.ndarray:
    """``(n, k)`` C-ordered copy of the ``(k, n)`` matrix ``rows`` (``out[r, j] = rows[j, r]``), by tiles of rows so that every tile of the output is written while it is cache resident.

    Writing the candidate columns of a block one by one into an ``(n, k)`` matrix touches a cache line and a page per element (a stride of ``k`` items); filling contiguous rows and transposing once is the same
    data movement without that pattern."""
    k, n = rows.shape
    out = np.empty((n, k), dtype=rows.dtype)
    tile = 64
    for r0 in range(0, n, tile):
        r1 = min(n, r0 + tile)
        for j in range(k):
            for r in range(r0, r1):
                out[r, j] = rows[j, r]
    return out


@njit(cache=True, parallel=True)
def column_fingerprints(bits: np.ndarray) -> np.ndarray:
    """``(k, 2)`` uint64 content fingerprints of the columns of the ``(n, k)`` integer matrix ``bits`` (the raw bit patterns of a float matrix): two independent FNV-1a style lanes over the column in row order, so
    two columns with equal values in the same order get equal fingerprints and a different value or order changes both. Reads the columns in place (no per-column copy of a strided column)."""
    n, k = bits.shape
    out = np.empty((k, 2), dtype=np.uint64)
    for j in prange(k):
        h1 = np.uint64(14695981039346656037)
        h2 = np.uint64(0x9E3779B97F4A7C15)
        for i in range(n):
            v = np.uint64(bits[i, j])
            h1 = (h1 ^ v) * np.uint64(1099511628211)
            h2 = (h2 ^ (v + np.uint64(0x632BE59BD9B4E019))) * np.uint64(0xC2B2AE3D27D4EB4F)
            h2 ^= h2 >> np.uint64(29)
        out[j, 0] = h1
        out[j, 1] = h2
    return out


@njit(cache=True)
def rows_to_columns_into(rows: np.ndarray, out: np.ndarray, col0: int) -> None:
    """``out[r, col0 + j] = rows[j, r]``: the ``(k, n)`` block ``rows`` written, tile by tile, into columns ``col0 .. col0 + k`` of the ``(n, K)`` matrix ``out`` (see ``rows_to_columns``)."""
    k, n = rows.shape
    tile = 64
    for r0 in range(0, n, tile):
        r1 = min(n, r0 + tile)
        for j in range(k):
            for r in range(r0, r1):
                out[r, col0 + j] = rows[j, r]
