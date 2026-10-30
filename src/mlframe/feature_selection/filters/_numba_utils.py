"""Shared cross-module ``@njit`` helpers.

INVARIANT: every ``@njit`` helper called from more than one ``filters/*`` submodule lives here; single-module njit helpers stay in their owning module.

Rationale: numba dispatcher objects bind to the importing module at compile time. Duplicating a helper across modules compiles two distinct dispatchers
under the same Python name, invalidates the cache on every cross-module dispatch, causes silent recompiles, longer cold starts, and on some Windows
configurations file-lock races against ``__pycache__``. Consolidating shared kernels here keeps the dispatcher graph clean and cache invalidation
predictable. Enforced by ``mlframe/tests/test_meta/test_filters_numba_invariant.py``.
"""
from __future__ import annotations

import math

import numpy as np
from numba import njit, prange

from ._quantile_bin_stages import _quantile_crossing_cuts, _quantile_crossing_offsets
from typing import Any, Optional, Sequence

from ._njit_key_helpers import arr2str, unpack_and_sort  # noqa: F401 -- re-exported; defined in the leaf so info_theory can import them without a cycle

import os

from pyutilz.numbalib import _generate_combinations_recursive_njit_core

from ._internals import LARGE_CONST, MAX_CONFIRMATION_CAND_NBINS
from .info_theory import conditional_mi, conditional_symmetric_uncertainty, mi

# Re-exported from the split-out sibling (see its module docstring): row-statistic, offset-product and entropy kernels.
from ._numba_utils_offset import (  # noqa: F401
    STAT_NAMES,
    MAX_SUBSET,
    LSE_BETA,
    N_EDGE_SAMPLE,
    _MIN,
    _MAX,
    _MED,
    _RNG,
    _STD,
    _LSE_POS,
    _LSE_NEG,
    _stat_of,
    _row_value,
    _mi,
    eval_candidates,
    stat_column,
    NACC,
    N_EDGE_SUBSAMPLE,
    N_BASELINES,
    _SINGULAR_REL_TOL,
    _MIN_SPAN,
    _solve4,
    ols2_shift,
    _edges_of,
    _bin_of,
    _mi_from_counts,
    scan_offset_products,
    weighted_sum_heldout_mi,
    _backfit,
    _entropy_from_counts_njit,
)


@njit(cache=True)
def count_cand_nbins(X: Sequence[int], factors_nbins: np.ndarray) -> int:
    """Sum the bin count across the factors named in candidate ``X``. Used by ``screen_predictors`` to gate the confirmation step (permutation testing is
    skipped when the conditioning set has more bins than ``MAX_CONFIRMATION_CAND_NBINS``)."""
    sum_cand_nbins = 0
    for factor in X:
        sum_cand_nbins += factors_nbins[factor]
    return sum_cand_nbins


@njit(cache=True)
def distribute_permutations(npermutations: int, n_workers: int) -> list:
    """Split ``npermutations`` across ``n_workers``; the remainder lands on the last worker."""
    avg_perms_per_worker = npermutations // n_workers
    diff = npermutations - avg_perms_per_worker * n_workers
    workload = [avg_perms_per_worker] * n_workers
    if diff > 0:
        workload[-1] = workload[-1] + diff
    return workload


@njit(nogil=True, cache=True)
def _joint_mi_3d_njit(x: np.ndarray, z: np.ndarray, y: np.ndarray, K_x: int, K_z: int, K_y: int) -> float:
    """I((X, Z); Y) via plug-in on the 3-D joint count cube.

    Treats (x_i, z_i) as a flat composite category, then collapses to a
    2-D (composite, y) joint and computes plug-in MI.
    """
    n = x.shape[0]
    if n <= 0:
        return 0.0
    K_xz = K_x * K_z
    joint = np.zeros((K_xz, K_y), dtype=np.float64)  # Intentional float64: numba kernel body buffer or scalar (crit helper cannot run inside njit)
    for i in range(n):
        composite = int(x[i]) * K_z + int(z[i])
        joint[composite, int(y[i])] += 1.0
    n_f = float(n)
    Px = np.zeros(K_xz, dtype=np.float64)  # Intentional float64: numba kernel body buffer or scalar (crit helper cannot run inside njit)
    Py = np.zeros(K_y, dtype=np.float64)  # Intentional float64: numba kernel body buffer or scalar (crit helper cannot run inside njit)
    for i in range(K_xz):
        for j in range(K_y):
            v = joint[i, j]
            Px[i] += v
            Py[j] += v
    mi = 0.0
    for i in range(K_xz):
        if Px[i] <= 0.0:
            continue
        for j in range(K_y):
            v = joint[i, j]
            if v <= 0.0 or Py[j] <= 0.0:
                continue
            p = v / n_f
            mi += p * math.log(p * n_f / (Px[i] * Py[j] / n_f))
    return max(0.0, mi)


@njit(cache=True)
def _pair_mm_mi_njit(code_x, code_y, target, kx, ky, kt, min_rows_per_cell):
    """Fused O(n) Miller-Madow-corrected joint MI ``I({X,Y}; T)`` for a CODE PAIR - the
    nopython equivalent of ``joint_synergy_mi(code_x, code_y, target)``, with NO
    ``np.unique`` densify and NO ``np.add.at`` scatter.

    ``code_x`` / ``code_y`` are int64 per-feature codes in ``[0, kx)`` / ``[0, ky)``; ``target``
    is the int64 target codes in ``[0, kt)``. The (x, y) cell is the mixed-radix id
    ``cx*ky + cy`` over the DENSE ``kx*ky`` grid; the joint table is ``(kx*ky) x kt``. MI is
    summed over OCCUPIED cells with the SAME estimator as the numpy reference
    (``sum p*log(p/(px*py))``), debited by the MM bias ``(occ_x-1 + occ_y-1 - (occ-1))/(2n)``
    using OCCUPIED joint / target counts - the identical bias ``joint_synergy_mi`` applies
    on its renumbered occupied grid (densification is order-preserving for occupancy, so the
    occupied counts and the MM debit are unchanged; the MI value matches to FP reduction order).

    OCCUPANCY FLOOR: matches ``joint_synergy_mi`` - the floor there compares ``n`` against
    ``min_rows_per_cell * (occupied-joint-cardinality) * (target max code + 1)``. We count
    occupied joint cells after the histogram pass and use the FULL target cardinality ``kt``
    (= ``yt.max()+1``), applying the identical gate, returning 0.0 below it.

    Returns ``max(0.0, mi - mm)`` (>= 0)."""
    n = code_x.shape[0]
    if n == 0:
        return 0.0
    ncells = kx * ky
    joint = np.zeros(ncells * kt, dtype=np.float64)  # Intentional float64: numba kernel body buffer or scalar (crit helper cannot run inside njit)
    px = np.zeros(ncells, dtype=np.float64)  # Intentional float64: numba kernel body buffer or scalar (crit helper cannot run inside njit)
    py = np.zeros(kt, dtype=np.float64)  # Intentional float64: numba kernel body buffer or scalar (crit helper cannot run inside njit)
    for r in range(n):
        cell = code_x[r] * ky + code_y[r]
        t = target[r]
        joint[cell * kt + t] += 1.0
        px[cell] += 1.0
        py[t] += 1.0
    # occupied joint-cell count == the renumbered joint cardinality the numpy ref uses for the floor.
    occ_cells = 0
    for c in range(ncells):
        if px[c] > 0.0:
            occ_cells += 1
    occ_y = 0
    for t in range(kt):
        if py[t] > 0.0:
            occ_y += 1
    if min_rows_per_cell > 0.0 and n < min_rows_per_cell * float(occ_cells) * float(kt):
        return 0.0
    inv_n = 1.0 / n
    mi = 0.0
    occ = 0
    for cell in range(ncells):
        if px[cell] <= 0.0:
            continue
        pxc = px[cell] * inv_n
        base = cell * kt
        for t in range(kt):
            cnt = joint[base + t]
            if cnt > 0.0:
                occ += 1
                pj = cnt * inv_n
                mi += pj * np.log(pj / (pxc * (py[t] * inv_n)))
    mm = (occ_cells - 1 + occ_y - 1 - (occ - 1)) / (2.0 * n)
    val = mi - mm
    return val if val > 0.0 else 0.0


@njit(cache=True, fastmath=True, inline="always")
def _edge_bin_codes(val: np.ndarray, n_bins: int, codes_out: np.ndarray) -> int:
    """Equi-frequency PERCENTILE-EDGE bin ``val`` (finite double) into ``codes_out`` (int32, in
    [0, n_bins)). Matches the GPU orth twin ``_hermite_fe_mi._plugin_mi_classif_batch_cuda_resident``
    EXACTLY: the FIXED ``n_bins-1`` interior edges ``cp.percentile(X, qs)[1:-1]`` (np.quantile linear
    interpolation at virtual index ``q*(n-1)``) with NO dedup, then ``searchsorted(interior, side='right')``.

    NOTE the deliberate difference from ``_usability_njit_pool._qbin_into`` (the usability-pool binner),
    which np.unique-DEDUPS the edges and so collapses tied bins to a variable cardinality. The orth/basis
    family's GPU kernel does NOT dedup - it keeps all n_bins-1 interior edges - so on tied / duplicate
    values the two conventions assign different codes. This twin follows the GPU (no dedup) so the CPU and
    GPU batcher backends score the orth family identically on tied columns (verified ~1e-9, the
    test_fe_edge_mi_parity 'tied' case). Returns ``n_bins`` (the fixed bin count)."""
    n = val.shape[0]
    nq = n_bins + 1
    # Edges need only the order statistics at the lerp anchors (lo, lo+1) of each of the nq quantile
    # points - ~2*nq positions, not the whole sorted array. np.partition places EXACTLY those positions
    # at their sorted values in O(n) (introselect) vs the O(n log n) full sort; part[lo]/part[hi] are then
    # the identical order statistics np.sort would give, so the edges (and MI) are BIT-IDENTICAL.
    los = np.empty(nq, dtype=np.int64)
    fracs = np.empty(nq, dtype=np.float64)  # Intentional float64: numba kernel body buffer or scalar (crit helper cannot run inside njit)
    kths = np.empty(2 * nq, dtype=np.int64)
    m = 0
    for k in range(nq):
        pos = (k / (nq - 1)) * (n - 1)  # quantile fraction k/n_bins -> virtual index q*(n-1)
        lo = int(np.floor(pos))
        hi = lo + 1 if lo < n - 1 else lo
        los[k] = lo
        fracs[k] = pos - lo
        kths[m] = lo; kths[m + 1] = hi; m += 2
    part = np.partition(val, kths[:m])
    edges = np.empty(nq, dtype=np.float64)  # Intentional float64: numba kernel body buffer or scalar (crit helper cannot run inside njit)
    for k in range(nq):
        lo = los[k]
        hi = lo + 1 if lo < n - 1 else lo
        edges[k] = part[lo] + (part[hi] - part[lo]) * fracs[k]
    ni = n_bins - 1  # interior edges live at edges[1 .. n_bins-1]; searchsorted-right via bisect
    if ni <= 0:
        for i in range(n):
            codes_out[i] = 0
        return n_bins
    for i in range(n):
        v = val[i]
        lo = 0
        hi = ni
        while lo < hi:
            mid = (lo + hi) // 2
            if v < edges[1 + mid]:
                hi = mid
            else:
                lo = mid + 1
        codes_out[i] = lo
    return n_bins


@njit(cache=True, fastmath=True)
def _plugin_mi_classif_edge_njit(x: np.ndarray, y: np.ndarray, n_bins: int) -> float:
    """Plain plug-in MI of continuous x (1-D double) with discrete y (1-D int64), EDGE-binning x.
    Same plug-in estimator (nats) and occupied-cell reduction order as
    ``_hermite_fe_mi._plugin_mi_classif_njit`` - only the x binning is percentile-edge, not rank."""
    n = x.shape[0]
    if n == 0:
        return 0.0
    y_min = y[0]
    y_max = y[0]
    for i in range(1, n):
        if y[i] < y_min:
            y_min = y[i]
        if y[i] > y_max:
            y_max = y[i]
    n_classes = (y_max - y_min) + 1

    codes = np.empty(n, dtype=np.int32)
    _edge_bin_codes(x, n_bins, codes)

    hist_xy = np.zeros((n_bins, n_classes), dtype=np.int64)
    hist_x = np.zeros(n_bins, dtype=np.int64)
    hist_y = np.zeros(n_classes, dtype=np.int64)
    for i in range(n):
        b = codes[i]
        c = y[i] - y_min
        hist_xy[b, c] += 1
        hist_x[b] += 1
        hist_y[c] += 1

    log_n = np.log(n)
    mi = 0.0
    for b in range(n_bins):
        if hist_x[b] == 0:
            continue
        log_hx = np.log(hist_x[b])
        for c in range(n_classes):
            n_xy = hist_xy[b, c]
            if n_xy == 0 or hist_y[c] == 0:
                continue
            mi += (n_xy / n) * (np.log(n_xy) + log_n - log_hx - np.log(hist_y[c]))
    if mi < 0.0:
        mi = 0.0
    return mi


@njit(cache=True, fastmath=True, parallel=True)
def plugin_mi_classif_batch_edge_njit(X_cols: np.ndarray, y: np.ndarray, n_bins: int = 20) -> np.ndarray:
    """Edge-binning plain plug-in MI of each column of ``X_cols`` (continuous, (n,k) double) with discrete
    y. Parallel over columns. The CPU batcher's per-column MI core; identical (to ~fp reduction order) to
    the GPU resident edge plug-in MI, so the two batcher backends select the same forms on every column."""
    k = X_cols.shape[1]
    out = np.zeros(k, dtype=np.float64)  # Intentional float64: numba kernel body buffer or scalar (crit helper cannot run inside njit)
    for j in prange(k):
        out[j] = _plugin_mi_classif_edge_njit(np.ascontiguousarray(X_cols[:, j]), y, n_bins)
    return out


@njit(cache=True, fastmath=True)
def _plugin_mi_classif_njit(x: np.ndarray, y: np.ndarray, n_bins: int = 20) -> float:
    """Plug-in MI estimator for continuous x (1-D double) and discrete y (1-D int64). Returns MI in nats.
    ~50x faster than sklearn for n<=10k, single-thread."""
    # Lazy import of parent-resident helpers: ``.hermite_fe`` re-imports
    # this sibling at its bottom, so a top-level ``from .hermite_fe
    # import ...`` would create a hard cycle the meta-test flags.
    n = x.shape[0]
    if n == 0:
        return 0.0  # empty column (fully-filtered subsample / empty finite mask): y[0] below would OOB-crash
    # Class axis spans [y_min, y_max]; labels may be negative / non-dense (a binned continuous target shifted below 0).
    # Sizing on max(y)+1 alone and indexing with the raw label underflows the histogram into out-of-bounds memory -> native AV.
    y_min = y[0]
    y_max = y[0]
    for i in range(1, n):
        if y[i] < y_min:
            y_min = y[i]
        if y[i] > y_max:
            y_max = y[i]
    n_classes = (y_max - y_min) + 1

    x_binned = _quantile_bin_njit(x, n_bins)

    hist_xy = np.zeros((n_bins, n_classes), dtype=np.int64)
    hist_x = np.zeros(n_bins, dtype=np.int64)
    hist_y = np.zeros(n_classes, dtype=np.int64)
    for i in range(n):
        b = x_binned[i]
        c = y[i] - y_min
        hist_xy[b, c] += 1
        hist_x[b] += 1
        hist_y[c] += 1

    log_n = math.log(n)
    mi = 0.0
    for b in range(n_bins):
        if hist_x[b] == 0:
            continue
        log_hx = math.log(hist_x[b])
        for c in range(n_classes):
            n_xy = hist_xy[b, c]
            if n_xy == 0 or hist_y[c] == 0:
                continue
            mi += (n_xy / n) * (math.log(n_xy) + log_n - log_hx - math.log(hist_y[c]))
    if mi < 0.0:
        mi = 0.0
    return mi

@njit(cache=True, fastmath=True)
def _plugin_mi_regression_njit(x: np.ndarray, y: np.ndarray, n_bins: int = 20) -> float:
    """Plug-in MI for continuous x (1-D) and continuous y (1-D). Bin both into n_bins equi-frequency bins, then plug-in estimator."""
    # Lazy import of parent-resident helpers: ``.hermite_fe`` re-imports
    # this sibling at its bottom, so a top-level ``from .hermite_fe
    # import ...`` would create a hard cycle the meta-test flags.
    n = x.shape[0]
    if n == 0:
        return 0.0  # empty column: log(n) and the binning below are undefined on n=0
    x_binned = _quantile_bin_njit(x, n_bins)
    y_binned = _quantile_bin_njit(y, n_bins)

    hist_xy = np.zeros((n_bins, n_bins), dtype=np.int64)
    hist_x = np.zeros(n_bins, dtype=np.int64)
    hist_y = np.zeros(n_bins, dtype=np.int64)
    for i in range(n):
        bx = x_binned[i]
        by = y_binned[i]
        hist_xy[bx, by] += 1
        hist_x[bx] += 1
        hist_y[by] += 1

    log_n = math.log(n)
    mi = 0.0
    for bx in range(n_bins):
        if hist_x[bx] == 0:
            continue
        log_hx = math.log(hist_x[bx])
        for by in range(n_bins):
            n_xy = hist_xy[bx, by]
            if n_xy == 0 or hist_y[by] == 0:
                continue
            mi += (n_xy / n) * (math.log(n_xy) + log_n - log_hx - math.log(hist_y[by]))
    if mi < 0.0:
        mi = 0.0
    return mi


@njit(cache=True, fastmath=True)
def _quantile_bin_sorted_njit(x: np.ndarray, n_bins: int) -> np.ndarray:
    """Quantile-bin a 1-D continuous array into n_bins equi-frequency bins. Returns int32 bin indices in [0, n_bins).

    bench-attempt-rejected: kth-order-statistic edges via numba np.partition + searchsorted (the CPU analogue
    of the GPU radix-edge binner) measured 2.6x SLOWER than this argsort form (99401: 17.8ms vs 46.5ms) -
    numba's np.partition re-copies the array per edge, and no multi-kth variant exists. argsort stays optimal
    (third measured rejection at this site; see also the two in _plugin_mi_classif_batch_njit).

    ``kind="mergesort"`` (2026-07-16, cProfile/cross-backend-parity investigation): numba's default argsort
    (quicksort family) is NOT stable, so on tie-heavy columns (e.g. an integer combiner with many exact
    duplicate values) its tie order is an unspecified implementation detail - and it does NOT match cupy's
    ``cp.argsort`` (verified stable, matches ``np.argsort(..., kind="stable")`` bit-for-bit). Every GPU rank-
    binning path in this codebase (``_gpu_resident_rank_bin.rank_bin_codes_batch_gpu_resident`` et al.) is
    documented as "byte-identical to this CPU njit rank binner" - a contract that silently broke on tied
    data (found via test_sf2_combiner_baseline_and_residue_mi_byte_identical: ~6%% MI divergence on a
    -999..999 integer combiner, n=50000, traced to exactly this tie-order mismatch). Mergesort is numba's
    stable-sort kind (verified bit-identical to ``np.argsort(kind="stable")``, which cupy's argsort already
    matches), so this is the ROOT fix, not a narrower patch on one caller - every GPU-vs-CPU rank-binning
    byte-identity claim in the codebase now actually holds instead of only holding on tie-free data.
    Measured cost: 0.93x-1.22x vs the old (arbitrary-tie-order) quicksort default across n in
    {5k,20k,99k,500k} on continuous AND tie-heavy synthetic columns - within noise to a modest ~20%% on this
    one sort step, accepted for a correctness fix (this function was never a documented-arbitrary-tie-order
    contract; nothing should have depended on quicksort's specific, unspecified tie behaviour)."""
    n = x.shape[0]
    sort_idx = np.argsort(x, kind="mergesort")
    out = np.empty(n, dtype=np.int32)
    pos = 0
    base = n // n_bins
    rem = n % n_bins
    for b in range(n_bins):
        size = base + (1 if b < rem else 0)
        for _ in range(size):
            out[sort_idx[pos]] = b
            pos += 1
    return out


@njit(cache=True)  # no fastmath: the NaN test below must not be optimised away (a NaN would reach the selection)
def _quantile_bin_njit(x: np.ndarray, n_bins: int) -> np.ndarray:
    """Quantile-bin a 1-D continuous array into n_bins equi-frequency bins. Returns int32 bin indices in [0, n_bins).

    Same result as :func:`_quantile_bin_sorted_njit` (the stable-argsort reference, kept below) without sorting: the values that open each bin are found with one multi-select, and a row's bin is the number of
    those opening values at or below it. A run of equal values that reaches across a boundary (checked on the selected values: the last value of a bin is not below the first of the next) is placed the way the stable sort places it, after the smaller values and in row order; a column with NaN or too few rows for the bins takes the sorted reference, so the output is identical in every case.
    bench-attempt-rejected (earlier, a different design): one ``np.partition`` per edge re-copied the array per edge and was 2.6x slower than the argsort form; a single multi-select on one copy is what is used here."""
    n = x.shape[0]
    if n < 8 * n_bins or n_bins < 2:
        return _quantile_bin_sorted_njit(x, n_bins)  # type: ignore[no-any-return]  # a numba dispatcher call is untyped for mypy
    for i in range(n):
        if x[i] != x[i]:
            return _quantile_bin_sorted_njit(x, n_bins)  # type: ignore[no-any-return]  # a numba dispatcher call is untyped for mypy
    any_crossing, crossing, cuts, starts = _quantile_crossing_cuts(n, n_bins, x)
    out = np.empty(n, dtype=np.int32)
    if not any_crossing:
        for i in range(n):
            v = x[i]
            lo = 0
            hi = n_bins - 1
            while lo < hi:
                m = (lo + hi) // 2
                if cuts[m] <= v:
                    lo = m + 1
                else:
                    hi = m
            out[i] = lo
        return out
    # Equal values that straddle a bin boundary are placed as the stable sort places them: after every smaller value, in row order. ``below`` is the number of smaller values of such a run
    # (counted once per distinct value) and ``seen`` how many of its rows have been placed so far.
    below, canon, seen = _quantile_crossing_offsets(n_bins, cuts, crossing, n, x)
    for i in range(n):
        v = x[i]
        lo = 0
        hi = n_bins - 1
        while lo < hi:
            m = (lo + hi) // 2
            if cuts[m] <= v:
                lo = m + 1
            else:
                hi = m
        if lo > 0 and cuts[lo - 1] == v and crossing[lo - 1]:
            c = canon[lo - 1]
            position = below[c] + seen[c]
            seen[c] += 1
            blo = 0
            bhi = n_bins - 1
            while blo < bhi:
                m = (blo + bhi) // 2
                if starts[m + 1] <= position:
                    blo = m + 1
                else:
                    bhi = m
            out[i] = blo
        else:
            out[i] = lo
    return out


# Disk caching is off only under CI: a restore-keys prefix fallback there restored a NUMBA_CACHE_DIR from a different source-tree state and the
# kernel then failed with a stale-cache TypingError. Locally the cache is on, saving ~40 s of cold compile per process on a default MRMR fit.
_EVALUATE_GAIN_DISK_CACHE = os.environ.get("CI", "").strip().lower() not in ("1", "true", "yes")


@njit(cache=_EVALUATE_GAIN_DISK_CACHE)
def _jmim_additional_knowledge(X, Z, y, factors_data, factors_nbins, dtype, nexisting, confidence_mode, cached_jmim_MIs, jmim_hit_counter, jmim_discount_only):
    """JMIM joint MI ``I({X, Z}; Y)`` for one ``(X, Z)`` pair, read from / written to the JMIM cache, with the ``nexisting`` exponent applied.

    The aggregator branch of ``evaluate_gain``, split out as its own njit function; the cache arguments may be None (a direct caller without the
    cache wired), which numba narrows below exactly as it did inside ``evaluate_gain``.
    """
    # Build the joint variable index multiset {X, Z} for JMIM's
    # I({X, Z}; Y). np.concatenate is supported in @njit but the
    # numba type-inferencer can fail to unify dtypes when X and Z
    # come from different upstream call paths; coerce both sides
    # to int64 explicitly so the unification is bit-identical to
    # mi()'s internal x = np.asarray(x, dtype=np.int64) cast.
    _x_int = np.asarray(X, dtype=np.int64)
    _z_int = np.asarray(Z, dtype=np.int64)
    xz_combined = np.unique(np.concatenate((_x_int, _z_int)))
    # JMIM joint-MI cache. The multiset {X} u Z recurs
    # across greedy rounds (and across (X, Z) swaps within a round),
    # so memoise the raw mi({X,Z}; y) keyed on arr2str(xz_combined).
    # Stored RAW; nexisting exponent applied below at read time, as
    # the plain-CMI branch does. ``y`` is fixed per fit -> not in key.
    # Independent of the cond-MI cache so JMIM and CMI values never
    # collide under the same arr2str(X)+"|"+arr2str(Z) key.
    # ``cached_jmim_MIs`` / ``jmim_hit_counter`` may be None when a
    # direct caller invokes evaluate_gain without the cache wired
    # (e.g. focused unit tests). Guard with ``is not None`` - numba
    # narrows the Optional so the typed-dict ``in``/index ops only
    # type-check on the non-None branch; the uncached fallback just
    # recomputes mi() every time (pre-cache behaviour).
    _jmim_key = arr2str(xz_combined)
    if (not confidence_mode) and (cached_jmim_MIs is not None) and (_jmim_key in cached_jmim_MIs):
        additional_knowledge = cached_jmim_MIs[_jmim_key]
        if jmim_hit_counter is not None:
            jmim_hit_counter[0] += 1
    else:
        additional_knowledge = mi(
            factors_data=factors_data,
            x=xz_combined, y=y,
            factors_nbins=factors_nbins, dtype=dtype,
        )
        if (not confidence_mode) and (cached_jmim_MIs is not None):
            cached_jmim_MIs[_jmim_key] = additional_knowledge
    if nexisting > 0:
        _disc = additional_knowledge ** (nexisting + 1)
        if jmim_discount_only and _disc > additional_knowledge:
            # joint MI > 1 -> the exponent would amplify; clamp to discount-only (never increase the joint MI).
            pass
        else:
            additional_knowledge = _disc
    return additional_knowledge


@njit(cache=_EVALUATE_GAIN_DISK_CACHE)
def _fleuret_additional_knowledge(
    X: Any, Z: Any, y: Any, factors_data: Any, factors_nbins: Any, nexisting: Any, entropy_cache: Any, cached_cond_MIs: Any, cached_jmim_MIs: Any, jmim_hit_counter: Any,
    can_use_x_cache: Any, can_use_y_cache: Any, dtype: Any, confidence_mode: Any, use_su: Any, use_jmim: Any, jmim_discount_only: Any, use_mm: Any,
) -> Any:
    """Redundancy term ``additional_knowledge`` for one ``(X, Z)`` pair: cached conditional MI (or its SU / JMIM variant) with the ``nexisting`` exponent applied.

    The Fleuret branch of ``evaluate_gain``, split out as its own njit function so the kernel body stays within the function-length gate.
    """
    key_found = False
    if not confidence_mode:
        #
        # ``arr2str`` already uses ``_`` as its element
        # separator, so the prior boundary ``"_"`` between
        # ``arr2str(X)`` and ``arr2str(Z)`` aliased every
        # partition of the multiset ``X u Z``. Confirmed:
        # X=[1,2] Z=[3,4], X=[1] Z=[2,3,4], X=[1,2,3] Z=[4]
        # all produced ``"1_2_3_4"`` -> wrong-answer cache
        # hits, silent. Affects every config with
        # ``max_veteranes_interactions_order >= 2`` or
        # engineered multi-element X/Z, biasing
        # conditional-MI scoring. Switching to ``"|"``
        # (a character ``arr2str`` cannot emit) makes the
        # boundary unambiguous.
        #
        # cache the RAW (pre-exponent) conditional MI
        # and apply ``** (nexisting + 1)`` at every
        # read site. Pre-fix the cache stored the
        # already-exponentiated value but the key
        # omitted ``nexisting``, so a subsequent
        # ``(X, Z)`` lookup with a different
        # ``nexisting`` returned the WRONG exponent
        # (and silently biased the Fleuret/CMIM
        # redundancy score). Caching the raw value
        # lets all nexisting tiers share the
        # underlying CMI compute.
        key = arr2str(X) + "|" + arr2str(Z)
        if key in cached_cond_MIs:  # caller always supplies a dict on the non-JMIM path (see docstring)
            additional_knowledge = cached_cond_MIs[key]
            # Apply the nexisting exponent at read time.
            if nexisting > 0:
                additional_knowledge = additional_knowledge ** (nexisting + 1)
            key_found = True

    if not key_found:

        # JMIM aggregator (Bennasar 2015).
        # When the thread-local JMIM toggle is on, replace
        # Fleuret's conditional MI ``I(X; Y | Z)`` with the
        # joint MI ``I({X, Z}; Y)``. Both feed the same
        # outer ``min_k`` aggregator. JMIM preserves the
        # synergistic information that CMIM rejects on
        # multi-collinear groups (Brown 2012 / Bennasar 2015
        # framework). Caching skipped on the JMIM branch
        # because the joint-MI computation has its own
        # per-pair entropy cost; future sprint may add a
        # joint-MI cache keyed on the multiset {X, Z, Y}.
        # SU branch bypasses entropy caches because the SU denominator
        # is path-dependent and re-using cached unconditional CMI numbers would
        # silently desync the normalization. Cheap (no caching) is correct here;
        # legacy raw-CMI path keeps cache wired in. ``use_su`` AND ``use_jmim``
        # are threaded from the Python-level caller (evaluate_candidate ->
        # evaluate_gain) - the njit kernel cannot read the Python thread-locals
        # at runtime (IMPORT_NAME / dynamic Python call are unsupported in @njit).
        if use_jmim:
            additional_knowledge = _jmim_additional_knowledge(
                X, Z, y, factors_data, factors_nbins, dtype, nexisting, confidence_mode, cached_jmim_MIs, jmim_hit_counter, jmim_discount_only,
            )
        elif use_su:
            additional_knowledge = conditional_symmetric_uncertainty(
                factors_data=factors_data, x=X, y=y, z=Z,
                factors_nbins=factors_nbins, dtype=dtype,
            )
        else:
            additional_knowledge = conditional_mi(
                factors_data=factors_data,
                x=X,
                y=y,
                z=Z,
                var_is_nominal=None,
                factors_nbins=factors_nbins,
                entropy_cache=entropy_cache,
                can_use_x_cache=can_use_x_cache,
                can_use_y_cache=can_use_y_cache,
                dtype=dtype,
                use_mm=use_mm,  # N-F2: redundancy carries the SAME MM bias correction as the relevance
            )

        #
        # write the RAW additional_knowledge to the
        # cache BEFORE applying the nexisting
        # exponent. Apply the exponent only to the
        # local ``additional_knowledge`` after the
        # write. This makes the cache nexisting-
        # independent and lets all callers share
        # the underlying CMI compute.
        # JMIM manages its OWN cache + exponent above
        # (cached_jmim_MIs, keyed on the {X} u Z multiset). Skip the
        # shared cond-MI write/exponent here so JMIM values are neither
        # stored under the CMI key nor double-exponentiated. The SU
        # branch is uncached by design; the plain-CMI branch keeps the
        # legacy write/exponent. ``key`` is undefined when use_jmim, so
        # this guard also avoids a NameError under JMIM.
        if not use_jmim:
            if not confidence_mode:
                cached_cond_MIs[key] = additional_knowledge  # caller always supplies a dict on the non-JMIM path
            if nexisting > 0:
                additional_knowledge = additional_knowledge ** (nexisting + 1)
    return additional_knowledge


# evaluate_gain parameter notes (kept beside the definition; the signature itself is packed):
# cached_jmim_MIs: JMIM joint-MI cache. Mirrors ``cached_cond_MIs`` but keyed on the multiset ``{X} u Z`` only (``arr2str(xz_combined)``); y is fixed per fit so it is not
#   part of the key. Stores the RAW ``mi({X,Z}; y)``; the nexisting exponent is applied at READ time, exactly as the plain-CMI branch does for its cache.
# jmim_hit_counter: 1-element int64 array hit counter for the JMIM cache. An array
#   (not a scalar) so the @njit kernel can mutate it in place and the caller can
#   read the post-loop count - proves the cache actually HITS, not just that it
#   is harmless. Counter[0] is incremented on every JMIM cache read-hit.
# use_su: SU normalization toggle threaded from process_candidates / mrmr_fit_impl.
# use_jmim: JMIM aggregator toggle, threaded from the Python-level caller for the same reason as ``use_su`` - ``@njit`` cannot do ``from X import Y`` or call a non-njit
#   thread-local reader at runtime (IMPORT_NAME opcode is unsupported).
# jmim_discount_only: MLFRAME_JMIM_EXPONENT_DISCOUNT_ONLY, read by the Python caller (see jmim_exponent_discount_only)
# use_mm: N-F2: Miller-Madow toggle for the redundancy CMI, threaded like use_su so the redundancy carries the SAME bias correction as the MM relevance.
#   Default False -> plug-in redundancy, unchanged.
@njit(cache=_EVALUATE_GAIN_DISK_CACHE)
def evaluate_gain(
    current_gain: float, last_checked_k: int, direct_gain: float, X: np.ndarray, y: np.ndarray, nexisting: int, best_gain: float,
    factors_data: np.ndarray, factors_nbins: np.ndarray, selected_vars: np.ndarray, mrmr_relevance_algo: str = "fleuret",
    mrmr_redundancy_algo: str = "fleuret", max_veteranes_interactions_order: int = 2, extra_knowledge_multipler: float = -1.0,
    sink_threshold: float = -1.0, entropy_cache: dict | None = None, cached_cond_MIs: dict | None = None, cached_jmim_MIs: dict | None = None,
    jmim_hit_counter: np.ndarray | None = None, can_use_x_cache: bool = False, can_use_y_cache: bool = False, dtype: type = np.int32,
    confidence_mode: bool = False, max_confirmation_cand_nbins: int = MAX_CONFIRMATION_CAND_NBINS, use_su: bool = False, use_jmim: bool = False,
    jmim_discount_only: bool = False, use_mm: bool = False,
) -> tuple:
    """``max_confirmation_cand_nbins`` is a parameter (not a baked-in constant). Default mirrors the legacy value; ``MRMR.fit`` overrides with ``quantization_nbins ** interactions_max_order * 2`` unless the user pins it explicitly."""

    positive_mode = False
    stopped_early = False
    sink_reasons = None

    k = 0
    for interactions_order in range(max_veteranes_interactions_order):
        # calls the njit-compiled core directly rather than pyutilz's plain-Python validating wrapper
        # (generate_combinations_recursive_njit): a bare Python function with a `raise` cannot be called
        # from this @njit context in nopython mode - numba reports "Untyped global name" on any cold
        # compile (a warm on-disk cache from before pyutilz's wrapper regressed to plain Python can mask
        # this). r = interactions_order + 1 is always >= 1 here, so the wrapper's r<0 validation is moot.
        # .astype(np.int32), not np.array(selected_vars, dtype=np.int32): now that every caller passes
        # a real ndarray (see the call site's own comment on why the reflected-list-typed np.array()
        # call this replaced existed at all), nopython mode's np.array() no longer accepts an ndarray
        # input ("not allowed in a homogeneous sequence") -- .astype() is the correct nopython-mode
        # array-to-array dtype cast and works for both an int64 and an already-int32 input.
        combs = _generate_combinations_recursive_njit_core(selected_vars.astype(np.int32), interactions_order + 1)[::-1]

        for Z in combs:

            if k > last_checked_k:
                if confidence_mode and count_cand_nbins(Z, factors_nbins) > max_confirmation_cand_nbins:
                    additional_knowledge = 0.0  # this is needed to skip checking against hi cardinality approved factors
                else:
                    # Gated on mrmr_redundancy_algo, not mrmr_relevance_algo: the conditional-MI machinery
                    # below is the REDUNDANCY mechanism (I(X;Y|Z) checked against every already-selected Z),
                    # independent of how relevance itself was scored (direct_gain, computed by the caller
                    # before entering evaluate_gain at all). Gating on mrmr_relevance_algo instead left
                    # `additional_knowledge` completely unset -- UnboundLocalError -- for the entirely valid
                    # mrmr_relevance_algo="pld" + mrmr_redundancy_algo="fleuret" (the redundancy default)
                    # combination, since neither branch of the original condition ever ran for it. The `or
                    # mrmr_relevance_algo == "fleuret"` disjunct preserves every previously-reachable
                    # combination byte-for-byte (relevance="fleuret" always entered this branch before,
                    # regardless of redundancy_algo, so it still does). mrmr_redundancy_algo in
                    # {"pld_max", "pld_mean"} is a separate, still-unimplemented gap (see
                    # _screen_predictors.py's docstring) -- not addressed here, and not newly broken by this
                    # fix (that combination never worked either way).
                    if mrmr_redundancy_algo == "fleuret" or mrmr_relevance_algo == "fleuret":
                        # additional_knowledge = I(X; Y | Z) = H(X, Z) + H(Y, Z) - H(Z) - H(X, Y, Z); I(X, Z) = entropy_x + entropy_z - entropy_xz.
                        additional_knowledge = _fleuret_additional_knowledge(
                            X, Z, y, factors_data, factors_nbins, nexisting, entropy_cache, cached_cond_MIs, cached_jmim_MIs, jmim_hit_counter,
                            can_use_x_cache, can_use_y_cache, dtype, confidence_mode, use_su, use_jmim, jmim_discount_only, use_mm,
                        )

                    # Account for possible extra knowledge from conditioning on Z; must update best_gain globally and log such cases. Order of discovery is
                    # not guaranteed, but cases are too precious to ignore. Also enables skipping higher-order interactions containing all approved candidates.
                    # CAVEAT (extra_knowledge_multipler > 0): this is NOT MRMR redundancy for that candidate. Once a single Z yields additional_knowledge above the
                    # multiplied direct_gain, ``positive_mode`` flips and the score becomes the MAX over Z of that extra-knowledge term, NOT the MIN-redundancy MRMR
                    # objective - the candidate is ranked by its single BEST synergistic relationship instead of its worst-case redundancy. Default is -1.0 (off), so
                    # the default selection stays pure MRMR; enable this only when you deliberately want a synergy-seeking (max-extra-knowledge) ranking.
                    if extra_knowledge_multipler > 0 and additional_knowledge > direct_gain * extra_knowledge_multipler:
                        if not positive_mode:
                            current_gain = additional_knowledge
                            positive_mode = True
                        else:
                            # rare chance that a candidate has many excellent relationships
                            if additional_knowledge > current_gain:
                                current_gain = additional_knowledge

                if not positive_mode and (additional_knowledge < current_gain):

                    current_gain = additional_knowledge

                    if best_gain is not None and current_gain <= best_gain:
                        # No point checking other Zs - current_gain already won't beat best_gain (assuming best_gain was estimated confidently; checked at end).
                        # Also fix what Z caused X (the most) to sink.
                        if sink_threshold > -1 and current_gain < sink_threshold:
                            sink_reasons = Z

                        stopped_early = True
                        return stopped_early, current_gain, k, sink_reasons
            k += 1

    return stopped_early, current_gain, k - 1, sink_reasons


# Typed uint64 zero used as the default ``base_seed`` for the njit permutation kernels. Built once as a
# module-level singleton so the argument default keeps its exact ``np.uint64`` dtype (the kernels do
# ``state >> np.uint64(..)`` arithmetic) without a function call in the signature default.
_DEFAULT_BASE_SEED = np.uint64(0)


@njit(cache=True)
def _fleuret_shuffle_col_lcg(col: np.ndarray, state: np.uint64) -> np.uint64:
    """Inline Fisher-Yates over a 1-D column slice using the same LCG schedule as ``permutation.shuffle_arr_lcg``.

    Returns the post-shuffle state so the caller threads it across per-column and per-permutation calls. Replaces ``np.random.shuffle(data_copy[:, idx])`` whose global numpy RNG state is process-wide
    and races under joblib parallel workers / multi-suite invocations.
    """
    n = len(col)
    for j in range(n - 1, 0, -1):
        state = state * np.uint64(6364136223846793005) + np.uint64(1442695040888963407)
        k = int(state >> np.uint64(33)) % (j + 1)
        tmp = col[j]
        col[j] = col[k]
        col[k] = tmp
    return state


@njit(cache=True)
def get_fleuret_criteria_confidence(
    data_copy: np.ndarray,
    factors_nbins: np.ndarray,
    x: tuple,
    y: tuple,
    selected_vars: list,
    npermutations: int,
    bootstrapped_gain: float,
    max_failed: int,
    nexisting: int,
    mrmr_relevance_algo: str = "fleuret",
    mrmr_redundancy_algo: str = "fleuret",
    max_veteranes_interactions_order: int = 1,
    extra_knowledge_multipler: float = -1.0,
    sink_threshold: float = -1.0,
    cached_cond_MIs: Optional[dict] = None,
    entropy_cache: Optional[dict] = None,
    extra_x_shuffling: bool = True,
    dtype: type = np.int32,
    base_seed: np.uint64 = _DEFAULT_BASE_SEED,
    use_su: bool = False,  # Threaded from Python-level parallel_fleuret.
    use_mm: bool = False,  # N-F2: threaded from parallel_fleuret (MM redundancy consistency).
    use_jmim: bool = False,  # S-F2: threaded from parallel_fleuret (JMIM confirmation statistic).
    jmim_discount_only: bool = False,  # threaded like use_jmim
) -> tuple:
    """Sub to njit work with random shuffling as well.

    Zero-permutation guard returns ``(0, 0)`` cleanly. The legacy code raised ``UnboundLocalError`` on ``i`` because the for-loop never assigned it.

    ``base_seed`` threads through inline LCG Fisher-Yates so per-permutation column shuffles are reproducible from a single seed; pre-fix code used ``np.random.shuffle`` on the process-global numpy RNG which raced under joblib parallel workers and made two parallel suite calls produce non-deterministic confidence values.
    """
    if npermutations == 0:
        return 0, 0

    # Mutate-and-restore: ``data_copy`` may ALIAS the caller's real ``factors_data`` (screen no longer takes a whole-matrix copy), so save the
    # original values of the ONLY columns we permute in place (y always, x when ``extra_x_shuffling``) and restore them before returning. Extra RAM
    # is O((len(y)+len(x))*n), never O(p*n). Shuffles stay cumulative WITHIN this call (matching the prior per-iteration behaviour); the restore only
    # affects cross-call state, which the parallel worker path already reset by copying, so the serial path now matches it.
    n_rows = data_copy.shape[0]
    _saved_y = np.empty((len(y), n_rows), dtype=data_copy.dtype)
    for _sj in range(len(y)):
        _saved_y[_sj, :] = data_copy[:, y[_sj]]
    _saved_x = np.empty((len(x), n_rows), dtype=data_copy.dtype)
    if extra_x_shuffling:
        for _sj in range(len(x)):
            _saved_x[_sj, :] = data_copy[:, x[_sj]]

    nfailed = 0
    _i = 0
    _lcg_state = np.uint64(base_seed) * np.uint64(2654435761) + np.uint64(1)
    # np.array(...), not the raw list: evaluate_gain is @njit and a plain Python list slipping
    # through gets numba's "reflected list" typing, which breaks on any ndarray-only method the
    # kernel calls (e.g. `.astype()`) -- see the identical conversion + rationale at evaluate_gain's
    # other call site in evaluation.py. Hoisted out of the permutation loop below (``selected_vars``
    # never changes across permutations) as well as out of the call's kwargs (rather than an inline
    # ternary there) because this function is itself @njit -- numba's bytecode interpreter rejects
    # a conditional expression inlined into a many-kwarg CALL_FUNCTION_EX call.
    if len(selected_vars) > 0:
        selected_vars_arr = np.array(selected_vars, dtype=np.int64)
    else:
        selected_vars_arr = np.empty(0, dtype=np.int64)
    for _i in range(npermutations):

        for idx in y:
            _lcg_state = np.uint64(_fleuret_shuffle_col_lcg(data_copy[:, idx], _lcg_state))

        if extra_x_shuffling:
            for idx in x:
                _lcg_state = np.uint64(_fleuret_shuffle_col_lcg(data_copy[:, idx], _lcg_state))

        _stopped_early, current_gain, _k, _sink_reasons = evaluate_gain(
            current_gain=LARGE_CONST,
            last_checked_k=-1,
            X=x,
            y=y,
            best_gain=None,
            factors_data=data_copy,
            factors_nbins=factors_nbins,
            selected_vars=selected_vars_arr,
            nexisting=nexisting,
            direct_gain=bootstrapped_gain,
            mrmr_relevance_algo=mrmr_relevance_algo,
            mrmr_redundancy_algo=mrmr_redundancy_algo,
            max_veteranes_interactions_order=max_veteranes_interactions_order,
            extra_knowledge_multipler=extra_knowledge_multipler,
            sink_threshold=sink_threshold,
            cached_cond_MIs=cached_cond_MIs,
            entropy_cache=entropy_cache,
            can_use_x_cache=not extra_x_shuffling,
            can_use_y_cache=False,
            confidence_mode=True,
            use_su=use_su,
            use_mm=use_mm,
            use_jmim=use_jmim,
            jmim_discount_only=jmim_discount_only,
        )

        if current_gain >= bootstrapped_gain:
            nfailed += 1
            if nfailed >= max_failed:
                break

    # Restore the permuted columns so an aliased ``factors_data`` is left byte-identical to entry (reached on both normal and early-break exit).
    for _sj in range(len(y)):
        data_copy[:, y[_sj]] = _saved_y[_sj, :]
    if extra_x_shuffling:
        for _sj in range(len(x)):
            data_copy[:, x[_sj]] = _saved_x[_sj, :]

    return nfailed, _i + 1

# Usability-pool njit kernels hosted in _numba_utils_usability.py (re-exported so importers keep one path).
from ._usability_njit_pool_codes import (  # noqa: F401
    _NJIT_UNARY_OP_CODES,
    _NJIT_USABILITY_BINARY_OP_CODES,
    njit_unary_codes_or_none,
    njit_binary_codes_or_none,
)
from ._numba_utils_usability import (  # noqa: F401
    _USABILITY_DEGENERATE_REL_TOL,
    _apply_unary,
    _apply_binary,
    _qbin_into,
    _marginal_mi_njit,
    _pair_combo_mi_njit,
    _pair_combo_mi_njit_parallel,
    _pair_combo_mi_njit_table,
    _pair_combo_mi_njit_table_parallel,
)

# Cross-module @njit helpers hosted in _numba_utils_cross_module.py (re-exported so importers keep one path).
from ._numba_utils_cross_module import (  # noqa: F401
    _accept_split_serial,
    _analytic_p_value,
    _cmi_from_counts,
    _codes_at_rows,
    _codes_from_cuts,
    _compact_labels,
    _conditional_shuffle_within_strata,
    _count_candidates_njit,
    _cuts_tie_free,
    _edge_value,
    _full_conditional_shuffle_ipf,
    _gammaincc_nb,
    _mdlp_cuts_presorted,
    _mdlp_dfs,
    _mi_codes,
    _mi_codes_njit,
    _n_unique_codes,
    _presorted_dedup,
    corrected_mi_best,
    _relevance_mi_1var_fused,
    _supervised_edges,
    column_tie_runs,
    quantile_codes_block,
    gather_binary_stream,
    root_decision_binary,
    root_decision_tie_free,
    stratified_cmi,
    stratified_cmi_null,
)
