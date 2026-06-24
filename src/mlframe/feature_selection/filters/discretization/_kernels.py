"""Low-level 2-D quantile-discretisation numba kernels.

Carved verbatim out of ``discretization/__init__.py`` (LOC-budget sibling
re-export, see mlframe/CLAUDE.md "Monolith split"). These are the per-column
quantile-edge + searchsorted kernels consumed by
``discretize_2d_quantile_batch`` in the parent module. They are moved here
unchanged (comments / docstrings / decorators preserved exactly); the parent
re-exports every public name so the package's import surface is unchanged
(e.g. ``from mlframe.feature_selection.filters.discretization import
_quantile_edges_2d_njit`` still works).
"""
from __future__ import annotations

import math

import numpy as np
from numba import njit, prange


@njit(parallel=True, nogil=True, cache=True)
def _quantile_edges_2d_njit(arr2d: np.ndarray, quantiles: np.ndarray, kths: np.ndarray, edges_out: np.ndarray) -> None:
    """Per-column linear-interpolation quantiles, BIT-IDENTICAL to
    ``np.percentile(arr2d, quantiles, axis=0)`` on a NaN-free buffer.

    Writes ``edges_out`` of shape ``(len(quantiles), n_cols)``; ``edges_out[q, j]`` is the
    ``quantiles[q]``-th percentile (0..100) of column ``j``. Replaces the numpy
    ``np.percentile(axis=0)`` call in ``discretize_2d_quantile_batch`` whose internal
    ``ndarray.partition`` was the FE-sweep's single dominant numpy hotspot (call-site
    profile on scene 1500x299: 114.5s / 20% of fit in ``partition``, 14208 calls -- the
    vectorised C partition re-partitions the FULL (n_rows x n_cols) buffer ONCE PER
    quantile (n_bins+1 of them) per discretise; this kernel sorts each column ONCE in a
    ``nogil`` per-column loop and reads ALL quantiles from the sorted column).

    BIT-IDENTITY to numpy's default ``method='linear'`` percentile (verified across
    float32/float64, ties, constant + heavy-tail columns, and every nbins):
      * SORT in the INPUT dtype: ``np.percentile(arr2d, axis=0)`` partitions the array in
        its OWN dtype (verified the 2-D ``axis=0`` float32 call equals the per-column 1-D
        float32 path), so the selected order statistics ``col[lo]`` / ``col[lo+1]`` must be
        the float32 values -- promoting to float64 BEFORE the sort can reorder float32 ties
        at distinct float64 values and diverge by ~1 ULP.
      * LERP in float64: numpy keeps the interpolation WEIGHT ``t`` in float64 even for a
        float32 array, so ``col[lo]`` is promoted to float64 in the multiply and the
        interpolation result is float64 -- matched here via ``float(col[lo])`` + a float64
        ``t``. (Net: select float32 order statistics, lerp them in float64 = numpy exactly.)
      * Full ``np.sort`` produces the exact ascending order statistics numpy's
        ``introselect`` partition selects at indices ``lo``/``lo+1`` (a sort IS a valid
        partition at every index), so ``col[lo]`` / ``col[lo+1]`` are identical values.
      * Virtual index: ``v = (q/100) * (n-1)``; ``lo = floor(v)``; numpy's exact ``_lerp``
        (``a + (b-a)*t`` for t<0.5, ``b - (b-a)*(1-t)`` for t>=0.5) with the ``lo == n-1``
        clamp -- the asymmetric form numpy uses to stay monotone + endpoint-exact.

    ``parallel=True`` (prange over columns) + ``nogil``: a SERIAL sort-per-column is actually
    ~0.6x SLOWER than numpy's vectorised C ``partition`` single-threaded (measured), so the win
    comes ONLY from spreading the per-column sorts across cores. The default scene FE path runs
    ``check_prospective_fe_pairs`` (-> ``_compute_one_fe_chunk`` -> here) on the MAIN thread for
    the common ``len(X) < 50000`` case (the ``else`` joblib branch only fires at >=50000 rows),
    so numba-parallel does NOT nest inside Python threads there. ``nogil`` is also set so that
    on the rare wide-data joblib (``backend="threading"``) path the GIL is released; numba's
    threading layer serialises nested parallel regions rather than deadlocking, so the worst
    case there is "no extra speedup", never a hang. Bit-identity is independent of thread count
    (each column is reduced independently). bench (scene FE buffer 1500-2407 x 4000-8000 cols):
    serial 0.62-0.67x vs numpy; parallel restores the win on multi-core.

    ``arr2d`` is ``(n_rows, n_cols)`` at its native dtype (float32/float64 -- one numba
    specialisation each). ``quantiles`` is float64 in [0, 100]. NaN handling is NOT done
    here (the caller routes NaN-bearing buffers to ``np.nanpercentile``); a NaN in a column
    would sort last and bias the edges, exactly as a raw ``np.percentile`` (non-nan) would,
    so the caller's NaN guard is what preserves correctness, identically to before.
    """
    n_rows = arr2d.shape[0]
    n_cols = arr2d.shape[1]
    n_q = quantiles.shape[0]
    if n_rows == 0 or n_cols == 0:
        return
    # prange over COLUMNS: each iteration owns a private ``col`` scratch (numba allocates it
    # per-iteration on the worker thread, no cross-thread aliasing) so the per-column sort runs
    # in parallel across cores. This is what turns the serial sort-per-column (which is SLOWER
    # than numpy's vectorised C partition single-threaded) into a net win: at K=4000-8000 FE
    # columns the parallel sort beats numpy's single-threaded partition by spreading the work.
    for j in prange(n_cols):
        # Sort each column in the INPUT dtype: numpy's ``np.percentile(axis=0)`` partitions the
        # array IN ITS OWN DTYPE (a float32 buffer is partitioned in float32 -- verified that the
        # 2-D ``axis=0`` call equals the per-column 1-D float32 path), so the selected order
        # statistics ``col[lo]`` / ``col[lo+1]`` must be the float32 values, not float64-promoted
        # ones (promoting before the sort can reorder float32 ties at distinct float64 values).
        col = np.empty(n_rows, dtype=arr2d.dtype)
        for r in range(n_rows):
            col[r] = arr2d[r, j]
        # PARTITION not SORT (2026-06-17): the quantile edges only read the order
        # statistics at ``kths`` (the ``lo``/``lo+1`` indices the lerp below touches, plus the
        # ``n-1`` clamp) -- ``np.partition(col, kths)[k]`` equals ``col.sort()[k]`` EXACTLY at
        # every ``k in kths`` (introselect places each kth at its final sorted position), so the
        # edges stay bit-identical to the full-sort path while dropping the cost from O(n log n)
        # to O(n) per column. Measured ~20% faster at n=100k (447 vs 555 ms), widening with n.
        col = np.partition(col, kths)
        for qi in range(n_q):
            # ``v`` / ``t`` are float64 (the quantile virtual index). numpy keeps the lerp
            # WEIGHT in float64 even for a float32 array, so ``col[lo]`` (float32) is promoted
            # to float64 in the multiply and the interpolation result is float64 -- matching
            # numpy bit-for-bit. (Sort in float32 order statistics, lerp them in float64.)
            v = (quantiles[qi] / 100.0) * (n_rows - 1)
            lo = int(math.floor(v))
            if lo >= n_rows - 1:
                edges_out[qi, j] = col[n_rows - 1]
            else:
                a = float(col[lo])
                b = float(col[lo + 1])
                t = v - lo
                # numpy's exact ``_lerp`` (numpy/lib/function_base): ``a + (b-a)*t`` for
                # t < 0.5 and ``b - (b-a)*(1-t)`` for t >= 0.5 -- the asymmetric form numpy
                # uses to keep the result monotone + endpoint-exact. Matching this branch
                # (in float64) makes the edges bit-identical to ``np.percentile``.
                diff_b_a = b - a
                if t >= 0.5:
                    edges_out[qi, j] = b - diff_b_a * (1.0 - t)
                else:
                    edges_out[qi, j] = a + diff_b_a * t


# bench-attempt-rejected (2026-06-07): ``fastmath=True`` on the searchsorted kernels (Q6).
# UNSAFE: fastmath asserts no-NaN, which silently breaks the load-bearing NaN -> rightmost-bin
# contract. Verified directly -- on a 5%-NaN buffer the plain kernel assigns NaN code 9 (rightmost,
# correct) while the fastmath kernel assigns code 0, which would re-bin NaN columns and DRIFT the
# selection. The speedup is also negligible (1.802ms -> 1.777ms, ~1%, the loop is integer-compare
# bound -- no float-reassociation to win). NOT applied. (Integer/count kernels gain nothing from
# fastmath since it is a float-only flag; the MI log-sum/div kernels are excluded by the user.)
@njit(nogil=True, cache=True)
def _searchsorted_2d_right_njit(edges_inner: np.ndarray, arr2d: np.ndarray, out: np.ndarray) -> None:
    """Per-column ``np.searchsorted(edges_inner[:, j], arr2d[:, j], side='right')`` in ONE
    nogil kernel, writing ordinal codes into ``out``.

    Replaces the Python ``for j in range(n_cols): out[:, j] = np.searchsorted(...)`` loop
    in ``discretize_2d_quantile_batch`` (370k dispatched searchsorted calls on scene's
    FE buffers -> serial-dispatch-bound). BIT-IDENTICAL to numpy's ``searchsorted(side='right')``:

      * ``side='right'`` returns the count of edges ``<= v`` (largest ``i`` with
        ``edges[:i] <= v``); the branch ``v < edges[mid] -> hi=mid else lo=mid+1``
        reproduces that exactly (ties advance ``lo`` -> rightmost).
      * NaN ``v``: every ``v < edges[mid]`` is False (IEEE), so ``lo`` walks to the end
        -> returns ``len(edges)``, the SAME index numpy assigns NaN (sorts after all),
        so a NaN row lands in the post-max bin identically to the per-column numpy path.

    SERIAL + ``nogil=True`` ON PURPOSE (not ``parallel=True``): ``discretize_2d_quantile_batch``
    is called inside ``_compute_one_fe_chunk`` under joblib ``backend="threading"`` (the FE
    pair-search dispatch), so a numba ``parallel=True`` prange here would nest numba-parallel
    inside Python threads and deadlock the threading layer (the same hazard that keeps
    ``_materialise_chunk_njit`` serial). With ``nogil`` the joblib threads run this kernel
    concurrently across cores; on the n_jobs=1 path it runs single-threaded but still removes
    the per-column numpy dispatch overhead.

    ``edges_inner`` is ``edges[1:-1]`` (the interior bin edges), shape ``(n_edges, n_cols)``,
    always float64; ``arr2d`` is ``(n_rows, n_cols)`` at its NATIVE dtype (float32 or float64 --
    numba compiles a specialisation per dtype). The per-element ``arr2d[r,j] < edges_inner[mid,j]``
    promotes a float32 value to float64 against the float64 edge, byte-identically to numpy's
    ``searchsorted(float64_edges, float32_col)``; this lets the caller pass the full-width FE
    buffer WITHOUT a float64 upcast copy (which would double a multi-GB float32 buffer and OOM).
    ``out`` is the pre-allocated ordinal-code array.
    """
    n_rows = arr2d.shape[0]
    n_cols = arr2d.shape[1]
    n_edges = edges_inner.shape[0]
    for j in range(n_cols):
        for r in range(n_rows):
            v = arr2d[r, j]
            lo = 0
            hi = n_edges
            while lo < hi:
                mid = (lo + hi) >> 1
                if v < edges_inner[mid, j]:
                    hi = mid
                else:
                    lo = mid + 1
            out[r, j] = lo


@njit(nogil=True, cache=True)
def _quantile_codes_1d_njit(arr: np.ndarray, quantiles: np.ndarray, kths: np.ndarray, codes_out: np.ndarray) -> None:
    """Fused 1-D quantile discretiser: BIT-IDENTICAL to
    ``np.searchsorted(np.nanpercentile(arr, quantiles)[1:-1], arr, side='right')`` on a
    NaN-FREE ``arr`` (caller gates), in ONE nogil kernel that partitions the column ONCE.

    The 1-D ``discretize_array(method='quantile')`` path was the dominant CPU residual on the
    golden F2 100k fit: 9754 calls, each doing ``np.nanpercentile`` (whose ``_quantile`` runs
    ``ndarray.partition`` once PER quantile -> ~7.8s ``partition``) plus ``np.searchsorted``
    (~7.7s). This kernel collapses both into a single ``np.partition(buf, kths)`` (O(n), reads
    exactly the order statistics the lerp touches -- the same one-partition technique already
    shipped in ``_quantile_edges_2d_njit``) followed by an inline binary search per element.

    bench-attempt-rejected (2026-06-24): wiring this into the 1-D ``discretize_array`` quantile
    path was a NET LOSS on the dev HW (single-column microbench, NaN-free float32, 200 reps:
    n=10k/nb=10 0.74x, n=10k/nb=20 0.70x, n=100k/nb=10 0.67x, n=100k/nb=20 0.58x vs numpy
    ``nanpercentile``+``searchsorted``). The 2-D batch sibling wins by amortising dispatch over
    MANY columns + a column ``prange``; a single serial 1-D column has neither, and numpy's
    vectorised C ``partition`` beats a numba scalar partition-copy + scalar binary-search loop.
    KEPT here (feedback_keep_all_kernel_versions) for re-bench on other HW: ``kths`` is the
    ``_quantile_edges_2d_njit`` recipe (the lo/lo+1 read indices, with the n-1 clamp).

    BIT-IDENTITY (same proof as ``_quantile_edges_2d_njit`` + ``_searchsorted_2d_right_njit``):
      * Edges: ``np.partition(col_copy, kths)[k] == np.sort(col_copy)[k]`` at every ``k in kths``
        (introselect places each kth at its final sorted position); the lerp ``v=(q/100)*(n-1)``,
        ``lo=floor(v)``, asymmetric ``_lerp`` (a+(b-a)t for t<0.5 else b-(b-a)(1-t)) with the
        ``lo==n-1`` clamp is numpy's exact ``method='linear'`` percentile. On a NaN-free array
        ``np.percentile == np.nanpercentile`` bit-for-bit.
      * Codes: ``side='right'`` binary search over the INTERIOR edges (``edges[1:-1]``) reproduces
        ``np.searchsorted(..., side='right')`` exactly (ties advance ``lo`` -> rightmost bin).
      * dtype: ``arr`` (float32/64) promotes to float64 in the float64 edge compares, identical to
        ``searchsorted(float64_edges, col)``. ``codes_out`` carries the caller's widened code dtype.

    NaN is NOT handled here (caller routes NaN-bearing arrays to the numpy ``nanpercentile`` path);
    a NaN would sort last and bias the edges, exactly as a raw ``np.percentile`` would, so the
    caller's NaN gate is what preserves correctness -- identical contract to the 2-D kernels.

    SERIAL + ``nogil`` (not ``parallel``): a single 1-D column has no inner parallelism to spread
    (the win is eliminating per-quantile re-partition + numpy dispatch), and the FE pair-search
    already calls this from inside joblib ``backend="threading"`` workers, so a nested numba prange
    would risk the threading-layer deadlock the 2-D serial kernel documents.
    """
    n = arr.shape[0]
    n_q = quantiles.shape[0]
    # Partition a private copy at exactly the order-statistic indices the lerp reads (O(n)).
    buf = arr.copy()
    buf = np.partition(buf, kths)
    # Compute the n_q quantile edges from the partitioned buffer (matches np.percentile linear).
    edges = np.empty(n_q, dtype=np.float64)
    for qi in range(n_q):
        v = (quantiles[qi] / 100.0) * (n - 1)
        lo = int(math.floor(v))
        if lo >= n - 1:
            edges[qi] = buf[n - 1]
        else:
            a = float(buf[lo])
            b = float(buf[lo + 1])
            t = v - lo
            diff_b_a = b - a
            if t >= 0.5:
                edges[qi] = b - diff_b_a * (1.0 - t)
            else:
                edges[qi] = a + diff_b_a * t
    # Interior edges only (drop the 0th/last like ``bins_edges[1:-1]``); side='right' search.
    n_edges = n_q - 2
    for r in range(n):
        x = arr[r]
        lo = 0
        hi = n_edges
        while lo < hi:
            mid = (lo + hi) >> 1
            # edges index +1 to skip the dropped 0th edge (interior == edges[1:-1]).
            if x < edges[mid + 1]:
                hi = mid
            else:
                lo = mid + 1
        codes_out[r] = lo


@njit(parallel=True, nogil=True, cache=True)
def _searchsorted_2d_right_njit_parallel(edges_inner: np.ndarray, arr2d: np.ndarray, out: np.ndarray) -> None:
    """``parallel=True`` (prange over COLUMNS) twin of ``_searchsorted_2d_right_njit`` --
    BYTE-IDENTICAL output, only the outer column loop is a numba ``prange`` so the per-column
    binary searches spread across cores (OPT-A, 2026-06-07).

    Kept as a SEPARATE kernel (``feedback_keep_all_kernel_versions``): the serial ``nogil``
    variant above MUST stay for the joblib ``backend="threading"`` FE path (>=50000 rows),
    where nesting a numba ``prange`` inside Python threads deadlocks the threading layer
    (the documented hazard in the serial kernel's docstring). This parallel twin is dispatched
    by ``discretize_2d_quantile_batch(..., parallel=True)`` ONLY from the SERIAL-MAIN-THREAD FE
    path (``len(X) < 50000`` in ``_mrmr_fe_step``: ``check_prospective_fe_pairs`` runs with NO
    joblib threads there, so numba-parallel does not nest), mirroring the already-shipped
    column-prange in ``_quantile_edges_2d_njit``.

    BIT-IDENTICAL: each column ``j`` owns a private edge slice ``edges_inner[:, j]`` and writes
    only its own ``out[:, j]``; there is ZERO cross-column data dependence, so the result is
    independent of thread count (the identical proof the serial ``_quantile_edges_2d_njit``
    column-prange already relies on). NaN handling (``v < edge`` always False -> ``lo`` walks to
    ``n_edges`` -> rightmost bin) is per-element and unchanged.
    """
    n_rows = arr2d.shape[0]
    n_cols = arr2d.shape[1]
    n_edges = edges_inner.shape[0]
    for j in prange(n_cols):
        for r in range(n_rows):
            v = arr2d[r, j]
            lo = 0
            hi = n_edges
            while lo < hi:
                mid = (lo + hi) >> 1
                if v < edges_inner[mid, j]:
                    hi = mid
                else:
                    lo = mid + 1
            out[r, j] = lo
