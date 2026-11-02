"""Batched one-vs-many |Pearson corr| kernel for the cross-stage engineered-column dedup.

``_fit_impl_core.py``'s cross-stage dedup loop rank-correlates each newly-appended engineered column
against every currently-kept engineered column it might collide with (Spearman-via-ranks, per that loop's
own comment). The common case - both columns fully finite (no NaN) - lets it reuse cached full-column
ranks instead of re-sorting, but the pairwise reduction itself still called ``np.corrcoef`` ONE PAIR AT A
TIME, up to O(K^2) times over the ~200 engineered columns a wide fit can produce: each call pays Python/numpy
dispatch overhead (boolean-index temporaries, a 2x2 corrcoef matrix built and discarded for one scalar)
independent of the K^2 total.

:func:`one_vs_many_abs_corr_masked` batches this: the caller maintains an APPEND-ONLY ``(K, n)`` buffer of
every fully-finite candidate's cached rank vector seen so far (rows never move once written) plus a boolean
``active`` mask (True while that row's column is still in the live ``kept`` set); one ``prange``-parallel
launch scores the current candidate against every ACTIVE row and skips inactive ones without touching their
memory.

bench-attempt-rejected (2026-07-13, first cut): a naive per-candidate ``np.vstack`` of the CURRENT kept
rank vectors was tried first - re-copies up to O(K) rows of length n on EVERY candidate, i.e. O(K^2 * n)
total memcpy across the whole dedup pass, the SAME order as the corrcoef calls it replaces, so it measured
a NET LOSS (0.88x, 200 engineered columns / 100k rows - see ``bench_eng_dedup_batch_corr.py``). This
append-only-buffer + active-mask design reads each row's memory AT MOST ONCE (when first appended) plus
once per SURVIVING comparison it is actually compared in - no per-candidate re-copy - and is what
actually wins (see the same benchmark file for the after numbers).
"""
from __future__ import annotations

import threading
import weakref
from typing import Optional, cast
import numba
import numpy as np
from .._fe_usability_signal import _crit_np_dtype
from numba import prange


@numba.njit(cache=True, parallel=True)
def average_ranks_many(cols, out: np.ndarray) -> None:
    """``out[k]`` = average ranks (ties share the mean rank, as ``pandas.Series.rank(method="average")``) of the all-finite 1-D column ``cols[k]``; one parallel task per column."""
    for k in prange(len(cols)):
        c = cols[k]
        n = c.shape[0]
        order = np.argsort(c, kind="mergesort")
        i = 0
        while i < n:
            e = i + 1
            while e < n and c[order[e]] == c[order[i]]:
                e += 1
            r = 0.5 * (i + 1 + e)
            for t in range(i, e):
                out[k, order[t]] = r
            i = e


@numba.njit(cache=True)
def _avg_ranks_inplace(v: np.ndarray, m: int, order: np.ndarray, ranks: np.ndarray) -> None:
    """Average ranks (ties share the mean rank, as ``pandas.Series.rank(method="average")``) of the first ``m`` values of ``v`` into ``ranks``; ``order`` is scratch."""
    order[:m] = np.argsort(v[:m], kind="mergesort")
    i = 0
    while i < m:
        e = i + 1
        while e < m and v[order[e]] == v[order[i]]:
            e += 1
        r = 0.5 * (i + 1 + e)
        for t in range(i, e):
            ranks[order[t]] = r
        i = e


@numba.njit(cache=True, parallel=True)
def masked_abs_spearman_many(a: np.ndarray, fin_a: np.ndarray, others, rel_tol: float, out: np.ndarray) -> None:
    """``out[k]`` = |Spearman rho| of ``a`` and ``others[k]`` over the rows finite in both, or ``-1`` when the pair is skipped.

    A pair is skipped (as the one-pair loop skips it) with fewer than 8 jointly finite rows, a constant masked column (std at most ``rel_tol`` times its largest magnitude) or constant ranks.
    One parallel task per kept column."""
    n = a.shape[0]
    for k in prange(len(others)):
        b = others[k]
        am = np.empty(n)
        bm = np.empty(n)
        m = 0
        for i in range(n):
            if fin_a[i] and np.isfinite(b[i]):
                am[m] = a[i]
                bm[m] = b[i]
                m += 1
        out[k] = -1.0
        if m < 8:
            continue
        ma = 0.0
        mb = 0.0
        for i in range(m):
            ma += am[i]
            mb += bm[i]
        ma /= m
        mb /= m
        va = 0.0
        vb = 0.0
        xa = 0.0
        xb = 0.0
        for i in range(m):
            va += (am[i] - ma) ** 2
            vb += (bm[i] - mb) ** 2
            if abs(am[i]) > xa:
                xa = abs(am[i])
            if abs(bm[i]) > xb:
                xb = abs(bm[i])
        if (va / m) ** 0.5 <= rel_tol * xa or (vb / m) ** 0.5 <= rel_tol * xb:
            continue
        order = np.empty(m, dtype=np.int64)
        ra = np.empty(m)
        rb = np.empty(m)
        _avg_ranks_inplace(am, m, order, ra)
        _avg_ranks_inplace(bm, m, order, rb)
        mra = 0.0
        mrb = 0.0
        for i in range(m):
            mra += ra[i]
            mrb += rb[i]
        mra /= m
        mrb /= m
        sab = 0.0
        saa = 0.0
        sbb = 0.0
        for i in range(m):
            da = ra[i] - mra
            db = rb[i] - mrb
            sab += da * db
            saa += da * da
            sbb += db * db
        if (saa / m) ** 0.5 <= 1e-12 or (sbb / m) ** 0.5 <= 1e-12:
            continue
        out[k] = abs(sab / (saa * sbb) ** 0.5)


@numba.njit(cache=True)
def _avg_ranks_presorted(v: np.ndarray, order_full: np.ndarray, cm: np.ndarray, m: int, seq: np.ndarray, ranks: np.ndarray) -> None:
    """Average ranks of the ``m`` rows kept by the compact map ``cm`` (row -> compact index, -1 for a dropped row), from the full sorted row order ``order_full`` of ``v``: no sort, the dropped rows are skipped in the walk."""
    c = 0
    for t in range(order_full.shape[0]):
        r = order_full[t]
        if cm[r] >= 0:
            seq[c] = r
            c += 1
    i = 0
    while i < m:
        e = i + 1
        while e < m and v[seq[e]] == v[seq[i]]:
            e += 1
        rk = 0.5 * (i + 1 + e)
        for t in range(i, e):
            ranks[cm[seq[t]]] = rk
        i = e


@numba.njit(cache=True, parallel=True)
def masked_abs_spearman_presorted(a: np.ndarray, fin_a: np.ndarray, order_a: np.ndarray, others, orders, rel_tol: float, out: np.ndarray) -> None:
    """Version 2 of :func:`masked_abs_spearman_many`: the sorted row order of the candidate (``order_a``, finite rows only) and of every kept column (``orders[k]``) is given, so a pair costs linear passes instead of two sorts.

    The averaged ranks, the skip rules and every sum are the version-1 ones in the same order, so the results are equal."""
    n = a.shape[0]
    for k in prange(len(others)):
        b = others[k]
        cm = np.full(n, -1, dtype=np.int64)
        am = np.empty(n)
        bm = np.empty(n)
        m = 0
        for i in range(n):
            if fin_a[i] and np.isfinite(b[i]):
                cm[i] = m
                am[m] = a[i]
                bm[m] = b[i]
                m += 1
        out[k] = -1.0
        if m < 8:
            continue
        ma = 0.0
        mb = 0.0
        for i in range(m):
            ma += am[i]
            mb += bm[i]
        ma /= m
        mb /= m
        va = 0.0
        vb = 0.0
        xa = 0.0
        xb = 0.0
        for i in range(m):
            va += (am[i] - ma) ** 2
            vb += (bm[i] - mb) ** 2
            if abs(am[i]) > xa:
                xa = abs(am[i])
            if abs(bm[i]) > xb:
                xb = abs(bm[i])
        if (va / m) ** 0.5 <= rel_tol * xa or (vb / m) ** 0.5 <= rel_tol * xb:
            continue
        seq = np.empty(m, dtype=np.int64)
        ra = np.empty(m)
        rb = np.empty(m)
        _avg_ranks_presorted(a, order_a, cm, m, seq, ra)
        _avg_ranks_presorted(b, orders[k], cm, m, seq, rb)
        mra = 0.0
        mrb = 0.0
        for i in range(m):
            mra += ra[i]
            mrb += rb[i]
        mra /= m
        mrb /= m
        sab = 0.0
        saa = 0.0
        sbb = 0.0
        for i in range(m):
            da = ra[i] - mra
            db = rb[i] - mrb
            sab += da * db
            saa += da * da
            sbb += db * db
        if (saa / m) ** 0.5 <= 1e-12 or (sbb / m) ** 0.5 <= 1e-12:
            continue
        out[k] = abs(sab / (saa * sbb) ** 0.5)


_ORDER_CACHE_LOCK = threading.Lock()
_ORDER_CACHE: "dict[int, tuple]" = {}  # id(column) -> (weak reference to the column, its finite rows in ascending value order); a dead reference never matches, so a reused id cannot hitry lives
_ORDER_CACHE_MAX = 96


def finite_sort_order(col: np.ndarray, *, cache: bool = True) -> np.ndarray:
    """The row ids of the finite values of ``col`` in ascending value order (stable), int64; cached per column object for the kept columns, which live for a whole fit."""
    if cache:
        with _ORDER_CACHE_LOCK:
            hit = _ORDER_CACHE.get(id(col))
        if hit is not None and hit[0]() is col:
            return cast("np.ndarray", hit[1])
    rows = np.flatnonzero(np.isfinite(col))
    order = rows[np.argsort(col[rows], kind="mergesort")]
    if cache:
        with _ORDER_CACHE_LOCK:
            for dead in [k for k, v in _ORDER_CACHE.items() if v[0]() is None]:
                _ORDER_CACHE.pop(dead, None)
            while len(_ORDER_CACHE) >= _ORDER_CACHE_MAX:
                _ORDER_CACHE.pop(next(iter(_ORDER_CACHE)), None)
            _ORDER_CACHE[id(col)] = (weakref.ref(col), order)
    return order


def clear_finite_sort_order_cache() -> None:
    """Drop every cached sort order; called when a fit ends so the orders of its columns do not stay in memory."""
    with _ORDER_CACHE_LOCK:
        _ORDER_CACHE.clear()


@numba.njit(cache=True, parallel=True, fastmath=False)
def _row_mean_and_centred_ss_njit(row: np.ndarray):
    """``(mean, sum((row - mean)^2))`` for one buffer row, in the same two passes the comparison kernel used to do inline."""
    n = row.shape[0]
    s = 0.0
    for i in range(n):
        s += row[i]
    m = s / n
    ss = 0.0
    for i in range(n):
        d = row[i] - m
        ss += d * d
    return m, ss


@numba.njit(cache=True, parallel=True, fastmath=False)
def _one_vs_many_pearson_abs_masked_njit(a_centred, saa, mat, active, row_mean, row_ss) -> np.ndarray:
    """|Pearson corr| of a candidate against every ACTIVE row of ``mat`` (k, n), computed in parallel over k.

    ``active`` (k,) boolean: row ``j`` is skipped entirely (output stays 0.0, its memory never read) when
    ``active[j]`` is False - lets the caller pass its FULL append-only history buffer as a zero-copy view
    and select the live subset via the mask instead of gathering/copying a fresh sub-matrix per call.

    The candidate arrives already centred, with its own centred sum-of-squares, and each row's mean and centred sum-of-squares arrive in
    ``row_mean`` / ``row_ss``. Those are constants of a row in an append-only buffer, and recomputing them here meant two extra reduction
    passes over ``mat[j]`` for every candidate the row was ever compared against, O(K^2 * n) over the scan. Only the cross term is left to
    compute, so each surviving comparison now reads the row once.

    Still a mean-then-centre formulation - mirrors ``batch_pair_usability_corr_gpu.py``'s ``_abs_pearson_form_reduction`` rationale for the
    same choice: a one-pass sum-of-squares formula risks catastrophic cancellation on a near-constant row, spuriously reading as correlated
    instead of ~0. Assumes both sides are already finite (the caller only routes fully-finite candidate/kept rank pairs here); a
    near-constant row is still floored to 0.0 via the same relative-variance guard the codebase's other abs-corr kernels use.
    """
    n = a_centred.shape[0]
    k = mat.shape[0]
    # Intentional float64: accumulator kept f64 (statistic / count sums need full precision)
    out = np.zeros(k, dtype=np.float64)
    if saa <= 1e-24 * n:
        return out
    for j in prange(k):
        if not active[j]:
            continue
        sbb = row_ss[j]
        if sbb <= 1e-24 * n:
            continue
        mb = row_mean[j]
        sab = 0.0
        for i in range(n):
            sab += a_centred[i] * (mat[j, i] - mb)
        denom = (saa * sbb) ** 0.5
        r = sab / denom
        out[j] = -r if r < 0.0 else r
    return out


def row_mean_and_centred_ss(row: np.ndarray) -> tuple[float, float]:
    """``(mean, centred sum-of-squares)`` of one buffer row, for the caller to cache when it appends the row."""
    return cast(tuple[float, float], _row_mean_and_centred_ss_njit(np.asarray(row, dtype=_crit_np_dtype())))


def one_vs_many_abs_corr_masked(a: np.ndarray, buf: np.ndarray, active: np.ndarray, row_mean: Optional[np.ndarray] = None, row_ss: Optional[np.ndarray] = None) -> np.ndarray:
    """|Pearson corr| of ``a`` (n,) against every ACTIVE row of ``buf`` (k, n), batched.

    Thin dtype-normalising wrapper around :func:`_one_vs_many_pearson_abs_masked_njit` so the call site doesn't have to know about the njit
    kernel's raw-matrix calling convention. Returns an empty ``(0,)`` array when ``buf`` has zero rows (nothing to compare against, kernel not
    invoked). ``row_mean`` / ``row_ss`` are the cached per-row moments; a caller that has not kept them gets them computed here, which is the
    old behaviour and costs the two extra passes per row this exists to avoid.
    """
    if buf.shape[0] == 0:
        return np.empty(0, dtype=_crit_np_dtype())
    arr = np.asarray(a, dtype=_crit_np_dtype())
    mat = np.asarray(buf, dtype=_crit_np_dtype())
    mean_a, saa = _row_mean_and_centred_ss_njit(arr)
    if row_mean is None or row_ss is None:
        # Intentional float64: accumulator kept f64 (statistic / count sums need full precision)
        row_mean = np.empty(mat.shape[0], dtype=np.float64)
        # Intentional float64: accumulator kept f64 (statistic / count sums need full precision)
        row_ss = np.empty(mat.shape[0], dtype=np.float64)
        for j in range(mat.shape[0]):
            row_mean[j], row_ss[j] = _row_mean_and_centred_ss_njit(mat[j])
    return np.asarray(
        _one_vs_many_pearson_abs_masked_njit(
            arr - mean_a,
            saa,
            mat,
            np.asarray(active, dtype=np.bool_),
            # Intentional float64: accumulator kept f64 (statistic / count sums need full precision)
            np.asarray(row_mean, dtype=np.float64),
            # Intentional float64: accumulator kept f64 (statistic / count sums need full precision)
            np.asarray(row_ss, dtype=np.float64),
        )
    )


__all__ = ["one_vs_many_abs_corr_masked", "row_mean_and_centred_ss"]
