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

import numba
import numpy as np
from numba import prange


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


def row_mean_and_centred_ss(row: np.ndarray):
    """``(mean, centred sum-of-squares)`` of one buffer row, for the caller to cache when it appends the row."""
    return _row_mean_and_centred_ss_njit(np.asarray(row, dtype=np.float64))


def one_vs_many_abs_corr_masked(a: np.ndarray, buf: np.ndarray, active: np.ndarray, row_mean=None, row_ss=None) -> np.ndarray:
    """|Pearson corr| of ``a`` (n,) against every ACTIVE row of ``buf`` (k, n), batched.

    Thin dtype-normalising wrapper around :func:`_one_vs_many_pearson_abs_masked_njit` so the call site doesn't have to know about the njit
    kernel's raw-matrix calling convention. Returns an empty ``(0,)`` array when ``buf`` has zero rows (nothing to compare against, kernel not
    invoked). ``row_mean`` / ``row_ss`` are the cached per-row moments; a caller that has not kept them gets them computed here, which is the
    old behaviour and costs the two extra passes per row this exists to avoid.
    """
    if buf.shape[0] == 0:
        return np.empty(0, dtype=np.float64)
    arr = np.asarray(a, dtype=np.float64)
    mat = np.asarray(buf, dtype=np.float64)
    mean_a, saa = _row_mean_and_centred_ss_njit(arr)
    if row_mean is None or row_ss is None:
        row_mean = np.empty(mat.shape[0], dtype=np.float64)
        row_ss = np.empty(mat.shape[0], dtype=np.float64)
        for j in range(mat.shape[0]):
            row_mean[j], row_ss[j] = _row_mean_and_centred_ss_njit(mat[j])
    return np.asarray(
        _one_vs_many_pearson_abs_masked_njit(
            arr - mean_a,
            saa,
            mat,
            np.asarray(active, dtype=np.bool_),
            np.asarray(row_mean, dtype=np.float64),
            np.asarray(row_ss, dtype=np.float64),
        )
    )


__all__ = ["one_vs_many_abs_corr_masked", "row_mean_and_centred_ss"]
