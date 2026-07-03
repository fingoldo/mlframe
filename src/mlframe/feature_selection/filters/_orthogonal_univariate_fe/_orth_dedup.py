"""Source-side collinear-column dedup for the orthogonal-univariate FE stage.

Carved out of ``_orthogonal_univariate_fe/__init__.py`` (2026-06-22, monolith-split: the package facade
re-exports ``_dedup_collinear_source_cols`` from its bottom). Self-contained -- depends only on numpy/pandas.
"""
from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
import pandas as pd
from numba import njit, prange

logger = logging.getLogger(__name__)

# The dedup is a DECISION (drop near-duplicate source columns), not a final statistic, so the correlations do not need
# every row: a Pearson |r| estimate from a large row sample has standard error ~ (1-r^2)/sqrt(m), far under the slack of
# a 0.999 threshold at m=100k. Capping the rows keeps the pass O(P^2 * cap) instead of O(P^2 * N) and -- critical on
# mlframe's 100+ GB frames -- bounds the transient (P, cap) work matrices regardless of N. Only engages when N > cap, so
# every small-n path (all tests) is byte-for-byte unchanged. Override via MLFRAME_FE_DEDUP_MAX_CORR_ROWS.
import os as _os

_MAX_CORR_ROWS = int(_os.environ.get("MLFRAME_FE_DEDUP_MAX_CORR_ROWS", "100000") or "100000")


def _pc_corr_numpy(Q: np.ndarray, R: np.ndarray) -> np.ndarray:
    """numpy(BLAS) backend: 6 masked matmuls. Multithreaded via the platform BLAS; the CPU default (see the dispatcher)."""
    Qm = np.isfinite(Q)
    Rm = np.isfinite(R)
    Q0 = np.where(Qm, Q, 0.0)
    R0 = np.where(Rm, R, 0.0)
    Qmf = Qm.astype(np.float64)
    Rmf = Rm.astype(np.float64)
    n = Qmf @ Rmf.T
    Sx = Q0 @ Rmf.T
    Sy = Qmf @ R0.T
    Sxx = (Q0 * Q0) @ Rmf.T
    Syy = Qmf @ (R0 * R0).T
    Sxy = Q0 @ R0.T
    with np.errstate(divide="ignore", invalid="ignore"):
        cov = Sxy - Sx * Sy / n
        varx = Sxx - Sx * Sx / n
        vary = Syy - Sy * Sy / n
        corr = np.abs(cov / np.sqrt(varx * vary))
    # nan -> 'not a duplicate' for the caller. Mirror the legacy per-pair skips: <8 common rows, or either side
    # near-constant on the common rows (std <= 1e-12 => var <= 1e-24) -- otherwise a 0/~0 ratio could read as inf >= thr.
    corr[(n < 8) | (varx <= 1e-24) | (vary <= 1e-24)] = np.nan
    return corr


@njit(parallel=True, cache=True)  # NO fastmath: the kernel relies on isfinite() for the NaN mask; nnan/ninf would elide it.
def _pc_corr_njit(Q: np.ndarray, R: np.ndarray) -> np.ndarray:
    """njit(parallel) backend: one fused pass per pair, no temporaries. MEASURED SLOWER than numpy (0.23-0.80x) -- a
    naive triple-loop cannot beat BLAS -- so it is never auto-selected; kept as a force-selectable option per
    REJECTED != DELETED. See ``_benchmarks/bench_pairwise_complete_corr.py``."""
    q = Q.shape[0]
    r = R.shape[0]
    n = Q.shape[1]
    out = np.empty((q, r), dtype=np.float64)
    for i in prange(q):
        for j in range(r):
            cnt = 0
            sx = 0.0
            sy = 0.0
            sxx = 0.0
            syy = 0.0
            sxy = 0.0
            for k in range(n):
                a = Q[i, k]
                b = R[j, k]
                if np.isfinite(a) and np.isfinite(b):
                    cnt += 1
                    sx += a
                    sy += b
                    sxx += a * a
                    syy += b * b
                    sxy += a * b
            if cnt < 8:
                out[i, j] = np.nan
            else:
                inv = 1.0 / cnt
                cov = sxy - sx * sy * inv
                vx = sxx - sx * sx * inv
                vy = syy - sy * sy * inv
                out[i, j] = np.nan if (vx <= 1e-24 or vy <= 1e-24) else abs(cov / np.sqrt(vx * vy))
    return out


def _pc_corr_cupy(Q: np.ndarray, R: np.ndarray) -> np.ndarray:
    """cupy(CUDA) backend: the same 6 masked matmuls on the GPU. Solo-benched 2.0-2.7x over numpy in a moderate band but
    LOSES at large P*n (GPU memory pressure) -- and, like the plug-in MI dispatcher, a solo win is eroded under joblib-
    worker GPU contention, so it is not the auto-default. OOM / no-cupy -> the dispatcher falls back to numpy."""
    import cupy as cp

    Qd = cp.asarray(Q)
    Rd = cp.asarray(R)
    Qm = cp.isfinite(Qd)
    Rm = cp.isfinite(Rd)
    Q0 = cp.where(Qm, Qd, 0.0)
    R0 = cp.where(Rm, Rd, 0.0)
    Qmf = Qm.astype(cp.float64)
    Rmf = Rm.astype(cp.float64)
    n = Qmf @ Rmf.T
    Sx = Q0 @ Rmf.T
    Sy = Qmf @ R0.T
    Sxx = (Q0 * Q0) @ Rmf.T
    Syy = Qmf @ (R0 * R0).T
    Sxy = Q0 @ R0.T
    cov = Sxy - Sx * Sy / n
    vx = Sxx - Sx * Sx / n
    vy = Syy - Sy * Sy / n
    corr = cp.abs(cov / cp.sqrt(vx * vy))
    corr[(n < 8) | (vx <= 1e-24) | (vy <= 1e-24)] = cp.nan
    return cp.asnumpy(corr)


def _pairwise_complete_abs_corr(Q: np.ndarray, R: np.ndarray) -> np.ndarray:
    """``|Pearson r|`` between every row of ``Q`` and every row of ``R`` over PAIRWISE-COMPLETE (both-finite) rows,
    vectorized instead of a Python per-pair ``np.corrcoef`` loop. ``Q``/``R``: ``(q, n)`` / ``(r, n)`` float64 with
    ``NaN`` for missing. Returns ``(q, r)``; entries with <8 common obs or zero variance -> ``nan`` ('not a duplicate').

    Backend dispatch (all three bit-identical to ~1e-16): numpy(BLAS) is the CPU default; ``cupy`` and ``njit`` are
    force-selectable via ``MLFRAME_FE_DEDUP_CORR_BACKEND=cupy|njit|numpy`` and consulted per-host via the kernel tuning
    cache. numpy is the default because (a) njit measured slower everywhere and (b) cupy's solo win is eroded under the
    FE pipeline's joblib-worker GPU contention (same lesson as ``lookup_mi_classif_backend``). cupy OOM -> numpy."""
    backend = _resolve_pc_backend(Q.shape[0], R.shape[0], Q.shape[1])
    if backend == "njit":
        return _pc_corr_njit(np.ascontiguousarray(Q, dtype=np.float64), np.ascontiguousarray(R, dtype=np.float64))
    if backend == "cupy":
        try:
            return _pc_corr_cupy(Q, R)
        except Exception as exc:  # cupy missing / OOM / driver error -> safe CPU fallback
            logger.warning("[FE dedup] cupy pairwise-corr backend failed (%s); falling back to numpy.", type(exc).__name__)
    return _pc_corr_numpy(Q, R)


def _resolve_pc_backend(q: int, r: int, n: int) -> str:
    """Pick 'numpy' | 'njit' | 'cupy'. Env force wins; else the per-host kernel-tuning cache (fallback 'numpy')."""
    forced = _os.environ.get("MLFRAME_FE_DEDUP_CORR_BACKEND", "").strip().lower()
    if forced in ("numpy", "njit"):
        return forced
    if forced in ("cupy", "cuda"):
        return "cupy"
    try:
        from mlframe.feature_selection._benchmarks.kernel_tuning_cache.dispatch import lookup_pairwise_corr_backend

        return lookup_pairwise_corr_backend(max(q, r), n)
    except Exception:
        return "numpy"


def _dedup_collinear_source_cols(
    X: pd.DataFrame, cols: Sequence[str], *, corr_threshold: float = 0.999,
) -> list[str]:
    """Drop near-duplicate source columns BEFORE basis enumeration.

    Layer 27 incident (2026-05-31): on 10 collinear sources (x2..x10 = x1 +
    1% jitter), the constructor emitted 10 He_2 columns and every one
    survived MRMR's redundancy gate because their CMI-residuals under
    quantile binning differed by tiny amounts above the relevance floor.
    Hybrid stage exploded the candidate set 10x and MRMR couldn't
    distinguish the duplicates.

    Fix: a cheap source-side dedup pass. Walks cols in order, computes the
    abs Pearson correlation against every column already kept; drops the
    candidate if it correlates above ``corr_threshold`` with anything in
    the kept set. ``0.999`` matches the 1% jitter test fixture while
    leaving real-world near-duplicates (corr in [0.95, 0.99]) untouched.

    Non-numeric / constant / all-NaN columns are passed through (not
    deduped, not dropped) so downstream basis evaluation handles them as
    before.

    Layer 30 perf (2026-05-31): the original implementation called
    ``np.corrcoef`` per (candidate, kept) pair which is O(p^2) numpy calls
    plus Python overhead. At p=200 cProfile attributed 5.0s out of 4.8s
    wall (cumulative) to this dedup pass — the dominant hotspot. The new
    implementation:

    1. Pre-classifies columns into pass-through (non-numeric / all-NaN /
       constant), partial-NaN (rare path), and fully-finite-and-varying.
    2. Stacks all fully-finite columns into one (p_dense, n) matrix and
       calls ``np.corrcoef`` once on the bulk matrix — one C call instead
       of O(p^2). Numerically bit-identical to the per-pair recipe (same
       reduction order in numpy's cov / std).
    3. For each candidate, looks up its row of the precomputed matrix
       against indices of kept dense columns: O(K) per candidate, O(p*K)
       total, no Python-side reductions.
    4. Partial-NaN columns fall back to the original masked-corr path
       (still per-pair, but the count of these is typically 0 — production
       hybrid path uses np.nanmean fill before reaching this dedup).

    Bench at p=200 n=2000 all-finite synthetic frame: 5.0s -> ~0.05s
    (100x).
    """
    if not cols:
        return list(cols)
    # ---- Pass 1: classify columns -------------------------------------------
    # Pre-classify each column so the bulk corrcoef in pass 2 only sees fully-
    # finite varying columns. Order-preservation matters because the kept
    # list mirrors the input order; we record per-col disposition first then
    # walk in order again in pass 3.
    n_rows = len(X)
    # Row-cap the correlation estimate (see _MAX_CORR_ROWS note). Deterministic seed so a fit is reproducible; sorted so
    # the sampled rows keep their original order (irrelevant to Pearson, but avoids surprising downstream if reused).
    _sample_idx = None
    if n_rows > _MAX_CORR_ROWS:
        _sample_idx = np.sort(np.random.default_rng(0).choice(n_rows, size=_MAX_CORR_ROWS, replace=False))
    classes: list[str] = []  # one of: "pass_through", "dense", "partial_nan"
    dense_idx: list[int] = []  # candidate index in cols -> dense-matrix row
    dense_rows: list[np.ndarray] = []  # the dense arrays themselves
    partial_arrays: dict[int, np.ndarray] = {}  # candidate index -> arr (with NaN)

    for i, c in enumerate(cols):
        if c not in X.columns or not pd.api.types.is_numeric_dtype(X[c]):
            classes.append("pass_through")
            continue
        arr = np.asarray(X[c].to_numpy(), dtype=np.float64)
        if _sample_idx is not None:
            arr = arr[_sample_idx]
        finite = np.isfinite(arr)
        if not finite.any():
            # All-NaN: pass-through, no kept_array stored. (Matches legacy:
            # legacy stored kept_arrays[c] = arr but immediately continued
            # on the next iteration's `mask.sum() < 8` check; the only
            # observable effect is that downstream partial-NaN candidates
            # don't compute corr against an all-NaN kept column anyway.)
            classes.append("pass_through")
            continue
        # std on the finite subset, matching legacy's constant-detection.
        if arr[finite].std() <= 1e-12:
            classes.append("pass_through")
            continue
        if finite.all():
            classes.append("dense")
            dense_idx.append(len(dense_rows))
            dense_rows.append(arr)
        else:
            classes.append("partial_nan")
            partial_arrays[i] = arr

    # ---- Pass 2: bulk corrcoef on the dense block ---------------------------
    # One C call replaces p_dense * (p_dense - 1) / 2 per-pair Python+numpy
    # roundtrips. Numerically equivalent to per-pair np.corrcoef because
    # numpy.corrcoef(M)[i, j] uses the same _cov / _std reduction order as
    # numpy.corrcoef(M[i], M[j])[0, 1] (verified bit-identical at p=200).
    if dense_rows:
        dense_matrix = np.vstack(dense_rows)
        # Empty (0, n) matrix corrcoef raises; only call when we have rows.
        corr_matrix = np.corrcoef(dense_matrix)
        # Single-row corrcoef returns a scalar 1.0 instead of (1, 1); normalize.
        if corr_matrix.ndim == 0:
            corr_matrix = np.array([[1.0]], dtype=np.float64)
        # Absolute corrs only; NaN -> not duplicate (matches legacy's
        # `if not np.isfinite(corr): continue` skip).
        abs_corr = np.abs(corr_matrix)
    else:
        abs_corr = None

    # Precompute the partial-NaN comparison blocks ONCE via the vectorized pairwise-complete kernel, replacing the
    # legacy O(P^2) per-pair ``np.corrcoef`` loop (the hot path on frames with many NaN-bearing columns, e.g. well logs
    # with missing measurements -- the case the legacy "typically 0 partial columns" assumption failed on). ``partial_mat``
    # rows are indexed by ``partial_pos`` in candidate order; ``pc_pd`` = |corr| vs each dense row, ``pc_pp`` = vs each
    # other partial row. Empty when there are no partial columns (the common all-dense case is untouched -> bit-identical).
    partial_order = [i for i in range(len(cols)) if classes[i] == "partial_nan"]
    partial_pos = {i: p for p, i in enumerate(partial_order)}
    if partial_order:
        partial_mat = np.vstack([partial_arrays[i] for i in partial_order])
        dense_mat_for_pc = dense_matrix if dense_rows else np.empty((0, partial_mat.shape[1]), dtype=np.float64)
        pc_pd = _pairwise_complete_abs_corr(partial_mat, dense_mat_for_pc)  # (n_partial, n_dense)
        pc_pp = _pairwise_complete_abs_corr(partial_mat, partial_mat)  # (n_partial, n_partial)
    else:
        pc_pd = pc_pp = None

    def _hits(vec: np.ndarray, idxs: list[int]) -> bool:
        """True iff any kept reference in ``idxs`` has finite |corr| >= threshold (nan == not-a-duplicate)."""
        if not idxs:
            return False
        sub = vec[idxs]
        return bool(np.any(np.isfinite(sub) & (sub >= corr_threshold)))

    # ---- Pass 3: walk in order, apply dedup verdict --------------------------
    kept: list[str] = []
    kept_dense_rows: list[int] = []  # dense-matrix row indices already kept
    kept_partial_pos: list[int] = []  # partial-matrix row indices already kept
    dense_pos = 0  # which dense candidate slot we're currently at
    for i, c in enumerate(cols):
        cls = classes[i]
        if cls == "pass_through":
            kept.append(c)
            # Pass-through columns are never used as a corr reference for downstream candidates (legacy stored arr but
            # the comparison always short-circuited via `.sum() < 8` or `.std() <= 1e-12`).
            continue
        if cls == "dense":
            row_idx = dense_idx[dense_pos]
            dense_pos += 1
            is_dup = _hits(abs_corr[row_idx], kept_dense_rows)
            # Also compare against any kept partial-NaN columns (rare): pc_pd[p, row_idx] is partial-p vs this dense col.
            if not is_dup and kept_partial_pos:
                col = pc_pd[:, row_idx]
                is_dup = bool(np.any(np.isfinite(col[kept_partial_pos]) & (col[kept_partial_pos] >= corr_threshold)))
            if not is_dup:
                kept.append(c)
                kept_dense_rows.append(row_idx)
            continue
        # cls == "partial_nan": vectorized lookups against kept dense + kept partial.
        p = partial_pos[i]
        is_dup = _hits(pc_pd[p], kept_dense_rows) or _hits(pc_pp[p], kept_partial_pos)
        if not is_dup:
            kept.append(c)
            kept_partial_pos.append(p)
    return kept
