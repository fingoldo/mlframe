"""Source-side collinear-column dedup for the orthogonal-univariate FE stage.

Carved out of ``_orthogonal_univariate_fe/__init__.py`` (2026-06-22, monolith-split: the package facade
re-exports ``_dedup_collinear_source_cols`` from its bottom). Self-contained -- depends only on numpy/pandas.
"""
from __future__ import annotations

import contextlib
import logging
import threading
from typing import Optional, Sequence

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


def _dedup_sample_idx(n_rows: int) -> Optional[np.ndarray]:
    """Deterministic row-sample index for the correlation pass (see ``_MAX_CORR_ROWS``), factored out so the
    fit-scoped memo's cache-key hash (:func:`_dedup_cache_key`) samples the EXACT same rows the correlation
    pass itself would -- same seed, same ``n_rows`` -> same draw both times."""
    if n_rows > _MAX_CORR_ROWS:
        return np.sort(np.random.default_rng(0).choice(n_rows, size=_MAX_CORR_ROWS, replace=False))
    return None


# Fit-scoped memo for _dedup_collinear_source_cols: mirrors heavy_tail_memo_scope() in
# hermite_fe._hermite_robust. Five independent opt-in FE-family builders (orth / extra-basis / gpu-resident /
# wavelet / hinge) each call this dedup on their own candidate columns; a "kitchen sink" config with several
# families on would otherwise repeat the identical O(P^2) corrcoef pass 2-5x within ONE MRMR.fit() call. The
# memo is OFF by default (cache is None) so every call outside an explicit dedup_collinear_memo_scope() is
# byte-for-byte unchanged. threading.local -> no cross-worker contamination.
_DEDUP_MEMO = threading.local()


@contextlib.contextmanager
def dedup_collinear_memo_scope():
    """Enable the fit-scoped ``_dedup_collinear_source_cols`` memo for one MRMR.fit() call, then clear it.
    Nesting-safe: an inner scope reuses the outer cache and the outer owner clears it."""
    if getattr(_DEDUP_MEMO, "cache", None) is not None:
        yield  # reuse outer scope; outer owner clears
        return
    _DEDUP_MEMO.cache = {}
    try:
        yield
    finally:
        _DEDUP_MEMO.cache = None


def _dedup_cache_key(X: pd.DataFrame, cols: Sequence[str], corr_threshold: float) -> tuple:
    """Content-hash key for the fit-scoped dedup memo: per-column ``(name, shape, content hash)`` over the
    SAME row-capped sample the correlation pass itself uses (:func:`_dedup_sample_idx`), so two calls that
    would run the identical corrcoef pass collapse to one cache entry regardless of which DataFrame OBJECT
    each FE-family caller built its view from. Non-numeric / missing columns always pass through
    unconditionally (Pass 1 below never reads their values), so only their identity needs to be in the key."""
    from .._fe_resident_operands import _content_hash

    n_rows = len(X)
    sample_idx = _dedup_sample_idx(n_rows)
    parts: list[tuple] = []
    for c in cols:
        if c not in X.columns or not pd.api.types.is_numeric_dtype(X[c]):
            parts.append((c, "non_numeric"))
            continue
        arr = np.asarray(X[c].to_numpy(), dtype=np.float64)
        if sample_idx is not None:
            arr = arr[sample_idx]
        arr = np.ascontiguousarray(arr)
        parts.append((c, arr.shape, _content_hash(arr)))
    return (tuple(parts), float(corr_threshold))


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
    return np.asarray(corr)


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

    # Q/R (the partial-NaN candidate block and its dense/partial comparison block) can recur with IDENTICAL
    # content across the <=6 _dedup_collinear_source_cols calls/fit (independent FE families -- orth/
    # extra-basis/gpu-resident/wavelet/hinge -- each deduping their own candidate columns against the SAME
    # underlying source frame): resident-cache the upload so a repeat-content call hits instead of re-
    # uploading. When Q/R genuinely differ per call (the common case, since each family's candidate set
    # differs), this is a plain cache miss -> a fresh upload, identical cost to the old raw cp.asarray -- so
    # the fix is correctness-neutral either way, only a possible win when content does recur. No ``dtype=``
    # override: Q/R keep their CALLER dtype (f32 under MLFRAME_CRIT_DTYPE_RELAXED), matching the original
    # ``cp.asarray(Q)``'s native-dtype passthrough exactly (unlike the njit backend, which explicitly
    # upcasts to f64) -- forcing float64 here would change the matmul precision, not just cache it.
    from .._fe_resident_operands import resident_operand
    Qd = resident_operand(Q, "orth_dedup_Q")
    Rd = resident_operand(R, "orth_dedup_R")
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
    return np.asarray(cp.asnumpy(corr))


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
        return np.asarray(_pc_corr_njit(np.ascontiguousarray(Q, dtype=np.float64), np.ascontiguousarray(R, dtype=np.float64)))
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
    # STRICT GPU mode: prefer the cupy backend (solo-benched 2.0-2.7x over numpy in the moderate band;
    # 12 numpy calls cost 335s on the wellbore-100k GPU-strict cProfile). The numpy default exists because
    # cupy's solo win erodes under joblib-worker GPU contention -- but STRICT mode's explicit contract is
    # "carry the FE compute on the device", and the per-call work floor inside fe_gpu_strict_enabled already
    # keeps trivially small dedups (p < 64 or n*p < 1M) on the CPU. Non-strict hosts keep the measured
    # default via the KTC lookup below, unchanged.
    try:
        from .._fe_gpu_strict import fe_gpu_strict_enabled

        if fe_gpu_strict_enabled(n=n, p=max(q, r)):
            return "cupy"
    except Exception:  # nosec B110 -- strict-gate probe failure must never break the dedup itself
        pass
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

    2026-07-12 FIT-SCOPED MEMO: wrap the call site in ``dedup_collinear_memo_scope()`` to memoize repeated
    calls with the SAME (cols content, corr_threshold) within one MRMR.fit() call -- five independent opt-in
    FE-family builders (orth / extra-basis / gpu-resident / wavelet / hinge) each call this dedup on their own
    candidate columns, and a "kitchen sink" config with several families on would otherwise repeat the
    identical O(P^2) corrcoef pass 2-5x. The memo is OFF by default (cache is None) so every call outside an
    explicit scope is byte-for-byte unchanged.
    """
    if not cols:
        return list(cols)
    _memo = getattr(_DEDUP_MEMO, "cache", None)
    _memo_key = None
    if _memo is not None:
        _memo_key = _dedup_cache_key(X, cols, corr_threshold)
        _hit = _memo.get(_memo_key)
        if _hit is not None:
            return list(_hit)
    # ---- Pass 1: classify columns -------------------------------------------
    # Pre-classify each column so the bulk corrcoef in pass 2 only sees fully-
    # finite varying columns. Order-preservation matters because the kept
    # list mirrors the input order; we record per-col disposition first then
    # walk in order again in pass 3.
    n_rows = len(X)
    # Row-cap the correlation estimate (see _MAX_CORR_ROWS note). Deterministic seed so a fit is reproducible; sorted so
    # the sampled rows keep their original order (irrelevant to Pearson, but avoids surprising downstream if reused).
    _sample_idx = _dedup_sample_idx(n_rows)
    classes: list[str] = []  # one of: "pass_through", "dense", "partial_nan"
    dense_idx: list[int] = []  # candidate index in cols -> dense-matrix row
    dense_rows: list[np.ndarray] = []  # the dense arrays themselves
    partial_arrays: dict[int, np.ndarray] = {}  # candidate index -> arr (with NaN)

    from .._fe_usability_signal import _crit_np_dtype
    _dt = _crit_np_dtype()  # f32 under MLFRAME_CRIT_DTYPE_RELAXED (default); coarse corr-threshold dedup is scale-robust. Hoisted out of the loop (was re-invoked once per column).
    for i, c in enumerate(cols):
        if c not in X.columns or not pd.api.types.is_numeric_dtype(X[c]):
            classes.append("pass_through")
            continue
        arr = np.asarray(X[c].to_numpy(), dtype=_dt)
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
    pc_pd: np.ndarray | None
    pc_pp: np.ndarray | None
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
            assert abs_corr is not None  # this "dense" cls only occurs when dense_rows was non-empty, which set abs_corr
            is_dup = _hits(abs_corr[row_idx], kept_dense_rows)
            # Also compare against any kept partial-NaN columns (rare): pc_pd[p, row_idx] is partial-p vs this dense col.
            if not is_dup and kept_partial_pos:
                assert pc_pd is not None  # kept_partial_pos non-empty implies partial_order was non-empty, which set pc_pd
                col = pc_pd[:, row_idx]
                is_dup = bool(np.any(np.isfinite(col[kept_partial_pos]) & (col[kept_partial_pos] >= corr_threshold)))
            if not is_dup:
                kept.append(c)
                kept_dense_rows.append(row_idx)
            continue
        # cls == "partial_nan": vectorized lookups against kept dense + kept partial.
        p = partial_pos[i]
        assert pc_pd is not None and pc_pp is not None  # this "partial_nan" cls only occurs when partial_order was non-empty, which set both
        is_dup = _hits(pc_pd[p], kept_dense_rows) or _hits(pc_pp[p], kept_partial_pos)
        if not is_dup:
            kept.append(c)
            kept_partial_pos.append(p)
    if _memo is not None:
        _memo[_memo_key] = kept
    return kept
