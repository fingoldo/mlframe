"""Chunked column-batched ANOVA F-statistic for the SHAP-proxied prefilter's stage-A.

Sklearn ``f_classif`` / ``f_regression`` densifies the full ``(n_samples, n_features)`` design as
float64 (always: their ``_safe_X`` call materialises a contiguous float64 view), then carves K
per-class halves, then squares-and-sums. At width=20000 / n_rows=10000 that is ~1.6 GB per copy
and 5-6 GB peak RSS just to rank columns marginally -- the C4 regime OOMs at the univariate stage
before any tree booster fits.

This module recomputes the same F-statistic in column chunks of size ``batch_size`` so peak
allocation is ``8 * n_samples * batch_size`` bytes independent of total feature count. The
arithmetic is the textbook one-way ANOVA + Pearson-F closed forms, identical to sklearn modulo
float64 rounding (~1e-9 relative on dense randn). -inf sentinel flags constant / degenerate
columns, matching ``_rank_univariate``'s downstream contract.

Batch size resolves through three priorities:
  1. ``user_value`` (clamped to ``[1, n_features]``) when explicitly passed.
  2. ``KernelTuningCache.lookup("shap_proxy_prefilter_univariate_batch", ...)`` keyed by power-of-2
     feature + sample buckets, when the cache carries an entry.
  3. Auto-size: ``min(_AUTO_BATCH_CHUNK_BYTES // (8*n_samples),
                       0.1*available_RAM // (8*n_samples), n_features)``
     clamped to ``[_AUTO_BATCH_MIN, _AUTO_BATCH_MAX]``. psutil fallback drops the RAM term."""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# 256 MB / chunk caps RSS overhead from a single column-batch to ~2 copies (matmul scratch + view).
_AUTO_BATCH_CHUNK_BYTES = 256 * 1024 * 1024
# Below ~256 the per-batch Python overhead dominates the BLAS time and serialisation drowns the win.
_AUTO_BATCH_MIN = 256
# Above ~8192 a single chunk approaches the sklearn allocation that we are paying batched ranking to avoid.
_AUTO_BATCH_MAX = 8192


def _pow2_bucket(x: int) -> int:
    """Smallest power-of-2 >= x (or 1 for non-positive inputs). Used to key the kernel cache so
    nearby sizes share an entry without a per-shape calibration run."""
    if x <= 1:
        return 1
    return 1 << (int(x - 1).bit_length())


def _available_ram_bytes() -> Optional[int]:
    """Available RAM in bytes via psutil, or None if psutil is missing or the probe fails."""
    try:
        import psutil

        return int(psutil.virtual_memory().available)
    except Exception:  # ImportError or psutil runtime failure -> caller falls through.
        return None


def resolve_batch_size(
    n_features: int,
    n_samples: int,
    user_value: Optional[int] = None,
) -> int:
    """Resolve the column-batch width for chunked F-statistic computation.

    Priority: explicit ``user_value`` (clamped) > KernelTuningCache hit > auto-size from
    available RAM and ``_AUTO_BATCH_CHUNK_BYTES``. Auto value is clamped to
    ``[_AUTO_BATCH_MIN, _AUTO_BATCH_MAX]`` and never exceeds ``n_features``."""
    n_features = max(1, int(n_features))
    n_samples = max(1, int(n_samples))
    if user_value is not None:
        return int(max(1, min(int(user_value), n_features)))

    try:
        from mlframe.feature_selection.filters import get_kernel_tuning_cache

        cache = get_kernel_tuning_cache()
        if cache is not None:
            feature_bucket = _pow2_bucket(n_features)
            sample_bucket = _pow2_bucket(n_samples)
            hit = cache.lookup(
                "shap_proxy_prefilter_univariate_batch",
                n_features=feature_bucket,
                n_samples=sample_bucket,
            )
            if hit is not None and "batch_size" in hit:
                cached = int(hit["batch_size"])
                return max(1, min(cached, n_features))
    except Exception as exc:  # pragma: no cover - defensive; cache must never crash ranking.
        logger.debug("shap_proxy_prefilter_univariate batch cache lookup failed: %s", exc)

    bytes_per_row = 8 * n_samples
    chunk_cap = max(1, _AUTO_BATCH_CHUNK_BYTES // max(1, bytes_per_row))
    candidate = chunk_cap
    ram = _available_ram_bytes()
    if ram is not None:
        ram_cap = max(1, int(0.1 * ram) // max(1, bytes_per_row))
        candidate = min(candidate, ram_cap)
    candidate = min(candidate, n_features)
    candidate = max(_AUTO_BATCH_MIN, min(_AUTO_BATCH_MAX, candidate))
    candidate = min(candidate, n_features)
    return int(candidate)


def _coerce_2d_float(X) -> np.ndarray:
    """Normalise input to a contiguous float32-or-float64 ndarray view.

    Sklearn's f_classif / f_regression pass the input through ``check_X_y`` which
    keeps native float32/float64 unchanged and only coerces non-float dtypes.
    We mirror that: native float32 stays float32, float64 stays float64, anything
    else upcasts to float64. Diverging from sklearn here would silently change the
    F-statistic by ~1e-4 (single-precision SST cancellation) versus a fresh
    ``f_classif`` call on the same data — the drop-in contract callers rely on."""
    if hasattr(X, "values"):
        X = X.values
    X = np.ascontiguousarray(X)
    if X.dtype not in (np.float32, np.float64):
        X = X.astype(np.float64, copy=False)
    if X.ndim != 2:
        raise ValueError(f"expected 2-D array, got shape {X.shape}")
    return X


def f_classif_chunked(
    X,
    y,
    *,
    batch_size: Optional[int] = None,
    use_gemm: bool = True,
) -> np.ndarray:
    """Column-batched ANOVA F-statistic, sklearn ``f_classif`` parity.

    Returns a length-``n_features`` float64 vector. -inf flags constant within-class (zero
    within-group SS) and degenerate (N <= K) columns -- same sentinel ``_rank_univariate`` writes
    after the sklearn call. Allocation per batch is ``8 * n_samples * batch_size`` bytes plus K
    boolean masks of length n_samples; the original (n_samples, n_features) is never materialised
    as float64.

    ``use_gemm=True`` (default) folds the K per-class fancy-index passes into one BLAS GEMM per
    chunk via a (K, N) class-indicator matrix; the legacy K-loop is reachable with
    ``use_gemm=False`` for parity testing. The GEMM path is auto-disabled when X is float32
    because sgemm's reduction order does not bit-match sklearn's per-class
    ``safe_sqr(a).sum(axis=0)`` accumulation at single precision (~4e-4 drift), and the
    drop-in-sklearn-parity contract from iter39 requires byte-identical float32 output."""
    X = _coerce_2d_float(X)
    n_samples, n_features = X.shape
    y_arr = np.asarray(y).ravel()
    if y_arr.shape[0] != n_samples:
        raise ValueError(f"y length {y_arr.shape[0]} does not match X rows {n_samples}")

    classes, counts = np.unique(y_arr, return_counts=True)
    K = int(classes.shape[0])
    N = int(n_samples)
    # Output is always float64: F-statistic is a scalar ratio whose downstream consumers
    # (ranking, percentile cuts, caching) treat it as float64; sklearn likewise returns
    # float64 even when the X input was float32. Per-batch accumulation stays in X.dtype
    # to bit-match sklearn's safe_sqr/sum chain.
    out = np.empty(n_features, dtype=np.float64)
    if K < 2 or N <= K:
        out.fill(-np.inf)
        return out

    masks = [(y_arr == c) for c in classes]  # K boolean masks, K * N bytes, dwarfed by the chunk.
    acc_dtype = X.dtype
    n_per_class = counts.astype(acc_dtype)

    # GEMM requires float64 accumulation to preserve the sklearn-parity contract: at float32
    # sgemm reorders sums differently from sklearn's per-class safe_sqr.sum(axis=0), drifting
    # ~4e-4 vs sklearn's f_classif(X32, y) and breaking the cached-vs-fresh F-score equality
    # tested in test_f_classif_float32_input_matches_sklearn_float32.
    gemm_active = use_gemm and acc_dtype == np.float64
    # Indicator matrix (K, N) for the GEMM path; one fancy-index avoided per chunk -> K passes
    # collapse to a single BLAS dgemm call. Built once outside the chunk loop so its cost is
    # amortised across all column batches.
    indicators = None
    if gemm_active:
        indicators = np.zeros((K, N), dtype=acc_dtype)
        for k, mask in enumerate(masks):
            indicators[k, mask] = acc_dtype.type(1.0)

    bs = resolve_batch_size(n_features, n_samples, batch_size)
    bs = max(1, min(bs, n_features))

    df_between = float(K - 1)
    df_within = float(N - K)

    eps = float(np.finfo(acc_dtype).eps)
    for start in range(0, n_features, bs):
        stop = min(start + bs, n_features)
        chunk = X[:, start:stop]  # (N, b) view
        b = stop - start
        if gemm_active:
            # Single dgemm: (K,N) @ (N,b) -> (K,b). Replaces K fancy-index materialisations and
            # K column-reductions with one BLAS call; chunk_sq is the only (N,b) scratch we pay
            # for, same as the legacy path's (n_k,b) per-class block taken K times.
            sums = indicators @ chunk
            chunk_sq = chunk * chunk
            sumsq = indicators @ chunk_sq
        else:
            sums = np.empty((K, b), dtype=acc_dtype)
            sumsq = np.empty((K, b), dtype=acc_dtype)
            for k, mask in enumerate(masks):
                block = chunk[mask, :]  # (n_k, b)
                sums[k] = block.sum(axis=0)
                # Use (block * block).sum to match sklearn's safe_sqr(a).sum(axis=0) accumulation
                # order. einsum would route through a BLAS dot that reorders sums and breaks
                # bit-parity with sklearn under float32.
                sumsq[k] = (block * block).sum(axis=0)
        grand_sum = sums.sum(axis=0)  # (b,)
        total_sumsq = sumsq.sum(axis=0)  # (b,)
        correction = (grand_sum * grand_sum) / acc_dtype.type(N)
        sst = total_sumsq - correction
        ssbn = (sums * sums / n_per_class[:, None]).sum(axis=0) - correction
        sswn = sst - ssbn
        # Constant-column detection: sst is the CENTERED total sum of squares (total_sumsq -
        # correction). For a literally-constant column sst is 0 modulo float cancellation, whose
        # magnitude is bounded by eps * max(total_sumsq, correction) -- the noise floor of forming
        # the centered quantity. Gate against that centered FP floor, NOT against the raw uncentered
        # total_sumsq scaled by N: a large-mean low-variance column has a huge total_sumsq, so the
        # old eps*|total_sumsq|*N threshold ballooned far above the column's genuine centered
        # variance and silently dropped informative columns. Using the centered cancellation floor
        # keeps such columns while still catching pure-FP-drift constants.
        cancel_floor = eps * np.maximum(np.abs(total_sumsq), np.abs(correction))
        const_mask = sst <= cancel_floor
        with np.errstate(divide="ignore", invalid="ignore"):
            f = (ssbn / df_between) / (sswn / df_within)
        f64 = f.astype(np.float64, copy=False)
        f64 = np.where(np.isfinite(f64), f64, -np.inf)
        # Negative F values arise only from float cancellation on zero within-group SS;
        # clamp to -inf so they sort with constant columns (sklearn's nan path does the same).
        f64 = np.where(f64 < 0.0, -np.inf, f64)
        f64 = np.where(const_mask, -np.inf, f64)
        out[start:stop] = f64
    return out


def f_regression_chunked(
    X,
    y,
    *,
    batch_size: Optional[int] = None,
) -> np.ndarray:
    """Column-batched univariate Pearson-F, sklearn ``f_regression`` parity.

    Returns a length-``n_features`` float64 vector. -inf for constant columns (zero variance ->
    zero col_norm) and N <= 2 (zero residual df). Allocation per batch is ``8 * n_samples *
    batch_size`` bytes plus 2 length-N working vectors."""
    # Sklearn's r_regression / f_regression hard-coerce X and y to float64 via
    # check_X_y(..., dtype=np.float64). Unlike f_classif (which honours float32 input),
    # the regression path is always float64 inside sklearn, so we mirror that to match
    # bit-for-bit.
    X = _coerce_2d_float(X)
    if X.dtype != np.float64:
        X = X.astype(np.float64, copy=False)
    n_samples, n_features = X.shape
    y_arr = np.asarray(y, dtype=np.float64).ravel()
    if y_arr.shape[0] != n_samples:
        raise ValueError(f"y length {y_arr.shape[0]} does not match X rows {n_samples}")

    N = int(n_samples)
    out = np.empty(n_features, dtype=np.float64)
    if N <= 2:
        out.fill(-np.inf)
        return out

    y_mean = float(y_arr.mean())
    y_centered = y_arr - y_mean
    y_norm = float(np.sqrt(np.dot(y_centered, y_centered)))
    # Constant target test: residual norm must be meaningful relative to the magnitude scale.
    # Float drift on a uniform y leaves y_centered ~ N * eps * |y|; gate at a generous multiple
    # of that bound so a y of magnitude M doesn't survive as "non-constant" via FP cancellation.
    y_scale = max(abs(y_mean), float(np.max(np.abs(y_arr))) if y_arr.size else 1.0, 1.0)
    drift_bound = float(np.finfo(np.float64).eps) * y_scale * N
    if not np.isfinite(y_norm) or y_norm <= drift_bound:
        # Constant target -> Pearson r undefined; matches sklearn's NaN -> our -inf contract.
        out.fill(-np.inf)
        return out

    bs = resolve_batch_size(n_features, n_samples, batch_size)
    bs = max(1, min(bs, n_features))
    df_resid = float(N - 2)

    eps64 = float(np.finfo(np.float64).eps)
    for start in range(0, n_features, bs):
        stop = min(start + bs, n_features)
        chunk = X[:, start:stop]
        col_mean = chunk.mean(axis=0)
        # Centred chunk: one (N, b) float64 allocation per iteration; this is the bounded peak.
        chunk_c = chunk - col_mean
        col_norm_sq = np.einsum("ij,ij->j", chunk_c, chunk_c)
        col_norm = np.sqrt(col_norm_sq)
        # Column-magnitude proxy for the FP-drift threshold (same shape as the constant-target gate above).
        col_scale = np.maximum(np.abs(col_mean), 1.0)
        col_drift_bound = eps64 * col_scale * N
        const_col = col_norm <= col_drift_bound
        numer = chunk_c.T @ y_centered  # (b,) Pearson numerator
        with np.errstate(divide="ignore", invalid="ignore"):
            r = numer / (col_norm * y_norm)
            r2 = r * r
            # Clamp r2 into [0, 1) so the F denominator stays positive even with float drift.
            r2 = np.clip(r2, 0.0, np.nextafter(1.0, 0.0))
            f = r2 * df_resid / (1.0 - r2)
        f = np.where(np.isfinite(f), f, -np.inf)
        f = np.where(const_col, -np.inf, f)
        out[start:stop] = f
    return out
