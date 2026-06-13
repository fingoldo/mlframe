"""Screening helpers used by CompositeTargetDiscovery: column extraction (_extract_column_array / _is_numeric_column), correlation guards (_safe_corr / _safe_abs_corr_all / _residualise), mutual-information scoring (_mi_pair_bin / _mi_to_target), tiny-model CV-RMSE rerank (_build_tiny_model / _tiny_cv_rmse_raw_y / _tiny_cv_rmse_y_scale / *_multiseed variants), stratified sampling (_sample_indices). composite.py re-exports every symbol below for full back-compat."""


from __future__ import annotations

import contextlib
import logging
import math
import os
import sys
import warnings
from timeit import default_timer as timer
from typing import (
    TYPE_CHECKING, Any, Callable, Dict, List, Optional, Sequence, Tuple, Union,
)

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone

try:
    import numba as _numba

    _HAS_NUMBA = True
except Exception:  # pragma: no cover - numba is a hard dep; allow graceful skip.
    _numba = None  # type: ignore
    _HAS_NUMBA = False

try:
    import polars as pl  # type: ignore
    _HAS_POLARS = True
except Exception:  # pragma: no cover
    pl = None  # type: ignore
    _HAS_POLARS = False


def _is_polars_df(x: Any) -> bool:
    """Prefer explicit isinstance check over duck-typing."""
    return _HAS_POLARS and isinstance(x, pl.DataFrame)


from ..estimator import CompositeTargetEstimator, _y_train_clip_bounds
from ..transforms import get_transform
from ._corr_numba import safe_abs_corr_all_dispatch as _safe_abs_corr_all_dispatch

if TYPE_CHECKING:
    from ..transforms import Transform  # used as forward annotation in _tiny_cv_rmse_y_scale signature

logger = logging.getLogger(__name__)


def _extract_column_array(df: Any, col: str, rows: np.ndarray | None = None) -> np.ndarray:
    """Pull a single column out as a 1-D float32 ndarray. Polars / pandas
    only -- never materialise a whole-frame conversion.

    ``rows`` (optional integer index array): materialise ONLY those rows. The
    screening / rerank callers keep a 20-100k sample of a 4M+ row column, so
    pulling the full column first (``to_numpy()`` on a non-f32 dtype allocates
    every row) then slicing wastes O(N) per column over ~500 columns. The
    polars ``gather`` / pandas positional-take materialises O(len(rows)) and is
    value-identical to ``_extract_column_array(df, col)[rows]``.

    float32 halves the memory of the discovery feature-matrix vs the prior
    float64 default. On a 4M-row x ~500-col frame that's the difference
    between a 15.9 GB allocation (OOM on hosts running the trainer at ~100
    GB) and an 8 GB allocation (fits). MI / correlation kernels downstream
    are noise-bounded by sampling and binning, not by mantissa precision;
    polynomial-residual / multi-base linear least-squares promote to
    float64 internally at the per-call site (``_composite_transforms_linear``)
    where the conditioning actually matters.

    Return-value contract: callers MUST treat the result as read-only. When
    the source column already has float32 dtype the polars/pandas backing
    buffer is returned zero-copy; in-place mutation would corrupt the source
    DataFrame. Use ``.copy()`` at the call site if mutation is required."""
    if _is_polars_df(df):
        # Polars Series.to_numpy() already returns an ndarray; the prior
        # np.asarray wrapper allocated a redundant view. copy=False keeps
        # the astype zero-copy when the source dtype already matches.
        s = df.get_column(col)
        if rows is not None:
            s = s.gather(rows)
        return s.to_numpy().astype(np.float32, copy=False)
    if isinstance(df, pd.DataFrame):
        # na_value=np.nan so pandas nullable extension dtypes (Int64/Float64/boolean) holding NA cast to float32 instead of raising; no-op on plain numpy dtypes.
        if rows is not None:
            return df[col].iloc[rows].to_numpy(dtype=np.float32, na_value=np.nan)
        return df[col].to_numpy(dtype=np.float32, na_value=np.nan)
    raise TypeError(
        f"CompositeTargetDiscovery: unsupported df type {type(df).__name__}"
    )


def _is_numeric_column(df: Any, col: str) -> bool:
    """True if ``col`` is numeric in ``df``. Falls back to False on
    error -- discovery skips non-numeric base candidates rather than
    risking a cast bomb on object-dtype columns."""
    try:
        if _is_polars_df(df):
            import polars as pl  # lazy
            dtype = df.schema[col]
            # ``dtype.is_numeric()`` covers Float*, Int*, UInt* (added in
            # polars 0.19); fall back to a hard-coded set on older
            # versions that ship the dtypes but not the helper.
            try:
                return bool(dtype.is_numeric())
            except AttributeError:
                return dtype in {
                    pl.Float32, pl.Float64,
                    pl.Int8, pl.Int16, pl.Int32, pl.Int64,
                    pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
                }
        if isinstance(df, pd.DataFrame):
            return pd.api.types.is_numeric_dtype(df[col])
    except Exception:
        return False
    return False


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation that returns 0.0 (not NaN) on degenerate
    inputs (constant array, all-NaN). Used in the forbidden-base
    near-derived filter where NaN would falsely pass the threshold.

    Implemented via explicit centred-vector dot product instead of
    ``np.corrcoef`` to avoid the 2x2 matrix construction overhead
    (~1.17x faster on 80K rows).
    """
    finite = np.isfinite(a) & np.isfinite(b)
    n = int(finite.sum())
    if n < 3:
        return 0.0
    a_f = a[finite]
    b_f = b[finite]
    a_dev = a_f - a_f.mean()
    b_dev = b_f - b_f.mean()
    var_a = float(np.dot(a_dev, a_dev))
    var_b = float(np.dot(b_dev, b_dev))
    if var_a < 1e-24 or var_b < 1e-24:
        return 0.0
    return float(np.dot(a_dev, b_dev) / np.sqrt(var_a * var_b))


def _safe_abs_corr_all(
    y: np.ndarray, X: np.ndarray,
) -> np.ndarray:
    """Vectorised ``|corr(y, X[:, j])|`` for all j, size-aware backend dispatch.

    Public entry point: for large feature matrices (``n >= 20k`` AND ``F >= 64``,
    per the numba ladder) this dispatches to a ``numba.njit(parallel=True)`` kernel
    (``_corr_numba.safe_abs_corr_all_dispatch``) that walks each column in registers
    with no (n, F) centred temporary; small inputs use the numpy reference below
    (``_safe_abs_corr_all_numpy``). The kernel is numerically equivalent to the
    reference within ~1e-9 everywhere and ~1e-12 in the near-1 leak-threshold region
    (borderline columns re-decided with the exact numpy primitive -- see
    ``_corr_numba``). See ``_benchmarks/bench_safe_corr_dispatch.py`` for numbers.
    """
    return _safe_abs_corr_all_dispatch(
        y, X, reference_fn=_safe_abs_corr_all_numpy,
    )


def _safe_abs_corr_all_numpy(
    y: np.ndarray, X: np.ndarray,
) -> np.ndarray:
    """Vectorised numpy reference for ``|corr(y, X[:, j])|`` over all columns.

    Single matrix op gives 2.2x over the per-column loop on dense
    inputs (200 features x 80K rows: 558ms vs 1220ms). Returns 0.0
    for columns whose centred dot is below numerical tolerance.

    NaN handling differs from ``_safe_corr``: rows where ``y`` is
    non-finite are masked GLOBALLY (not per-column). For composite-
    target use this is acceptable because callers gate columns on
    sufficient finite count BEFORE passing them in (see
    ``_filter_features``); per-column NaN masking is reserved for
    the scalar ``_safe_corr`` path.
    """
    y_finite = np.isfinite(y)
    n_finite = int(y_finite.sum())
    if n_finite < 3:
        return np.zeros(X.shape[1])
    # Gate the row-subset copy: when y is all-finite (the typical case once the
    # caller has gated columns on finite count), X[y_finite] would copy the
    # whole (N, F) matrix for nothing -- on a 4M x 500 f32 frame that is an
    # 8 GB transient on top of the sampler's budgeted column_stack alloc.
    if n_finite == y_finite.shape[0]:
        y_f = y
        X_f = X
    else:
        y_f = y[y_finite]
        X_f = X[y_finite]
    y_dev = y_f - y_f.mean()
    var_y = float(np.dot(y_dev, y_dev))
    if var_y < 1e-24:
        return np.zeros(X.shape[1])
    # Keep the centred X_dev (numerically stable -- the sumsq-minus-n*mean^2
    # computational formula risks catastrophic cancellation on large-offset
    # columns, which would corrupt the near-1 leak-corr decision). But fold the
    # variance with einsum so we do NOT also allocate the (X_dev*X_dev) square
    # temporary: 3 full-matrix temps -> 1 (plus the X_f-copy gate above).
    X_means = X_f.mean(axis=0)
    X_dev = X_f - X_means
    var_X = np.einsum("ij,ij->j", X_dev, X_dev)
    out = np.zeros(X.shape[1])
    safe = var_X >= 1e-24
    if safe.any():
        cov = X_dev[:, safe].T @ y_dev
        denom = np.sqrt(var_y * var_X[safe])
        out[safe] = np.abs(cov / denom)
    return out


def _residualise(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """OLS-residualise ``y`` against ``x``. Returns ``y - alpha*x - beta``.
    If x is constant, returns ``y - mean(y)``."""
    finite = np.isfinite(y) & np.isfinite(x)
    if finite.sum() < 3 or np.std(x[finite]) < 1e-12:
        out = y.astype(np.float64).copy()
        out -= float(np.mean(out[finite])) if finite.any() else 0.0
        return out
    X = np.column_stack([x[finite].astype(np.float64), np.ones(int(finite.sum()))])
    coef, *_ = np.linalg.lstsq(X, y[finite].astype(np.float64), rcond=None)
    alpha = float(coef[0])
    beta = float(coef[1])
    out = y.astype(np.float64) - alpha * x - beta
    return out


def _mi_pair_bin(
    x: np.ndarray, y: np.ndarray, *, nbins: int,
) -> float:
    """Discrete MI between two 1-D continuous arrays via quantile binning.

    Discretises both axes into ``nbins`` quantile bins (so each bin
    holds ~equal mass), then computes
    ``MI = sum_ij p(i, j) * log(p(i, j) / (p_x(i) * p_y(j)))``
    using the joint frequency table. Equivalent to the bin-based MI
    estimator widely used in feature-selection libraries.

    Tradeoffs vs the kNN Kraskov estimator (sklearn default):

    - **5-10x faster** on n>1000: O(n + nbins^2) vs O(n*log(n))
      kd-tree queries.
    - **Biased low on heavy-tail distributions** because the equal-mass
      bins concentrate rare-tail values into one bin, hiding
      structure.
    - **Less sensitive to small sample size**: the kNN estimator
      becomes unstable below n=50; bin-based stays usable down to
      ~5*nbins rows.

    Implementation notes (engineering-honest, after benchmarking):

    Several optimisation attempts were tried and rejected:

    - **numba JIT of the full pipeline** (commit history: tried with
      both partial-JIT and full-JIT kernels). On n=1000 the JIT
      gives a 2.6x speedup, but on n>=10000 it is *slower* than
      numpy because numpy's sort / searchsorted / bincount are
      SIMD-vectorised C, and numba's JIT'd Python loops cannot
      beat them. Plus a one-shot ~5 s compile cost on first call.
      Production callers always pass mi_sample_n>=20K rows, so
      numpy wins where it matters. Removed.
    - **np.partition instead of np.quantile** for cut edges. The
      single-position partition is 1.5x faster than np.quantile,
      but the multi-position np.partition (one call selecting all
      nbins-1 positions) becomes O(n * nbins) and ends up *slower*
      on n>=100K than np.quantile's optimised sort-based path.
      Reverted to np.quantile.

    Verdict: the numpy implementation here is at the
    speed-of-vectorised-C floor for this algorithm. Further wins
    require dropping to a different algorithm entirely (e.g. a
    streaming hash-bin estimator that avoids the O(n log n) sort
    altogether).
    """
    finite = np.isfinite(x) & np.isfinite(y)
    if finite.sum() < 5 * nbins:
        return 0.0
    x_f = x[finite]
    y_f = y[finite]
    qs = np.linspace(0, 1, nbins + 1)[1:-1]
    x_edges = np.quantile(x_f, qs)
    y_edges = np.quantile(y_f, qs)
    x_idx = np.searchsorted(x_edges, x_f, side="right").astype(np.int64)
    y_idx = np.searchsorted(y_edges, y_f, side="right").astype(np.int64)
    np.clip(x_idx, 0, nbins - 1, out=x_idx)
    np.clip(y_idx, 0, nbins - 1, out=y_idx)
    combo = x_idx * nbins + y_idx
    joint_counts = np.bincount(combo, minlength=nbins * nbins).reshape(nbins, nbins)
    n_total = float(joint_counts.sum())
    if n_total <= 0:
        return 0.0
    pxy = joint_counts.astype(np.float64) / n_total
    px = pxy.sum(axis=1, keepdims=True)
    py = pxy.sum(axis=0, keepdims=True)
    nz = pxy > 0
    log_terms = np.zeros_like(pxy)
    log_terms[nz] = np.log(pxy[nz] / (px * py)[nz])
    mi = float((pxy * log_terms).sum())
    return max(0.0, mi)


# Bin codes only ever hold 0..nbins-1 (plus the -1 non-finite sentinel),
# so int16 (range -32768..32767) suffices for nbins up to int16-max and halves
# the (n_rows, n_cols) prebin buffer + improves gather/bincount cache locality
# vs the prior int64. BUT the codes feed ``_mi_from_binned_pair``'s
# ``combo = x_idx*nbins + y_idx`` whose max value is ``nbins**2 - 1``; that
# overflows int16 once ``nbins**2 - 1 > 32767`` i.e. ``nbins >= 182``. We gate
# the storage dtype on that boundary (int16 below, int32 from 182 up so the
# code values themselves always fit), and ``_mi_from_binned_pair`` upcasts the
# combo to int64 internally so the joint-count index never overflows regardless
# of the input code dtype -- keeping MI bit-identical to the int64 path.
_PREBIN_INT16_MAX_NBINS = 182  # nbins**2 - 1 must stay <= np.iinfo(np.int16).max (32767)


def _prebin_code_dtype(nbins: int) -> np.dtype:
    """Narrowest signed int dtype that holds bin codes 0..nbins-1 (and the -1
    sentinel) AND keeps ``nbins**2 - 1`` representable in int16 for the
    downstream combo. int16 for ``nbins < 182``, else int32."""
    return np.dtype(np.int16) if nbins < _PREBIN_INT16_MAX_NBINS else np.dtype(np.int32)


def _prebin_feature_columns(
    feature_matrix: np.ndarray, *, nbins: int,
) -> np.ndarray:
    """Pre-compute quantile edges and bin indices for every feature column.

    Returns an (n_samples, n_features) integer array of bin indices
    (0..nbins-1, with -1 marking non-finite rows).  Call once per
    screening pass; reuse across all candidate targets via
    ``_mi_to_target_prebinned``.  This saves the ``np.quantile`` +
    ``np.searchsorted`` cost per column per candidate -- for the
    typical 8-candidate × 25-feature sweep, roughly half the MI wall
    time.

    Dtype: int16 when ``nbins < 182`` else int32 (see ``_prebin_code_dtype``).
    int16 quarters the prebin buffer vs the prior int64 and improves
    gather/bincount cache locality; the downstream ``_mi_from_binned_pair``
    upcasts its ``combo`` internally so MI stays bit-identical to the int64 path.

    Uses ``np.nanquantile`` so NaN values don't corrupt the cut
    points; non-finite rows get sentinel -1 and are skipped by the
    downstream ``_mi_to_target_prebinned``.
    """
    code_dtype = _prebin_code_dtype(nbins)
    n_rows, n_cols = feature_matrix.shape
    if n_rows < 5 * nbins:
        return np.full((n_rows, n_cols), -1, dtype=code_dtype)
    q_edges = np.linspace(0.0, 1.0, nbins + 1)[1:-1]
    binned = np.empty((n_rows, n_cols), dtype=code_dtype)
    for j in range(n_cols):
        binned[:, j] = _prebin_one_column(
            feature_matrix[:, j], q_edges=q_edges, nbins=nbins, code_dtype=code_dtype,
        )
    return binned


def _prebin_one_column(
    col: np.ndarray, *, q_edges: np.ndarray, nbins: int, code_dtype: np.dtype,
) -> np.ndarray:
    """Bin ONE float column into int codes (0..nbins-1, -1 for non-finite).

    The exact per-column body lifted out of :func:`_prebin_feature_columns` so
    the eager (whole-matrix) and lazy (one-column-at-a-time) prebin paths run
    BYTE-IDENTICAL quantile + searchsorted + clip code -- guaranteeing the lazy
    code matrix is bit-identical to the eager one, not merely "equivalent".

    ``q_edges`` is the shared ``linspace(0,1,nbins+1)[1:-1]`` quantile grid; the
    caller computes it ONCE and reuses it across columns. ``col`` must be a
    contiguous float array of the screen sample's length."""
    col_finite = np.isfinite(col)
    if col_finite.sum() < 5 * nbins:
        return np.full(col.shape[0], -1, dtype=code_dtype)
    # nanquantile drops NaN but not inf; on an all-finite column np.quantile is bit-identical and faster, so take it in the common case.
    cut = np.quantile(col, q_edges) if col_finite.all() else np.nanquantile(col, q_edges)
    col_idx = np.full(col.shape[0], -1, dtype=code_dtype)
    col_idx[col_finite] = np.searchsorted(
        cut, col[col_finite], side="right",
    ).astype(code_dtype)
    np.clip(col_idx, 0, nbins - 1, out=col_idx, where=col_idx >= 0)
    return col_idx


def _prebin_feature_columns_lazy(
    df: Any, cols: Sequence[str], rows: np.ndarray, *, nbins: int,
) -> np.ndarray:
    """Lazy sibling of :func:`_prebin_feature_columns` for a DataFrame carrier.

    Pulls ONE column at a time from ``df`` (via :func:`_extract_column_array`
    on the SAMPLED ``rows``), bins it, writes only the int16/int32 codes into the
    output buffer, and lets the transient float32 column fall out of scope before
    the next is pulled. The whole ``(len(rows), F)`` float32 feature matrix is
    therefore NEVER materialised -- peak extra RAM is one column
    (``O(len(rows))``) instead of the full ``(len(rows), F)`` plane.

    BIT-IDENTICAL to ``_prebin_feature_columns(_build_feature_matrix(df, cols,
    rows))``: each column is binned by the SHARED :func:`_prebin_one_column`
    kernel on the SAME extracted float32 values, so the resulting code matrix
    matches the eager path element-for-element. Only the float-matrix transient
    is avoided -- the numerics are untouched.

    100GB-frame rule: ``rows`` is the small MI-screen sample (``mi_sample_n``),
    and the only float allocation alive at any instant is one sampled column, so
    even a 4M x 500 polars frame never builds the (n, F) float plane. The eager
    path stays the default for ndarray carriers / where the float matrix is also
    consumed downstream (dedup / knn); the caller size-gates the dispatch."""
    code_dtype = _prebin_code_dtype(nbins)
    n_rows = int(np.asarray(rows).shape[0])
    n_cols = len(cols)
    if n_rows < 5 * nbins:
        return np.full((n_rows, n_cols), -1, dtype=code_dtype)
    q_edges = np.linspace(0.0, 1.0, nbins + 1)[1:-1]
    binned = np.empty((n_rows, n_cols), dtype=code_dtype)
    for j, c in enumerate(cols):
        col = _extract_column_array(df, c, rows=rows)
        binned[:, j] = _prebin_one_column(
            col, q_edges=q_edges, nbins=nbins, code_dtype=code_dtype,
        )
        del col  # drop the float32 transient before pulling the next column.
    return binned


# Shared FS/MRMR <-> discovery prebin cache: investigated and REJECTED as UNSAFE (not feasible).
# MRMR feature-selection (filters/mrmr.py) and composite discovery use two INDEPENDENT MI stacks;
# sharing a single per-suite feature-prebin cache across them would CHANGE the bins feeding MI ->
# change MI -> change which features get selected. They are not interchangeable on four axes:
#   1. strategy: MRMR default ``nbins_strategy='mdlp'`` is SUPERVISED Fayyad-Irani discretization
#      (target-aware, per-feature adaptive edge counts via _adaptive_nbins.per_feature_edges, also
#      knuth/blocks/fd variants); discovery here is UNSUPERVISED fixed equi-mass quantile binning
#      (np.quantile(col, linspace(0,1,nbins+1)[1:-1]) + searchsorted). Supervised != unsupervised
#      edges -> different codes for the same column.
#   2. nbins: MRMR ``quantization_nbins=10`` default vs discovery ``nbins=16`` default.
#   3. row population: MRMR bins its OWN train rows; discovery bins the MI-SCREEN SAMPLE
#      (_sample_indices -> mi_sample_n subset). Different rows -> different quantile edges.
#   4. code dtype/layout: MRMR int32 (discretize_2d_array, min_ncats=50) vs discovery int16/int32.
# Because the codes diverge, a shared cache is NOT bit-identical and would silently alter selection
# -- forbidden per the bit-identity gate. The discovery-internal PrebinCache (cache.py) already
# de-duplicates re-bins WITHIN discovery (same data+sample+nbins across configs/targets), which is
# the only safe, bit-identical sharing seam. Do not force a cross-stack cache.


def _prebin_feature_columns_cached(
    feature_matrix: np.ndarray, *, nbins: int, use_cache: bool = True,
) -> np.ndarray:
    """Cache-consulting wrapper around :func:`_prebin_feature_columns`.

    The bin codes are deterministic on ``(feature_matrix VALUES + dtype + shape, nbins)`` -- a
    second discovery on the SAME screen sample + nbins but a different config recomputes the
    IDENTICAL codes. This wrapper keys an in-process :class:`PrebinCache` by a content hash
    (``prebin_matrix_signature``) so the second run skips the O(n*F*log n) per-column quantile
    binning and returns BIT-IDENTICAL codes from cache.

    100GB-frame rule: ``feature_matrix`` here is the SMALL screen sample (``mi_sample_n`` rows),
    not the raw frame; the cached value is the int16/int32 code matrix (half/quarter the float
    bytes) and is size-gated inside ``PrebinCache.put`` so a pathological sample is never pinned.

    ``use_cache=False`` bypasses lookup AND store (force-recompute path for benches / tests that
    measure the uncached cost or assert bit-identity against a fresh recompute).
    """
    if not use_cache:
        return _prebin_feature_columns(feature_matrix, nbins=nbins)
    from ..cache import get_prebin_cache, prebin_matrix_signature

    cache = get_prebin_cache()
    sig = prebin_matrix_signature(feature_matrix, nbins)
    cached = cache.get(sig)
    if cached is not None:
        return cached
    codes = _prebin_feature_columns(feature_matrix, nbins=nbins)
    cache.put(sig, codes)
    return codes


def _mi_per_feature_prebinned(
    feature_binned: np.ndarray,
    target: np.ndarray,
    *,
    nbins: int,
    exclude_col: int | None = None,
) -> np.ndarray | None:
    """Per-feature MI(target, feature_binned[:, j]) vector (length F).

    Returns None for the degenerate-shape / too-few-finite cases the aggregate
    treats as 0.0. The per-feature values are INDEPENDENT of which other
    columns are present, so a caller that screens many bases (each excluding
    one column) can compute this vector ONCE and derive ``mi_y`` per base by
    excluding the base's column, instead of re-binning + re-MI'ing the shared
    columns per base.

    ``exclude_col`` (optional column index): when set, the returned vector spans
    every column EXCEPT that one -- i.e. it is bit-identical to passing
    ``np.delete(feature_binned, exclude_col, axis=1)`` but WITHOUT materialising
    the deleted (n, F-1) matrix copy. The per-column loop simply skips
    ``exclude_col`` and packs the survivors in the same ``[0..k-1, k+1..]`` order
    ``np.delete`` would produce, so each survivor's MI is computed from the exact
    same column and the result is identical element-for-element. This lets a
    base-loop caller hold the full prebinned matrix once and derive each base's
    ``mi_y`` by exclusion with zero per-base matrix allocation.
    """
    if feature_binned.shape[0] == 0 or feature_binned.shape[1] == 0:
        return None
    if feature_binned.shape[0] != target.shape[0]:
        return None
    n_cols_in = feature_binned.shape[1]
    drop = exclude_col if (exclude_col is not None and 0 <= exclude_col < n_cols_in) else None
    if drop is not None and n_cols_in == 1:
        return None  # excluding the only column leaves an empty feature set.
    # Gate the size check on target-finite only; the per-column inner loop
    # handles its own -1 sentinel masking (a COLUMN-0-NaN early return would
    # zero MI for the whole batch silently).
    finite = np.isfinite(target)
    n_fin = int(finite.sum())
    if n_fin < 5 * nbins:
        return None
    # Gate the whole-matrix boolean slice: on an all-finite target the
    # feature_binned[finite] copy is the whole (n, F) int64 matrix for nothing.
    if n_fin == finite.shape[0]:
        t_f = target
        fb_f = feature_binned
    else:
        t_f = target[finite]
        fb_f = feature_binned[finite]
    qs = np.linspace(0.0, 1.0, nbins + 1)[1:-1]
    t_edges = np.nanquantile(t_f, qs)
    t_idx = np.searchsorted(t_edges, t_f, side="right").astype(np.int64)
    np.clip(t_idx, 0, nbins - 1, out=t_idx)
    out_len = fb_f.shape[1] - (1 if drop is not None else 0)
    per_feat = np.empty(out_len, dtype=np.float64)
    out_j = 0  # write cursor into the survivor vector (skips ``drop``).
    for j in range(fb_f.shape[1]):
        if drop is not None and j == drop:
            continue
        col_b = fb_f[:, j]
        # Filter out -1 sentinel (non-finite feature rows) for this column
        col_valid = col_b >= 0
        n_cv = int(col_valid.sum())
        if n_cv < 5 * nbins:
            per_feat[out_j] = 0.0
            out_j += 1
            continue
        # Skip the two per-column gathers when the column has no sentinel.
        if n_cv == col_b.shape[0]:
            per_feat[out_j] = _mi_from_binned_pair(col_b, t_idx, nbins=nbins)
        else:
            per_feat[out_j] = _mi_from_binned_pair(
                col_b[col_valid], t_idx[col_valid], nbins=nbins,
            )
        out_j += 1
    return per_feat


def _aggregate_mi_per_feature(per_feat: np.ndarray | None, aggregation: str) -> float:
    if per_feat is None or per_feat.size == 0:
        return 0.0
    if aggregation == "sum":
        return float(np.sum(per_feat))
    return float(np.mean(per_feat))


def _aggregate_mi_per_feature_excluding(
    per_feat: np.ndarray | None, aggregation: str, exclude_col: int,
) -> float:
    """Aggregate ``per_feat`` over every entry EXCEPT ``exclude_col``.

    Bit-identical to ``_aggregate_mi_per_feature(np.delete(per_feat, exclude_col),
    aggregation)`` but builds the survivor subset via a boolean mask rather than
    ``np.delete``. ``per_feat[mask]`` yields the exact same contiguous
    ``[0..k-1, k+1..]`` ordering ``np.delete`` produces, so ``np.sum`` / ``np.mean``
    over it reduce in identical order (FP-associativity preserved). The mask
    subset is a tiny length-F vector; the win is dropping the redundant
    ``np.delete`` call that the base-loop ran twice per base.
    """
    if per_feat is None or per_feat.size == 0:
        return 0.0
    if not (0 <= exclude_col < per_feat.size):
        return _aggregate_mi_per_feature(per_feat, aggregation)
    mask = np.ones(per_feat.size, dtype=bool)
    mask[exclude_col] = False
    return _aggregate_mi_per_feature(per_feat[mask], aggregation)


def _mi_to_target_prebinned(
    feature_binned: np.ndarray,
    target: np.ndarray,
    *,
    nbins: int,
    aggregation: str = "mean",
    exclude_col: int | None = None,
) -> float:
    """MI between pre-binned feature columns and ``target``.

    ``feature_binned`` is the output of ``_prebin_feature_columns`` --
    integer bin indices (0..nbins-1, -1 for non-finite rows) with the
    SAME number of rows as ``target``.  Only the target is binned here
    (once), saving the per-column quantile + searchsorted cost.

    ``exclude_col`` (optional): aggregate over every feature column EXCEPT this
    one without allocating an ``np.delete``-d (n, F-1) matrix copy -- the
    per-feature loop skips it in place. Bit-identical to pre-deleting the column.
    """
    per_feat = _mi_per_feature_prebinned(
        feature_binned, target, nbins=nbins, exclude_col=exclude_col,
    )
    return _aggregate_mi_per_feature(per_feat, aggregation)


def _mi_from_binned_pair_numpy(
    x_idx: np.ndarray, y_idx: np.ndarray, *, nbins: int,
) -> float:
    """Numpy reference for :func:`_mi_from_binned_pair` (kept callable for tests / benches).

    ``x_idx`` / ``y_idx`` may arrive as int16 (the narrowed prebin code
    dtype). The flattened index ``x_idx*nbins + y_idx`` reaches ``nbins**2 - 1``,
    which overflows int16 at ``nbins >= 182`` (and even int16*python-scalar can
    wrap on intermediate products), so the combo is computed in int64 explicitly.
    This is purely a width promotion -- the index values are unchanged, so MI is
    bit-identical to the legacy int64-code path. ``np.bincount`` requires int.
    """
    combo = x_idx.astype(np.int64) * nbins + y_idx
    joint_counts = np.bincount(combo, minlength=nbins * nbins).reshape(nbins, nbins)
    n_total = float(joint_counts.sum())
    if n_total <= 0:
        return 0.0
    pxy = joint_counts.astype(np.float64) / n_total
    px = pxy.sum(axis=1, keepdims=True)
    py = pxy.sum(axis=0, keepdims=True)
    nz = pxy > 0
    log_terms = np.zeros_like(pxy)
    log_terms[nz] = np.log(pxy[nz] / (px * py)[nz])
    return max(0.0, float((pxy * log_terms).sum()))


if _HAS_NUMBA:
    @_numba.njit(cache=True, fastmath=False)
    def _mi_from_binned_pair_njit_kernel(x_idx, y_idx, nbins):  # type: ignore[no-untyped-def]
        # Single-pass joint histogram + marginals, then MI = sum pxy*log(pxy/(px*py)).
        # Reproduces the numpy reference's arithmetic term-for-term: the joint counts are
        # integer-exact (no FP), px/py are the same row/col sums, and each non-zero cell adds
        # pxy*log(pxy/(px*py)) -- the only divergence from numpy is FP reduction ORDER of the
        # final accumulation (numpy reduces the (nbins,nbins) product array; this walks cells
        # row-major), which lands ~1e-16 on the natural-log MI scale, far under the 1e-12 gate.
        n = x_idx.shape[0]
        joint = np.zeros(nbins * nbins, dtype=np.int64)
        for i in range(n):
            joint[int(x_idx[i]) * nbins + int(y_idx[i])] += 1
        n_total = 0
        for k in range(nbins * nbins):
            n_total += joint[k]
        if n_total <= 0:
            return 0.0
        inv_n = 1.0 / float(n_total)
        # Row (px) and column (py) marginal probabilities.
        px = np.zeros(nbins, dtype=np.float64)
        py = np.zeros(nbins, dtype=np.float64)
        for a in range(nbins):
            base = a * nbins
            for b in range(nbins):
                c = joint[base + b]
                if c != 0:
                    p = c * inv_n
                    px[a] += p
                    py[b] += p
        mi = 0.0
        for a in range(nbins):
            base = a * nbins
            pxa = px[a]
            for b in range(nbins):
                c = joint[base + b]
                if c != 0:
                    pxy = c * inv_n
                    mi += pxy * np.log(pxy / (pxa * py[b]))
        if mi < 0.0:
            return 0.0
        return mi


def _mi_from_binned_pair(
    x_idx: np.ndarray, y_idx: np.ndarray, *, nbins: int,
) -> float:
    """MI from two already-binned integer arrays (0..nbins-1).

    Hot kernel: called ~9.8k times per discovery run (per-feature MI AND inside the
    per-permutation null loop in ``_auto_base``, so the cost multiplies with
    ``n_targets x auto_base_top_k x npermutations``). Dispatches to a ``numba.njit``
    single-pass histogram+MI kernel (``cache=True``) which is ~4x faster than the numpy
    ``bincount``+log path at the production sample size and bit-identical within ~1e-12
    (only the final-sum FP reduction order differs). The numpy reference stays callable as
    ``_mi_from_binned_pair_numpy`` for tests / benches; falls back to it when numba is
    unavailable.
    """
    if not _HAS_NUMBA:
        return _mi_from_binned_pair_numpy(x_idx, y_idx, nbins=nbins)
    # Contiguous int64 inputs let the kernel index without per-element dtype branches; the
    # asarray is a no-op when the caller already passes contiguous codes (the common path
    # slices a column / shuffled codes), so no (n)-copy on the hot loop.
    xi = np.ascontiguousarray(x_idx)
    yi = np.ascontiguousarray(y_idx)
    return float(_mi_from_binned_pair_njit_kernel(xi, yi, int(nbins)))


def _mi_per_feature_y_fixed(
    feature_matrix: np.ndarray,
    y: np.ndarray,
    *,
    nbins: int,
) -> np.ndarray:
    """Per-feature MI between each column of ``feature_matrix`` and a fixed ``y``.

    Specialises ``_mi_pair_bin`` for the (feature-loop, fixed-target) case
    used by ``CompositeTargetDiscovery._auto_base``: the naive
    ``[_mi_pair_bin(F[:, j], y, nbins) for j in cols]`` re-quantiles ``y``
    on every iteration; this hoists the y-binning out of the loop. The
    per-feature work reduces to ``np.quantile(col) + searchsorted +
    bincount`` -- mathematically identical to the per-call path, so the
    returned per-feature MI is bit-exact vs the naive baseline.

    Inputs are assumed already cross-column finite-masked by the caller
    (matches ``_mi_pair_bin``'s contract when used inside the feature
    loop at ``composite_discovery._auto_base``).

    Benchmark (n=500K, k=30, nbins=50): naive 2888.5 ms -> hoisted 1729.3 ms => 1.67x faster.
    Matches the naive ``_mi_pair_bin`` loop within the ``_mi_from_binned_pair`` njit kernel's
    ~1e-12 FP-reduction-order contract (integer contingency tables identical; only the final
    MI-sum reduction order differs).
    """
    n_rows, n_cols = feature_matrix.shape
    out = np.zeros(n_cols, dtype=np.float64)
    if n_rows < 5 * nbins or n_cols == 0:
        return out
    qs = np.linspace(0.0, 1.0, nbins + 1)[1:-1]
    y_edges = np.quantile(y, qs)
    y_idx = np.searchsorted(y_edges, y, side="right").astype(np.int64)
    np.clip(y_idx, 0, nbins - 1, out=y_idx)
    for j in range(n_cols):
        col = feature_matrix[:, j]
        x_edges = np.quantile(col, qs)
        x_idx = np.searchsorted(x_edges, col, side="right").astype(np.int64)
        np.clip(x_idx, 0, nbins - 1, out=x_idx)
        out[j] = _mi_from_binned_pair(x_idx, y_idx, nbins=nbins)
    return out


def _mi_per_feature_y_fixed_per_col(
    feature_matrix: np.ndarray,
    y: np.ndarray,
    *,
    nbins: int,
    min_finite: int = 50,
) -> np.ndarray:
    """Per-feature MI(y, x_j) with PER-PAIR (per-column) NaN masking.

    NaN-aware sibling of ``_mi_per_feature_y_fixed``. Where that function
    assumes the caller already cross-column finite-masked the matrix (so
    every column shares ONE row subset), this one masks each column on its
    OWN ``isfinite(col) & isfinite(y)`` rows -- mirroring ``_mi_pair_bin``'s
    internal mask and the ``_mi_per_feature_prebinned`` ``-1``-sentinel
    contract.

    Motivation: ``_auto_base`` previously ranked candidates by MI computed on
    the GLOBAL all-column finite intersection
    (``np.all(isfinite(x_matrix), axis=1)``). For mid-range-NaN columns that
    intersection is a non-random (MNAR) subset of rows -- the MI ranking is
    estimated on exactly the rows where every feature happens to be observed,
    biasing which base wins. Per-pair masking estimates each column's MI on
    its own observed rows, removing the selection bias and matching the
    per-pair contract already used by ``_mi_to_target``.

    Bit-identity: on an all-finite ``feature_matrix`` (and finite ``y``)
    every per-pair mask is the full row set, the hoisted y-binning is reused
    for every column, and the result is bit-identical to
    ``_mi_per_feature_y_fixed``. A column with too few jointly-finite rows
    (``< min_finite``) returns 0.0 for that column only (never NaN, never
    pulling rows out from under its neighbours).
    """
    n_rows, n_cols = feature_matrix.shape
    out = np.zeros(n_cols, dtype=np.float64)
    if n_cols == 0:
        return out
    y_finite = np.isfinite(y)
    n_y = int(y_finite.sum())
    if n_y < max(min_finite, 5 * nbins):
        return out
    qs = np.linspace(0.0, 1.0, nbins + 1)[1:-1]
    # When y is fully finite (the common case) bin it ONCE and reuse the
    # codes for the no-NaN columns; only re-bin y on the surviving rows for
    # the columns that actually carry NaN. This keeps the all-finite path
    # bit-identical to (and as fast as) ``_mi_per_feature_y_fixed``.
    y_all_finite = n_y == y_finite.shape[0]
    if y_all_finite:
        y_edges_full = np.quantile(y, qs)
        y_idx_full = np.searchsorted(y_edges_full, y, side="right").astype(np.int64)
        np.clip(y_idx_full, 0, nbins - 1, out=y_idx_full)
    for j in range(n_cols):
        col = feature_matrix[:, j]
        col_finite = np.isfinite(col)
        if y_all_finite and col_finite.all():
            # No extra NaN in this column -> shared full-row path (the
            # ``_mi_per_feature_y_fixed`` branch, bit-identical).
            x_edges = np.quantile(col, qs)
            x_idx = np.searchsorted(x_edges, col, side="right").astype(np.int64)
            np.clip(x_idx, 0, nbins - 1, out=x_idx)
            out[j] = _mi_from_binned_pair(x_idx, y_idx_full, nbins=nbins)
            continue
        pair = col_finite & y_finite
        if int(pair.sum()) < max(min_finite, 5 * nbins):
            out[j] = 0.0
            continue
        col_p = col[pair]
        y_p = y[pair]
        x_edges = np.quantile(col_p, qs)
        x_idx = np.searchsorted(x_edges, col_p, side="right").astype(np.int64)
        np.clip(x_idx, 0, nbins - 1, out=x_idx)
        y_edges = np.quantile(y_p, qs)
        y_idx = np.searchsorted(y_edges, y_p, side="right").astype(np.int64)
        np.clip(y_idx, 0, nbins - 1, out=y_idx)
        out[j] = _mi_from_binned_pair(x_idx, y_idx, nbins=nbins)
    return out


def _mi_to_target(
    feature_matrix: np.ndarray,
    target: np.ndarray,
    *,
    n_neighbors: int,
    random_state: int,
    estimator: str = "knn",
    nbins: int = 16,
    aggregation: str = "mean",
) -> float:
    """Aggregated MI of each feature column with ``target``.

    Two estimators:

    - ``"knn"`` (default): sklearn's Kraskov kNN estimator. Higher
      accuracy on heavy-tail distributions; slow on n>10k.
    - ``"bin"``: quantile-bin estimator (``_mi_pair_bin`` per column).
      5-10x faster; biased low on heavy-tail.

    Aggregation across feature columns:

    - ``"mean"`` (default):
      ``sum_j MI(x_j, target) / n_features``. Invariant to feature
      count -- the metric stays comparable when the feature set
      shrinks (because the base column was excluded). The legacy
      ``"sum"`` aggregation is biased:
      sum-of-marginal-MI overcounts shared information when X is
      correlated, AND the over-count differs between numerator
      ``MI(T, X_no_base)`` and denominator ``MI(y, X_no_base)``
      because changing the target reweights how features overlap.
      Mean is the simplest fix that removes the dimension confound.
    - ``"sum"``: legacy behaviour. Set explicitly for backward-
      compatibility on existing benchmarks.

    Per-pair finite masking (NOT a global all-column AND-mask): each
    column's MI is computed on the rows where THAT column AND the target
    are both finite. A single mostly-NaN column therefore only degrades
    its own MI to 0.0 rather than dropping the rows out from under every
    other column -- the old ``np.all(isfinite(feature_matrix), axis=1)``
    intersection let one 99%-NaN column zero the entire sweep (one bad
    column could leave <50 jointly-observed rows for a 25-column matrix
    of otherwise-dense features). This mirrors the prebinned path's
    per-column ``-1`` sentinel handling in ``_mi_per_feature_prebinned``
    and ``_mi_pair_bin``'s own internal per-pair mask.
    """
    if feature_matrix.ndim != 2 or feature_matrix.shape[1] == 0:
        return 0.0
    target_finite = np.isfinite(target)
    if int(target_finite.sum()) < 50:
        return 0.0
    n_cols = feature_matrix.shape[1]
    per_feat = np.zeros(n_cols, dtype=np.float64)
    if estimator == "bin":
        # ``_mi_pair_bin`` masks ``isfinite(x) & isfinite(y)`` itself, so
        # pass full columns + full target; its own gate returns 0.0 for
        # columns with too few jointly-finite rows.
        for j in range(n_cols):
            per_feat[j] = _mi_pair_bin(feature_matrix[:, j], target, nbins=nbins)
    else:
        from sklearn.feature_selection import mutual_info_regression
        # sklearn's Kraskov estimator cannot ingest NaN, so mask per pair
        # here and run the single-column estimator on the surviving rows.
        for j in range(n_cols):
            col = feature_matrix[:, j]
            pair_finite = target_finite & np.isfinite(col)
            if int(pair_finite.sum()) < 50:
                per_feat[j] = 0.0
                continue
            per_feat[j] = float(mutual_info_regression(
                col[pair_finite].reshape(-1, 1), target[pair_finite],
                n_neighbors=n_neighbors, random_state=random_state,
            )[0])
    if aggregation == "sum":
        return float(np.sum(per_feat))
    # Default: mean.
    return float(np.mean(per_feat))


# Tiny-model RMSE / CV helpers live in sibling ``_screening_tiny``; re-exported here.
from ._screening_tiny import (  # noqa: F401, E402
    _silence_tiny_model_output,
    _build_tiny_model,
    _tiny_cv_rmse_raw_y,
    _tiny_cv_rmse_y_scale_multiseed,
    _tiny_cv_rmse_raw_y_multiseed,
    _per_bin_rmse,
    _per_bin_from_fold_preds,
    _tiny_cv_rmse_y_scale,
)


def _sample_indices(
    n: int, sample_n: int | None, random_state: int,
    *,
    strategy: str = "random",
    y: np.ndarray | None = None,
    n_strata: int = 10,
) -> np.ndarray:
    """Return a sorted array of row indices to use for MI screening.

    Two strategies:

    - ``"random"`` (default): uniform random sample of ``sample_n``
      rows from ``n``. Cheap, unbiased on average, but high-variance
      on heavy-tail targets (the rare-tail rows that carry most of
      the signal can be over- or under-represented in any one draw).

    - ``"stratified_quantile"``: bin ``y`` into ``n_strata`` quantile
      bins, then sample ``sample_n / n_strata`` rows from each bin.
      Tail rows get oversampled relative to natural frequency. Use
      when ``y`` is heavy-tail (financial returns, fraud scores,
      queue lengths) -- gives stable MI rankings across runs because
      each bin contributes a guaranteed number of rows.

    Sorted so the (mostly-temporal) row order is preserved -- avoids
    biasing the MI estimate on temporal data.
    """
    if sample_n is None or n <= sample_n:
        return np.arange(n)
    rng = np.random.default_rng(random_state)
    if strategy == "random" or y is None or n_strata < 2:
        idx = rng.choice(n, size=sample_n, replace=False)
        idx.sort()
        return idx
    if strategy != "stratified_quantile":
        raise ValueError(
            f"_sample_indices: unknown strategy '{strategy}'. "
            "Choose from 'random' or 'stratified_quantile'."
        )

    # Stratified quantile sampling. Bin y into n_strata quantile
    # bins, sample ceil(sample_n / n_strata) from each.
    y_arr = np.asarray(y, dtype=np.float64).reshape(-1)
    if y_arr.size != n:
        # Caller passed mismatched y; fall back to random.
        idx = rng.choice(n, size=sample_n, replace=False)
        idx.sort()
        return idx
    finite_mask = np.isfinite(y_arr)
    if finite_mask.sum() < n_strata * 2:
        # Too few finite y; can't stratify, fall back to random.
        idx = rng.choice(n, size=sample_n, replace=False)
        idx.sort()
        return idx
    # Compute quantile cuts on finite y.
    qs = np.linspace(0, 1, n_strata + 1)[1:-1]
    cuts = np.quantile(y_arr[finite_mask], qs)
    # Assign each finite row to a stratum [0, n_strata-1]; non-finite
    # rows get a separate stratum at the end so they aren't dropped
    # silently.
    stratum = np.searchsorted(cuts, y_arr, side="right")
    np.clip(stratum, 0, n_strata - 1, out=stratum)
    stratum[~finite_mask] = n_strata  # extra "non-finite" bin

    per_stratum = max(1, sample_n // n_strata)
    picked: list[np.ndarray] = []
    for s in range(n_strata + 1):
        bin_rows = np.where(stratum == s)[0]
        if bin_rows.size == 0:
            continue
        take = min(bin_rows.size, per_stratum)
        if take == bin_rows.size:
            picked.append(bin_rows)
        else:
            chosen = rng.choice(bin_rows, size=take, replace=False)
            picked.append(chosen)
    out = np.concatenate(picked) if picked else np.arange(min(n, sample_n))
    # Downsample the per-stratum overshoot uniformly with the seeded rng; truncating after sort would excise only the largest (latest) indices, a systematic temporal bias against the sort-for-temporal-fidelity rationale.
    if out.size > sample_n:
        out = rng.choice(out, size=sample_n, replace=False)
    out.sort()
    return out
