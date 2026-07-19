"""Dataset-level discretisation entry points.

`categorize_dataset` is the top-level entry called by ``MRMR.fit`` (pandas /
polars, NaN-aware, optional adaptive per-feature binning). It leans on the
numeric-column kernels and missing-value handling in the parent module
``discretization``; those are lazy-imported in-body to avoid an import cycle.
`create_redundant_continuous_factor` is a test / benchmark synthetic-data helper.
"""
from __future__ import annotations

import hashlib
import logging
import os
import threading
from typing import Any, Callable, Optional, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_xxh3_128: Optional[Callable] = None
try:
    import xxhash as _xxhash

    _xxh3_128 = _xxhash.xxh3_128_digest
except Exception as e:  # nosec B110 - xxhash optional: falls back to blake2b below
    logger.debug("xxhash unavailable, falling back to blake2b for content hashing: %s", e)


def _discretize_input_dtype():
    """Working dtype for the numeric matrix that ``categorize_dataset`` discretises.

    ``categorize_dataset`` copies ALL numeric columns into one dense array before binning
    (``arr = df[...].to_numpy(...)``) -- a second full-frame copy that coexists with the caller's
    (already large) engineered float frame, and is the dominant term of the large-n FE peak (a 1M-row
    fit projects to ~21GB, OOMing a 16GB box). Quantile / MDLP edges + searchsorted do NOT need
    float64: float32 edges differ only at ~1e-7 (far below the selection-altering ~1e-3 bar) and only
    for values sitting exactly on a bin edge.

    DEFAULT ON as of 2026-07-09 (the ~21GB->~10GB memory cut; corrective mechanisms default ON once
    validated -- see ``tests/feature_selection/MRMR_FE_PERF_NOTES.md`` 2026-06-17 entry, which measured
    selection IDENTICAL float32-vs-float64 on the canonical 60k fit). Set ``MLFRAME_DISCRETIZE_FLOAT32=0``
    to force the legacy float64 path (e.g. bit-exact replay of a pre-2026-07-09 fit).
    """
    if os.environ.get("MLFRAME_DISCRETIZE_FLOAT32", "").strip() in ("0", "false", "False"):
        return np.float64
    return np.float32


# ---------------------------------------------------------------------------
# Cross-instance per-column code cache for the UNSUPERVISED numeric path.
# ---------------------------------------------------------------------------
# Unsupervised binning (``nbins_strategy=None`` -> ``discretize_2d_array``) bins each numeric column
# INDEPENDENTLY of every other column AND of the target, so a column's ordinal codes are a pure function of
# (its values, n_bins, method, min_ncats, dtype). Across a suite that trains MANY targets on the SAME feature
# frame, every feature column recurs byte-identically while only the (injected) target column is new, so
# caching per-column codes lets the 2nd..Nth target reuse the feature codes and rebin only the target. The
# cache is process-wide + LRU-bounded; keyed on a content hash so a hit is bit-identical BY CONSTRUCTION.
# NOTE: deliberately NOT used on the supervised ``nbins_strategy`` path (mdlp/optimal_joint/...), where bins
# depend on y -> a feature's codes differ per target and sharing would be WRONG. Disable via env if needed.
from collections import OrderedDict as _OrderedDict

_NUMERIC_CODE_CACHE: "_OrderedDict[bytes, np.ndarray]" = _OrderedDict()
# Total bytes of cached column-code arrays to retain (column codes are n_rows*dtype; gate so a 100GB-frame
# fit cannot pin unbounded RAM). Override via env. Default 512 MB.
_NUMERIC_CODE_CACHE_MAX_BYTES = int(os.environ.get("MLFRAME_DISCRETIZE_COL_CACHE_MAX_BYTES", str(512 * 1024 * 1024)))
_NUMERIC_CODE_CACHE_BYTES = 0
# categorize_dataset runs under joblib backend="threading"; guard every mutation of the OrderedDict AND the
# byte counter so concurrent workers cannot corrupt the dict (RuntimeError on concurrent mutation) or drift
# the accounting. The expensive discretize kernel + blake2b hashing stay OUTSIDE the lock -> still parallel.
_NUMERIC_CODE_CACHE_LOCK = threading.Lock()


def _log_numeric_code_cache_occupancy() -> None:
    """Emit a DEBUG-level snapshot of the process-wide numeric-code cache's current occupancy (entry count + bytes vs its cap), so cache growth/eviction is visible when needed without polling the module globals directly."""
    logger.debug(
        "categorize_dataset: numeric-code cache occupancy: %d entries, %d bytes (cap %d bytes).",
        len(_NUMERIC_CODE_CACHE), _NUMERIC_CODE_CACHE_BYTES, _NUMERIC_CODE_CACHE_MAX_BYTES,
    )


def clear_numeric_code_cache() -> int:
    """Drop the process-wide per-column unsupervised-discretization code cache; returns the entry count cleared."""
    global _NUMERIC_CODE_CACHE_BYTES
    with _NUMERIC_CODE_CACHE_LOCK:
        n = len(_NUMERIC_CODE_CACHE)
        _NUMERIC_CODE_CACHE.clear()
        _NUMERIC_CODE_CACHE_BYTES = 0
    return n


def _discretize_2d_array_col_cached(arr, *, n_bins, method, min_ncats, dtype, discretize_2d_array):
    """Per-column cached wrapper over ``discretize_2d_array`` for the unsupervised numeric path.

    Bit-identical to ``discretize_2d_array(arr, ...)`` (each column is rebuilt by the SAME kernel on the
    SAME values; only previously-seen columns are served from cache). Columns not yet cached are computed in
    ONE batched ``discretize_2d_array`` call over the uncached sub-matrix, then memoised. Gated off when the
    cache is disabled, the matrix is empty, or a single column's codes would exceed the byte budget.
    """
    global _NUMERIC_CODE_CACHE_BYTES
    if os.environ.get("MLFRAME_DISCRETIZE_COL_CACHE", "1").strip() in ("0", "false", "False"):
        return discretize_2d_array(arr=arr, n_bins=n_bins, method=method, min_ncats=min_ncats, min_values=None, max_values=None, dtype=dtype)
    n_rows, n_cols = (arr.shape[0], arr.shape[1]) if arr.ndim == 2 else (arr.shape[0] if arr.ndim else 0, 0)
    if n_cols == 0 or n_rows == 0:
        return discretize_2d_array(arr=arr, n_bins=n_bins, method=method, min_ncats=min_ncats, min_values=None, max_values=None, dtype=dtype)

    _param_tag = repr((int(n_bins), str(method), int(min_ncats), np.dtype(dtype).str)).encode()
    # Hash every column FIRST, outside the lock (the hashing is the costly part and needs no shared
    # state), so concurrent threads hash in parallel.
    # ONE bulk transpose instead of ``n_cols`` separate ``np.ascontiguousarray(arr[:, j])`` copies: each
    # column-slice of the C-order ``arr`` is non-contiguous (strided), so copying them one at a time (the
    # previous code) pays that strided-gather cost ``n_cols`` times; ``arr.T`` made contiguous ONCE gives
    # each column as a contiguous ROW (``arrT[j]``), no further copy needed. Bit-identical bytes either way.
    arrT = np.ascontiguousarray(arr.T)
    keys: list = []
    if _xxh3_128 is not None:
        # xxh3 (non-cryptographic, content-only fingerprint -- this cache key never needs collision
        # resistance against an adversary) is an order of magnitude faster than blake2b for this purpose;
        # 128-bit digest keeps the same collision margin blake2b's digest_size=16 gave.
        keys.extend(_xxh3_128(arrT[j]) + _param_tag for j in range(n_cols))
    else:
        for j in range(n_cols):
            hh = hashlib.blake2b(arrT[j].tobytes(), digest_size=16)
            hh.update(_param_tag)
            keys.append(hh.digest())

    out = np.empty((n_rows, n_cols), dtype=dtype)
    uncached_cols: list = []
    with _NUMERIC_CODE_CACHE_LOCK:
        for j in range(n_cols):
            hit = _NUMERIC_CODE_CACHE.get(keys[j])
            if hit is not None and hit.shape[0] == n_rows:
                out[:, j] = hit
                _NUMERIC_CODE_CACHE.move_to_end(keys[j])
            else:
                uncached_cols.append(j)

    if uncached_cols:
        # Compute the uncached columns OUTSIDE the lock (the whole point of the threading backend).
        sub = np.ascontiguousarray(arr[:, uncached_cols])
        sub_codes = discretize_2d_array(arr=sub, n_bins=n_bins, method=method, min_ncats=min_ncats, min_values=None, max_values=None, dtype=dtype)
        _per_col_bytes = n_rows * np.dtype(dtype).itemsize
        with _NUMERIC_CODE_CACHE_LOCK:
            for _i, j in enumerate(uncached_cols):
                col_codes = np.ascontiguousarray(sub_codes[:, _i])
                out[:, j] = col_codes
                # ``keys[j] not in cache`` guards the byte counter against a concurrent thread that
                # recomputed the SAME column meanwhile (the stored array is bit-identical either way, but
                # double-adding its bytes would drift the accounting and over-evict).
                if _per_col_bytes <= _NUMERIC_CODE_CACHE_MAX_BYTES and keys[j] not in _NUMERIC_CODE_CACHE:
                    _NUMERIC_CODE_CACHE[keys[j]] = col_codes
                    _NUMERIC_CODE_CACHE_BYTES += _per_col_bytes
                    while _NUMERIC_CODE_CACHE_BYTES > _NUMERIC_CODE_CACHE_MAX_BYTES and len(_NUMERIC_CODE_CACHE) > 1:
                        _ek, _ev = _NUMERIC_CODE_CACHE.popitem(last=False)
                        _NUMERIC_CODE_CACHE_BYTES -= _ev.shape[0] * _ev.dtype.itemsize
    return out


def create_redundant_continuous_factor(
    df: pd.DataFrame,
    factors: Sequence[str],
    agg_func: Callable = np.sum,
    noise_percent: float = 5.0,
    dist: Optional[Any] = None,
    dist_args: tuple = (),
    name: str | None = None,
    sep: str = "_",
    random_state: int | np.random.RandomState | None = None,
) -> None:
    """Out of a few continuous factors, craft a new factor with known relationship and amount of redundancy. Used by tests / benchmark harnesses, not by ``MRMR`` directly.

    ``random_state`` seeds the fallback uniform noise (and is passed to ``dist.rvs`` when ``dist`` supports it) so the crafted factor is reproducible without disturbing
    numpy's process-global RNG.
    """
    from sklearn.utils import check_random_state

    rng = check_random_state(random_state)
    if dist:
        rvs = dist.rvs
        # Wave 31 (2026-05-20): assert -> AttributeError.
        if not callable(rvs):
            raise AttributeError(f"dist must have a callable .rvs method; got {dist!r}.")
        noise = rvs(*dist_args, size=len(df), random_state=rng)
    else:
        noise = rng.random(len(df))

    val_min, val_max = noise.min(), noise.max()
    if np.isclose(val_max, val_min):
        noise = np.zeros(len(noise), dtype=np.float32)
    else:
        noise = (noise - val_min) / (val_max - val_min)

    if not name:
        name = sep.join(factors) + sep + f"{noise_percent:.0f}%{dist.name if dist else ''}noise"

    df[name] = agg_func(df[factors].values, axis=1) * (1 + (noise - 0.5) * noise_percent / 100)


def categorize_dataset(
    df,
    method: str = "quantile",
    n_bins: int = 4,
    min_ncats: int = 50,
    dtype=np.int16,
    missing_strategy: str = "fillna_zero",
    nbins_strategy: Optional[str] = None,
    nbins_strategy_kwargs: Optional[dict] = None,
    y_for_strategy=None,
    cache_dir: Optional[str] = None,
    max_categorical_cardinality: Optional[int] = None,
):
    """Convert a DataFrame into an ordinal-encoded ``(n_samples, n_features)`` array. Accepts pandas or polars (DataFrame or LazyFrame -- materialised at the
    boundary). ``missing_strategy`` controls NaN handling: see :func:`_handle_missing`."""
    from . import (
        _handle_missing,
        _maybe_collect_lazy,
        _multi_col_factorize_native,
        discretize_2d_array,
    )

    df = _maybe_collect_lazy(df)

    data = None
    numerical_cols = []
    categorical_factors = []

    try:
        import polars as pl
        _is_polars = isinstance(df, pl.DataFrame)
    except ImportError:
        _is_polars = False

    if _is_polars:
        def _is_pl_cat(dt):
            """True if a polars dtype should be treated as categorical (string/categorical/boolean/enum) rather than numeric for discretization purposes."""
            return dt == pl.Utf8 or dt == pl.String or dt == pl.Categorical or dt == pl.Boolean or (hasattr(pl, "Enum") and isinstance(dt, pl.Enum))

        numerical_cols = [name for name, dt in df.schema.items() if not _is_pl_cat(dt)]
        categorical_cols_detected = [name for name, dt in df.schema.items() if _is_pl_cat(dt)]
    else:
        numerical_cols = df.head(5).select_dtypes(exclude=("category", "object", "string", "bool")).columns.values.tolist()
        categorical_cols_detected = None

    _dt = _discretize_input_dtype()
    if _is_polars:
        # An all-categorical polars frame (no numeric columns) makes df.select([]).to_numpy() collapse to shape (0, 0)
        # -- ZERO rows -- so the later per-column np.append(numeric, categorical) raises a row-count mismatch. Build an
        # explicit (n_rows, 0) array instead so the categorical branch initialises/extends a full-height matrix (pandas
        # already yields (n, 0) here).
        if numerical_cols:
            arr = df.select(numerical_cols).to_numpy().astype(_dt, copy=False)
        else:
            arr = np.empty((df.height, 0), dtype=_dt)
    else:
        arr = df[numerical_cols].to_numpy(dtype=_dt, na_value=np.nan)

    # Snapshot the NaN positions BEFORE _handle_missing rewrites them: the
    # "separate_bin" strategy fills NaN with the column median so np.percentile
    # produces clean edges, then we overwrite the same positions in the
    # discretized output with bin=n_bins (max+1 per column). Net effect: NaN
    # gets its own honest category that MI estimators see correctly.
    # 2026-05-30 Wave 9.1 fix (loop iter 11): include 'propagate' alongside
    # 'separate_bin' so NaN positions get re-routed to the dedicated NaN
    # bin instead of silently colliding with the top real bin via
    # np.searchsorted(NaN -> ej.size).
    _nan_mask = np.isnan(arr) if (missing_strategy in ("separate_bin", "propagate") and arr.size > 0) else None

    # Unified NaN handling for both pandas and polars. Pass the already-computed ``_nan_mask`` (when
    # available) so _handle_missing does not rescan ``arr`` for NaN a second time.
    arr = _handle_missing(arr, strategy=missing_strategy, nan_mask=_nan_mask)

    # 2026-05-29 Wave 7: per-column adaptive bin chooser.
    # When ``nbins_strategy`` is provided, compute per-column edges via the
    # _adaptive_nbins dispatcher, apply them with np.searchsorted, and pad to
    # the global max nbins so downstream MRMR sees a uniform-nbins matrix.
    #
    # PERF NOTE (2026-06-19, wide-data reuse audit CK-secondary): the unsupervised path below
    # (nbins_strategy is None -> _discretize_2d_array_col_cached -> discretize_2d_array) ALREADY routes to the
    # CUDA quantile kernel where it wins (measured 2.89x: 1276ms->442ms on the (20000,2000) discretize op;
    # size-gated via _DISCRETIZE_SPEC). The SUPERVISED path here (mdlp/optimal_joint/...) has NO GPU kernel --
    # measured ~10.5s at (20000,2000) (~52s extrapolated to 100k) vs ~3.4s for the CUDA-eligible quantile path.
    # A CUDA MDLP would be a large NEW recursive-supervised kernel; it is DEFERRED behind the dominant Fleuret
    # CMI redundancy loop (~290s at 100k) which the batched-CUDA CMI kernel attacks first. A cheaper interim
    # lever (column-parallelising per_feature_edges) is a candidate if MDLP discretization becomes the wall.
    if nbins_strategy is not None:
        from .._adaptive_nbins import per_feature_edges
        _strategy_kwargs = dict(nbins_strategy_kwargs or {})
        # Pass y if the strategy is supervised.
        _needs_y = str(nbins_strategy).lower() in (
            "mdlp", "fayyad_irani", "optimal_joint", "cv",
            "mah", "mah_sci", "sci", "marx",
        )
        _y_arr = None
        if _needs_y and y_for_strategy is not None:
            _y_arr = np.asarray(y_for_strategy).ravel()
        edges_per_col = per_feature_edges(
            arr, y=_y_arr, method=nbins_strategy, cache_dir=cache_dir, **_strategy_kwargs,
        )
        # Dispatcher-level silent-degenerate-fallback guardrail (2026-07-19): ``per_feature_edges`` already
        # logs a per-column WARNING when a strategy returns empty edges for a column with real variance (see
        # its own guardrail), but ``categorize_dataset`` is the actual top-level entry MRMR.fit calls and it
        # previously used ``edges_per_col`` completely blindly -- an empty-edges column silently becomes an
        # all-bin-0 column further down (the ``+inf``-padded searchsorted resolves every value to bin 0), with
        # NO signal at this call site. This is exactly how the MDLP overflow bug (3.0**n_classes -> inf,
        # acceptance check always False) went undetected in production: nothing at the entry point a caller
        # actually watches ever surfaced it. Aggregate + name the affected columns here so a production fit's
        # logs are diagnosable without per-strategy vigilance, regardless of which strategy is selected.
        _degenerate_col_names: list = []
        for _j, _ej in enumerate(edges_per_col):
            if _ej is not None and hasattr(_ej, "size") and _ej.size == 0:
                _col_vals = arr[:, _j]
                _col_finite = _col_vals[np.isfinite(_col_vals)]
                if _col_finite.size and np.unique(_col_finite).size > 1:
                    _degenerate_col_names.append(numerical_cols[_j] if _j < len(numerical_cols) else _j)
        if _degenerate_col_names:
            logger.warning(
                "categorize_dataset: nbins_strategy=%r produced EMPTY bin edges for %d column(s) with real "
                "variance -- each collapses to a single degenerate bin (all rows share one code), silently "
                "destroying that column's MI signal for the rest of this fit: %s",
                nbins_strategy, len(_degenerate_col_names), _degenerate_col_names[:20],
            )
        # Per-column searchsorted; pad to global max nbins.
        n_rows = arr.shape[0]
        n_cols = arr.shape[1]
        per_col_bins = [int(e.size + 1) for e in edges_per_col]
        max_bins = max(max(per_col_bins) if per_col_bins else 1, 1)
        # Validate the requested dtype can hold ``max_bins`` (matches the
        # post-discretize NaN-bin overflow check below).
        if max_bins > np.iinfo(dtype).max:
            raise ValueError(
                f"nbins_strategy={nbins_strategy!r} produced {max_bins} bins which "
                f"exceeds dtype {dtype} max {np.iinfo(dtype).max}. "
                f"Use a wider dtype or constrain the strategy (e.g. knuth_m_max_cap=64)."
            )
        data = np.empty((n_rows, n_cols), dtype=dtype)
        # BATCHED (2026-07-13): pad every column's interior-edge vector to the SAME length
        # (``n_edges_max``, already implied by ``max_bins`` above) with ``+inf`` sentinels, then run ONE
        # ``_searchsorted_2d_right_njit_parallel`` call over the padded (n_edges_max, n_cols) matrix instead
        # of a per-column ``np.searchsorted`` Python-dispatch loop -- this file's own prior comment flagged
        # this exact gap ("max_bins is already computed one line above"). Padding is safe: ``side='right'``
        # counts edges <= v, and a finite v is never >= +inf, so padding entries are never counted --
        # identical to the per-column call with that column's OWN (shorter) edge vector. A column with
        # ``ej.size == 0`` is an all-``+inf`` row: every comparison resolves to bin 0, matching the
        # original explicit ``data[:, j] = 0`` branch exactly. NaN never reaches this call (``_handle_missing``
        # above unconditionally fills or raises on NaN for every strategy), so the kernel's separate
        # NaN-walks-to-``len(edges)`` contract -- which WOULD differ under padding -- never engages here.
        # PARALLEL (not the serial ``nogil`` twin): measured -- the serial kernel is a WASH-TO-SLIGHT-LOSS
        # against numpy's per-column ``searchsorted`` (0.83-1.26x across n=20k-100k, p=200-4000; numpy's C
        # binary search is already fast per call, so a single-threaded scalar njit loop doesn't beat it) while
        # the ``prange``-parallel twin is a consistent 6-10x win at the same scales. Safe here (unlike the
        # FE pair-search hot path, which must stay serial to avoid nesting under joblib ``backend="threading"``):
        # ``categorize_dataset`` has exactly ONE call site (``_fit_impl_core.py``, ``MRMR.fit``'s top-level
        # entry, before any joblib worker spawns), so there is no nested-parallelism hazard.
        n_edges_max = max((int(e.size) for e in edges_per_col), default=0)
        edges_padded = np.full((n_edges_max, n_cols), np.inf, dtype=np.float64)
        for j, ej in enumerate(edges_per_col):
            if ej.size:
                edges_padded[: ej.size, j] = ej
        _codes = np.empty((n_rows, n_cols), dtype=np.int64)
        from ._kernels import _searchsorted_2d_right_njit_parallel
        # ``arr`` passed at its NATIVE dtype (float32 or float64): the kernel promotes per-element against
        # the float64 edges, byte-identical to pre-upcasting the whole column (see the kernel's own
        # docstring) -- avoids an (n_rows, n_cols) float64 copy on top of the input buffer.
        _searchsorted_2d_right_njit_parallel(edges_padded, arr, _codes)
        data[:, :] = _codes.astype(dtype)
    else:
        # Unsupervised numeric path: each column binned independently of others AND of the target, so per-column
        # codes are cached cross-instance (huge win across a suite's many targets on one feature frame). Bit-identical.
        data = _discretize_2d_array_col_cached(
            arr, n_bins=n_bins, method=method, min_ncats=min_ncats, dtype=dtype,
            discretize_2d_array=discretize_2d_array,
        )

    if _nan_mask is not None and _nan_mask.any():
        # 2026-05-30 Wave 9.1 fix (loop iter 9): per-COLUMN NaN bin code.
        # Pre-fix used the constructor ``n_bins`` as the dedicated NaN code
        # for every column, but the adaptive ``nbins_strategy`` branch
        # produces per-column bin counts that often exceed ``n_bins``
        # (e.g. FD gives ~22 for n=600 N(0,1), while ctor n_bins=4). So the
        # NaN code 4 silently collided with regular real-data bin 4 - NaN
        # observations got merged into a real bin, destroying the
        # missingness signal and biasing every downstream MI / SU / MRMR
        # score. Fix: each column's NaN code is one past that column's
        # highest regular code. Per-column scheme works because downstream
        # MI estimators treat each column independently and
        # ``data.max(axis=0) + 1`` (line 1151) recomputes ``nbins`` per col.
        if nbins_strategy is not None:
            nan_codes_per_col = np.asarray(per_col_bins, dtype=np.int64)
        else:
            # Legacy unsupervised path: discretize_2d_array can emit a real code == n_bins for some columns
            # ([0..n_bins], n_bins+1 distinct codes), so a flat NaN code = ctor n_bins collides with that real
            # top bin and destroys the missingness signal -- the same defect the adaptive branch fixes per-column.
            # ``max(n_bins, col_real_max + 1)`` keeps the NaN code at n_bins wherever it is already distinct
            # (bit-identical to the prior behaviour) and only pushes it one past the real max for a colliding column.
            col_real_max = data.max(axis=0).astype(np.int64) if data.size else np.zeros(arr.shape[1], dtype=np.int64)
            nan_codes_per_col = np.maximum(int(n_bins), col_real_max + 1)
        max_bin_after = int(nan_codes_per_col.max())
        if max_bin_after > np.iinfo(data.dtype).max:
            raise ValueError(
                f"separate_bin strategy needs dtype able to hold {max_bin_after}; "
                f"current dtype {data.dtype} max is {np.iinfo(data.dtype).max}. "
                "Pass a wider dtype to categorize_dataset."
            )
        # Per-column NaN code: broadcast across NaN-row positions.
        _rows, _c = np.where(_nan_mask)
        data[_rows, _c] = nan_codes_per_col[_c].astype(data.dtype)

    if _is_polars:
        if categorical_cols_detected:
            cast_exprs = []
            for c in categorical_cols_detected:
                dt = df.schema[c]
                if dt == pl.Boolean:
                    cast_exprs.append(pl.col(c).cast(pl.UInt32))
                elif dt in (pl.Utf8, pl.String):
                    cast_exprs.append(pl.col(c).cast(pl.Categorical).to_physical())
                else:
                    cast_exprs.append(pl.col(c).to_physical())
            _coded = df.select(cast_exprs)
            categorical_cols = categorical_cols_detected
            new_vals = _coded.to_numpy()
            if max_categorical_cardinality:
                # Fold the rare-category tail so the polars path honors the cap exactly like the pandas branch below;
                # cap_categorical_cardinality needs float64 to write the remapped codes (physical codes are unsigned).
                from . import cap_categorical_cardinality
                new_vals = cap_categorical_cardinality(new_vals.astype(np.float64, copy=False), int(max_categorical_cardinality))
        else:
            categorical_cols = []
            new_vals = None
    else:
        categorical_factors = df.select_dtypes(include=("category", "object", "string", "bool"))
        categorical_cols = []
        if categorical_factors.shape[1] > 0:
            categorical_cols = categorical_factors.columns.values.tolist()
            new_vals = _multi_col_factorize_native(categorical_factors)
            if max_categorical_cardinality:
                from . import cap_categorical_cardinality
                new_vals = cap_categorical_cardinality(new_vals, int(max_categorical_cardinality))
        else:
            new_vals = None
    if categorical_cols and new_vals is not None:
        # 2026-05-30 Wave 9.1 fix (loop iter 31): the categorical block
        # bypassed ``missing_strategy`` entirely. ``_multi_col_factorize_native``
        # / ``pd.factorize`` / ``.cat.codes`` emit ``-1`` for NaN, which then
        # silently flowed into the joint-histogram allocator and got
        # negative-index wrapped to the LAST real category bin (or, under
        # unsigned dtype, wrapped to 2^bits - 1 = a phantom huge category).
        # Net effect: NaN observations silently merged with the largest
        # real category, biasing every MI / SU / MRMR score on columns
        # with NaN in pd.Categorical / object / string / bool columns.
        # Sibling of iter 9 (numeric NaN bin collision) and iter 11
        # (propagate strategy silent merge).
        #
        # Fix: shift codes by +1 so NaN sentinel becomes 0 and real
        # categories become 1..K. Under ``missing_strategy='separate_bin'``
        # (the default) this gives NaN its own honest bin. Under
        # 'fillna_zero' the shift is equivalent: NaN ends up at bin 0
        # which any downstream code reading "0 = first category" treats
        # uniformly. Under 'raise', refuse if any -1 sentinel present.
        if _missing_strategy_str := str(missing_strategy):
            _has_nan = bool((new_vals < 0).any())
            if _has_nan and _missing_strategy_str == "raise":
                _nan_cnt = int((new_vals < 0).sum())
                raise ValueError(
                    f"categorize_dataset: {_nan_cnt} NaN value(s) in " f"categorical column(s) {categorical_cols} with " f"missing_strategy='raise'."
                )
            if _has_nan:
                # Shift +1: -1 -> 0, k -> k+1. Cast back to dtype after
                # shift (the shift increases the max by 1; auto-promote
                # below catches dtype overflow on the new max).
                new_vals = new_vals + 1
        # An empty block (0 rows) has no codes to bound; skip the overflow check whose ``.max(axis=0)`` would raise on the empty reduction axis.
        global_max = int(new_vals.max(axis=0).max()) if new_vals.size else -1
        max_cats = new_vals.max(axis=0) if new_vals.size else None
        if global_max > np.iinfo(dtype).max:
            assert max_cats is not None  # global_max > 0 implies new_vals.size > 0
            for _candidate in (np.int16, np.int32, np.int64):
                if global_max <= np.iinfo(_candidate).max:
                    logger.warning(
                        "categorize_dataset: %d category code(s) exceeded dtype %s; auto-promoting to %s to avoid silent wraparound.",
                        int((max_cats > np.iinfo(dtype).max).sum()),
                        dtype,
                        _candidate,
                    )
                    dtype = _candidate
                    break
            else:
                raise ValueError(f"categorize_dataset: category cardinality {global_max} exceeds int64 max; cannot encode.")
        new_vals = new_vals.astype(dtype)

        if data is None:
            data = new_vals
        else:
            data = np.append(data, new_vals, axis=1)

    # ``data.max(axis=0)`` raises on an empty reduction axis (0 columns OR 0 rows): return a typed empty result so callers
    # get a consistent ``(data, cols, nbins)`` triple instead of an opaque reduction ValueError.
    if data is None or data.size == 0:
        n_rows = data.shape[0] if data is not None else 0
        n_cols = len(numerical_cols) + len(categorical_cols)
        empty = data if data is not None else np.empty((n_rows, n_cols), dtype=dtype)
        _log_numeric_code_cache_occupancy()
        return empty, numerical_cols + categorical_cols, np.zeros(n_cols, dtype=np.int64)

    nbins = data.max(axis=0).astype(np.int64) + 1

    _log_numeric_code_cache_occupancy()
    return data, numerical_cols + categorical_cols, nbins
