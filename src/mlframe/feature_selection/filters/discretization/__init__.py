"""Numeric -> ordinal discretisation pipeline.

Public API
----------
* ``categorize_dataset(df, ...)`` -- top-level entry called by ``MRMR.fit``. Accepts pandas or polars (DataFrame / LazyFrame autocollected).
* ``discretize_array(arr, ...)`` -- single-column 1-D discretiser.
* ``discretize_2d_array(arr, ...)`` -- column-parallel njit version.
* ``discretize_sklearn(arr, ...)`` -- pure-numpy port of sklearn's ``KBinsDiscretizer`` for cases where sklearn's overhead matters.
* Lower-level numba helpers ``digitize``, ``quantize_dig``, ``quantize_search``, ``discretize_uniform``, ``get_binning_edges``.

Polars ``LazyFrame`` is auto-collected at the boundary. Both pandas and polars paths route NaN through a shared ``_handle_missing`` helper -- the chosen
strategy is documented and applied identically to both engines (legacy pandas silently used ``fillna(0.0)``; legacy polars let NaN propagate).
"""
from __future__ import annotations

import logging
import math
import os
import sys
import threading
import warnings
from typing import Any, Optional, Union

import numpy as np
import pandas as pd
from numba import njit, prange
# 2026-05-28: sklearn / astropy removed from categorize_1d_array hot path.
# Pure-numpy + numba kernels are ~10x faster than KBinsDiscretizer / OrdinalEncoder
# (single-threaded estimator-API overhead) and ~12x faster than astropy.histogram
# for the supported bin schemes. The legacy methods 'astropy' and 'discretizer'
# still resolve via thin compat shims below.
def _safe_code_dtype(n_bins: int, dtype: type) -> type:
    """Widen ``dtype`` to one that can hold ordinal codes ``0..n_bins-1``.

    Discretiser codes reach ``n_bins-1``; the default ``int8`` only holds 0..127, so an
    ``astype(int8)`` on ``n_bins>128`` wraps the top bins negative (modulo arithmetic),
    silently mis-binning the high-magnitude region. Auto-widen instead of silently
    corrupting: int8->int16->int32->int64 as the bin count grows. No-op for n_bins<=128.
    """
    try:
        info = np.iinfo(np.dtype(dtype))  # type: ignore[type-var]  # dtype may be non-integer; caught below
    except (ValueError, TypeError):
        return dtype  # non-integer requested dtype: caller owns the contract
    if n_bins - 1 <= info.max:
        return dtype
    for cand in (np.int16, np.int32, np.int64):
        if n_bins - 1 <= np.iinfo(cand).max:
            return cand
    return np.int64


def _native_ordinal_encode_2d(vals: np.ndarray) -> np.ndarray:
    """Drop-in pure-numpy replacement for sklearn OrdinalEncoder().fit_transform on a (n, 1) array.

    Returns float64 ordinals so downstream digitize / dtype-promotion logic stays bit-for-bit
    identical to the sklearn path. ``pd.factorize`` is asymptotically the same numpy unique
    + inverse-index lookup but skips estimator-validation overhead (~6x faster at n=10k).
    """
    flat = vals.reshape(-1)
    codes, _ = pd.factorize(flat, use_na_sentinel=True)
    return np.asarray(codes, dtype=np.float64).reshape(vals.shape)


def cap_categorical_cardinality(codes_2d: np.ndarray, max_cardinality: int) -> np.ndarray:
    """Collapse the rare-category tail of each column so no column exceeds ``max_cardinality`` distinct codes.

    Per column: the ``max_cardinality - 1`` MOST FREQUENT codes keep distinct ids ``0..max_cardinality-2`` (ordered by
    frequency desc), all rarer codes fold into one "other" bucket ``max_cardinality-1``; the ``-1`` NaN sentinel is
    preserved. Columns already at/below the cap are returned UNCHANGED (bit-identical). This is the standard high-card
    reduction: a categorical whose cardinality approaches N has sparse contingency cells, so its plug-in MI/CMI is
    unreliable anyway (the analytic null already guards on >=5 expected/cell) -- folding the rare tail DENSIFIES the cells
    (better MI) while capping the code range so the whole codes matrix fits a narrow int (int8 for cap<=127). Operates on
    a float64 code matrix (the ``_multi_col_factorize_native`` output) and returns float64.
    """
    if max_cardinality is None or max_cardinality < 2 or codes_2d.size == 0:
        return codes_2d
    out = codes_2d
    _copied = False
    _n, p = codes_2d.shape
    for j in range(p):
        col = codes_2d[:, j]
        finite = col[col >= 0]
        if finite.size == 0:
            continue
        # cardinality = number of distinct non-negative codes; cheap via max (codes are dense 0..k-1 from factorize).
        k = int(finite.max()) + 1
        if k <= max_cardinality:
            continue
        counts = np.bincount(finite.astype(np.int64), minlength=k)
        # keep the (cap-1) most frequent as 0..cap-2 (freq desc); everything else -> the "other" bucket cap-1.
        # DISCRETIZATION-13 fix: kind="stable" so two categories tied exactly at
        # the cutoff boundary have a deterministic, documented tie-break (first-seen/lowest-index order)
        # rather than relying on the non-guaranteed default sort's tie behaviour across numpy
        # versions/architectures. Sort by NEGATED counts (descending) directly, rather than
        # ascending-then-reverse -- reversing a stable ascending sort would itself un-stabilize the tie
        # order (it reverses ties' relative order too), defeating the point of "stable".
        keep = np.argsort(-counts, kind="stable")[: max_cardinality - 1]
        remap = np.full(k, max_cardinality - 1, dtype=np.float64)
        remap[keep] = np.arange(max_cardinality - 1, dtype=np.float64)
        if not _copied:
            out = codes_2d.copy()
            _copied = True
        mask = col >= 0
        out[mask, j] = remap[col[mask].astype(np.int64)]
    return out


def _multi_col_factorize_native(categorical_df: "pd.DataFrame") -> np.ndarray:
    """Multi-column ordinal encoding without sklearn OrdinalEncoder.

    Strategy (in order of preference):

    1. Pre-Categorical columns -> read ``.cat.codes`` directly (single C-level
       attribute access, no recomputation, no GIL contention). NaN is already
       encoded as -1 by pandas convention -- matches downstream contract.
    2. Non-Categorical object / string / bool columns -> joblib-threaded
       ``pd.factorize`` (releases GIL on the hash-table fill, threading wins).
    3. Single-column fallback -> sequential loop (zero overhead).

    Ordering contract: distinct categories get distinct integer codes; NaN -> -1.
    Code values themselves are NOT bit-for-bit identical to sklearn's
    OrdinalEncoder (.cat.codes uses category-dictionary order; OrdinalEncoder
    uses first-occurrence). For downstream MI estimation the value mapping is
    semantically equivalent (MI is invariant under bijective relabeling).

    Bench on 100-col 200k-row pre-Categorical DF (representative MRMR workload):
    ~7x faster than the sequential pd.factorize loop AND no GIL contention
    so callers can multi-thread on top.
    """
    n_rows = len(categorical_df)
    cols = list(categorical_df.columns)
    if not cols:
        return np.empty((n_rows, 0), dtype=np.float64)

    out = np.empty((n_rows, len(cols)), dtype=np.float64)
    needs_factorize: list = []  # (j, col) for non-pre-categorical columns
    for _j, _c in enumerate(cols):
        _ser = categorical_df[_c]
        if isinstance(_ser.dtype, pd.CategoricalDtype):
            # Fast path: ``.cat.codes`` is a vectorised C-level attribute
            # access; NaN already encoded as -1.
            out[:, _j] = _ser.cat.codes.to_numpy(dtype=np.float64, copy=False)
        else:
            needs_factorize.append((_j, _c))

    if needs_factorize:
        # Threshold raised 1 -> 8 (7-site joblib.Parallel audit, 2026-07-19): isolated/warmed/best-of-3+
        # measurement at the realistic column-count range for this branch (2-8 non-Categorical columns)
        # found the joblib threading pool never clearly wins there -- 2 cols -> 1.29x (but that case was
        # already serial pre-fix, since the old threshold was ``<= 1``), 8 cols -> 0.52x (loses), 40 cols
        # -> 0.93x (roughly even). No clean win was found across 2-8 columns, so that whole range now stays
        # serial; the pool is still used above 8 columns where the per-thread pd.factorize work is enough
        # to amortise joblib's dispatch overhead.
        if len(needs_factorize) <= 8:
            for _j, _c in needs_factorize:
                _codes, _ = pd.factorize(categorical_df[_c], use_na_sentinel=True)
                out[:, _j] = _codes.astype(np.float64)
        else:
            # joblib threading. pd.factorize releases the GIL on the hash build,
            # so threads parallelise. prefer='threads' avoids the pickling cost
            # of process workers on a categorical DF view.
            from joblib import Parallel, delayed as _delayed
            _results = Parallel(n_jobs=min(8, len(needs_factorize)), prefer="threads")(
                _delayed(lambda c: pd.factorize(categorical_df[c], use_na_sentinel=True)[0].astype(np.float64))(_c) for _j, _c in needs_factorize
            )
            for (_j, _), _codes in zip(needs_factorize, _results):
                out[:, _j] = _codes
    return out


def _native_kbins_quantile(vals: np.ndarray, n_bins: int) -> np.ndarray:
    """Drop-in pure-numpy replacement for sklearn KBinsDiscretizer(strategy='quantile', encode='ordinal').

    Uses np.nanpercentile for edge calc + np.searchsorted for bin lookup. ~12x faster than
    KBinsDiscretizer at n=10k single-column because we skip BaseEstimator validation +
    sklearn's CSR-friendly hot-path scaffolding. Output shape matches sklearn's: (n, 1) float64.
    """
    flat = np.asarray(vals, dtype=np.float64).reshape(-1)
    quantiles = np.linspace(0.0, 100.0, n_bins + 1)
    bin_edges = np.nanpercentile(flat, quantiles)
    # Inner edges only (drop both extremes, like sklearn's KBinsDiscretizer does internally).
    inner = bin_edges[1:-1]
    codes = np.searchsorted(inner, flat, side="right").astype(np.float64)
    return codes.reshape(vals.shape if vals.ndim == 2 else (-1, 1))


from mlframe.core.arrays import arrayMinMax

from ._discretization_edges import (
    _bayesian_blocks_bin_edges,
    _bayesian_blocks_inner,
    _bayesian_blocks_midpoints,
    _knuth_bin_edges,
    _knuth_log_posterior,
    discretize_sklearn,
    edges,
    get_binning_edges,
    histogram,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Unified missing-value handling for pandas / polars / numpy paths
# =============================================================================


def _handle_missing(arr: np.ndarray, *, strategy: str = "fillna_zero", nan_mask: Optional[np.ndarray] = None) -> np.ndarray:
    """Apply the configured NaN handling strategy.

    ``"fillna_zero"`` (legacy pandas behaviour): replace NaN with 0.0. Biases
    MI by mixing NaN rows into bin-0 with true-zero values; kept only for
    reproducibility of pre-2026-05-15 runs.
    ``"separate_bin"``: pass-through here; ``categorize_dataset`` handles the
    post-discretize bin-assignment so NaN rows land in a dedicated max+1 bin
    per column, making MI estimators see them as an honest category.
    ``"raise"``: refuse a column with NaN.
    ``"propagate"``: alias of ``"separate_bin"`` since the Wave 9.1
    iter-11 fix. Previously documented as "leave NaN in place" but
    that silently merged NaN rows into the column's TOP real bin via
    ``np.searchsorted`` (NaN -> ej.size = max real code), destroying
    any missingness-as-signal. Now median-fills here and the caller
    re-routes NaN positions to the dedicated NaN bin (2026-05-30 Wave 9.1
    iter-11 fix: propagate used to return the NaN-bearing array unchanged,
    silently merging NaN rows with the highest-value real category --
    verified a NaN-is-the-signal column dropped from MI=0.69 nats under
    separate_bin to MI=0.38 under the old propagate).

    ``nan_mask``: optional precomputed ``np.isnan(arr)`` boolean mask (same shape as ``arr``).
    ``categorize_dataset`` already computes this mask once (it needs it separately for the post-discretize
    NaN-bin re-routing), so passing it here avoids re-scanning the whole ``arr`` for NaN a second time on
    top of the ``np.nanmedian`` scan below. ``None`` (default) falls back to computing the mask internally,
    for any caller that does not have one on hand. Private -- external callers should use the public
    ``discretize_*`` family.

    Mutates ``arr`` in place (returns the same object) for the ``separate_bin``/``propagate`` fill when
    ``arr`` is writable. A polars single-numeric-column selection can hand back a genuine zero-copy,
    READ-ONLY view onto the DataFrame's Arrow buffer (verified empirically: ``df.select([one_col]).to_numpy()``
    is non-writeable; multi-column selections always materialise a fresh writable buffer since interleaving
    columns into one 2-D array cannot be zero-copy) -- mutating that in place would corrupt the caller's
    live data. Falls back to the legacy allocating ``np.where`` fill in that case.
    """
    _mask = nan_mask if nan_mask is not None else np.isnan(arr)
    if not _mask.any():
        return arr
    if strategy == "fillna_zero":
        return np.where(_mask, 0.0, arr)
    if strategy in ("separate_bin", "propagate"):
        # The actual bin re-routing happens in categorize_dataset after discretization. Here we replace
        # NaN with column median so np.percentile produces clean bin edges; the original NaN positions
        # are preserved via the caller's nan-mask and overwritten back to max_bin+1 there.
        # nanmedian emits RuntimeWarning("All-NaN slice encountered") on all-NaN columns, but the
        # np.where below handles that case explicitly by falling back to 0.0; suppress the noise to
        # keep test output clean.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            col_medians = np.nanmedian(arr, axis=0)
        # Empty / all-NaN columns: median is NaN; fall back to 0.0 for the
        # discretize edges (the column will be all-NaN-bin anyway).
        col_medians = np.where(np.isnan(col_medians), 0.0, col_medians)
        if arr.flags.writeable:
            # In-place broadcast-fill at the NaN positions -- avoids the full-size np.where allocation
            # (a third pass over ``arr`` on top of the ``_mask`` computation and the nanmedian scan).
            arr[_mask] = np.broadcast_to(col_medians, arr.shape)[_mask]
            return arr
        return np.where(_mask, col_medians, arr)
    if strategy == "raise":
        raise ValueError("input contains NaN values; pass strategy='fillna_zero' or 'separate_bin' or 'propagate' to discretize anyway")
    raise ValueError(f"unknown missing-value strategy: {strategy!r}")


# =============================================================================
# Polars LazyFrame autocollect
# =============================================================================


def _maybe_collect_lazy(df):
    """If ``df`` is a polars LazyFrame, materialise it; other inputs pass through. ``.collect_streaming()`` is intentionally not used -- if the caller wanted
    streaming, they should pass ``MRMR`` a frame that fits in memory."""
    try:
        import polars as pl
    except ImportError:
        return df
    if isinstance(df, pl.LazyFrame):
        logger.warning("MRMR autocollecting LazyFrame at boundary. Pass a materialised DataFrame to skip this copy.")
        return df.collect()
    return df


# =============================================================================
# 1-D categorisation helpers (legacy `categorize_1d_array` retained)
# =============================================================================


def categorize_1d_array(
    vals: np.ndarray,
    min_ncats: int,
    method: str,
    astropy_sample_size: int,
    method_kwargs: dict,
    dtype=np.int16,
    nan_filler: float = 0.0,
):
    """Per-column ordinal encoder used by ad-hoc external pipelines. Inside MRMR proper we use ``categorize_dataset`` below.

    ``astropy_sample_size`` is unused: the astropy-backed binning method it configured was removed (see the
    ``strategy`` removal note below for the sibling 'uniform'/'kmeans' cleanup); kept in the signature only
    for back-compat with external callers passing it positionally or as a keyword.

    Wave 50 (2026-05-20): ``nan_filler=0.0`` default mixes NaN rows with real-0 rows
    into bin-0, biasing MI estimation. New callers should pass ``nan_filler=None``
    to raise honestly on NaN input, or use a sentinel that cannot collide with real
    data (``np.nan_to_num(vals, nan=vals.min()-1)`` upstream). Default kept as 0.0
    for back-compat -- a WARN is emitted when NaNs are actually filled.
    """
    # 2026-05-28: drop sklearn OrdinalEncoder; the legacy code path created a
    # NEW estimator on every call (and never reused it), so the only contract
    # consumed was fit_transform's ordinal-encoding behaviour. The native
    # ``_native_ordinal_encode_2d`` shim above gives identical output bit-for-bit
    # at ~6x lower wall-clock.
    if vals.dtype.name != "category" and np.issubdtype(vals.dtype, np.bool_):
        vals = vals.astype(np.int8)

    if pd.isna(vals).any():
        # Wave 50: surface the legacy bias when it actually fires.
        if nan_filler is None:
            raise ValueError("categorize_1d_array: input contains NaN and nan_filler=None; " "drop NaN upstream or pick a non-colliding sentinel.")
        import warnings as _w
        _w.warn(
            f"categorize_1d_array: filling NaN with {nan_filler!r} biases MI by mixing "
            "NaN rows with real-equal values. Pass nan_filler=None to raise instead.",
            stacklevel=2,
        )
        vals = pd.Series(vals).fillna(nan_filler).values

    vals = vals.reshape(-1, 1)

    if vals.dtype.name != "category":
        nuniques = len(np.unique(vals[: min_ncats * 10]))
        if nuniques <= min_ncats:
            nuniques = len(np.unique(vals))
    else:
        nuniques = min_ncats

    if method == "discretizer":
        bins = method_kwargs.get("n_bins")
    else:
        bins = method_kwargs.get("bins")

    if vals.dtype.name != "category" and nuniques > min_ncats:
        if method == "discretizer":
            if bins is None:
                raise ValueError("categorize_1d_array: method='discretizer' requires method_kwargs['n_bins'].")
            if nuniques > bins:
                # 2026-05-28: native pure-numpy quantile binning (replaces sklearn KBinsDiscretizer).
                # Bit-for-bit identical output (np.nanpercentile + np.searchsorted) at ~12x lower wall-clock.
                _strategy = method_kwargs.get("strategy", "quantile")
                if _strategy != "quantile":
                    raise NotImplementedError(
                        f"categorize_1d_array: strategy={_strategy!r} no longer supported. "
                        f"Native path implements 'quantile' only; the previous sklearn-backed "
                        f"'uniform' / 'kmeans' modes were dead code in MRMR (hot path uses "
                        f"discretize_2d_array directly). Pass strategy='quantile' or switch upstream."
                    )
                new_vals = _native_kbins_quantile(vals, n_bins=int(bins))
            else:
                new_vals = _native_ordinal_encode_2d(vals)
        else:
            if method == "numpy":
                bin_edges = np.histogram_bin_edges(vals, bins=bins if bins is not None else "auto")
            elif method == "astropy":
                # 2026-05-28: astropy removed from the install graph. The legacy 'astropy'
                # method used Bayesian-blocks / Knuth-rule binning; both have native numba
                # implementations elsewhere in the project (see filters/supervised_binning.py).
                # Until callers migrate, downgrade to numpy's histogram_bin_edges with bin
                # count derived from the legacy 'bins' arg.
                _bins_for_numpy = bins if isinstance(bins, (int, np.integer)) else "auto"
                bin_edges = np.histogram_bin_edges(vals, bins=_bins_for_numpy)
                logger.info(
                    "categorize_1d_array: method='astropy' is deprecated; "
                    "using numpy histogram_bin_edges(bins=%r). astropy removed from install graph 2026-05-28.",
                    _bins_for_numpy,
                )
            else:
                # Wave 55 (2026-05-20): pre-fix, an unknown method (typo / "quantile" / "kmeans")
                # left bin_edges undefined and the next line raised UnboundLocalError. Raise
                # honestly with the offender so callers see a typed contract failure.
                raise ValueError(f"categorize_1d_array: unknown method={method!r}; expected one of " "'discretizer', 'numpy', 'astropy'.")

            if bin_edges[0] <= vals.min():
                bin_edges = bin_edges[1:]

            new_vals = _native_ordinal_encode_2d(np.digitize(vals, bins=bin_edges, right=True))
    else:
        new_vals = _native_ordinal_encode_2d(vals)

    # Wave 40 (2026-05-20): auto-promote dtype to avoid silent wraparound on
    # high-cardinality columns; matches categorize_dataset's promotion ladder.
    out = new_vals.ravel()
    out_max = int(out.max()) if out.size else 0
    if out_max > np.iinfo(dtype).max:
        for _candidate in (np.int16, np.int32, np.int64):
            if out_max <= np.iinfo(_candidate).max:
                logger.warning(
                    "categorize_1d_array: max code %d exceeds dtype %s; auto-promoting to %s to avoid silent wraparound.",
                    out_max, dtype, _candidate,
                )
                dtype = _candidate
                break
        else:
            raise ValueError(f"categorize_1d_array: cardinality {out_max} exceeds int64 max; cannot encode.")
    return out.astype(dtype)


# =============================================================================
# Low-level numba kernels (pure functions; no module-level side-effects)
# =============================================================================


@njit(cache=True)
def digitize(arr: np.ndarray, bins: np.ndarray, dtype=np.int32) -> np.ndarray:
    """Linear-scan bin-code assignment: the index of the first ``bin_edge >= val``, clamped to the top bin above the last edge."""
    # Values above the last edge (e.g. a transform-time row exceeding the
    # fit-time max) must clamp to the top bin. Pre-fix the inner break never
    # fired for ``val > bins[-1]``, leaving ``res[i]`` at its uninitialised
    # ``np.empty`` value -> garbage codes (run-to-run nondeterministic). The
    # sibling ``quantize_search`` correctly routes out-of-range high values
    # to the top bin; match that contract here.
    n_bins = len(bins)
    last = n_bins - 1
    res = np.empty(len(arr), dtype=dtype)
    for i, val in enumerate(arr):
        assigned = False
        for j, bin_edge in enumerate(bins):
            if val <= bin_edge:
                res[i] = j
                assigned = True
                break
        if not assigned:
            res[i] = last
    return res


@njit(cache=True)
def quantize_dig(arr, bins):
    """Bin-code assignment via ``np.digitize`` on the interior bin edges (excludes the outer -inf/+inf sentinels)."""
    return np.digitize(arr, bins[1:-1], right=True)


@njit(cache=True)
def quantize_search(arr, bins):
    """Bin-code assignment via ``np.searchsorted`` on the interior bin edges -- faster than ``quantize_dig``
    for sorted, dense inputs. NOT bit-equivalent on-edge (DISCRETIZATION-7 fix):
    ``digitize(..., right=True)`` and ``searchsorted(..., side="right")`` disagree at an exact edge value
    (digitize's right=True excludes the left boundary of each bin; searchsorted's side="right" includes it),
    so a value sitting exactly on a bin edge can land in a different bin between the two. This is the
    function actually used everywhere live; ``quantize_dig`` has zero non-warmup callers."""
    return np.searchsorted(bins[1:-1], arr, side="right")


@njit(cache=True)
def discretize_uniform(arr: np.ndarray, n_bins: int, min_value: Optional[float] = None, max_value: Optional[float] = None, dtype: type = np.int8) -> np.ndarray:
    """Equal-width binning of ``arr`` into ``n_bins`` over ``[min_value, max_value]`` (or the array's own range when unset)."""
    # 2026-05-30 Wave 9.1 fix (loop iter 33): the divisor was
    # ``(max - min + min/2)`` instead of the canonical ``(max - min)``.
    # That formula silently miscoded any positive-shifted input -
    # ``linspace(1000, 1100)`` into 10 bins collapsed to just bins
    # {0: 600, 1: 400} instead of 10 evenly populated bins. On purely
    # negative ranges the divisor went to zero (div-by-zero RuntimeWarning,
    # everything -> bin 0) or even negative (sign flip). The bug poisoned
    # every downstream MI / SU / MRMR score whenever ``method="uniform"``
    # was used on prices / distances / counts / epoch timestamps / any
    # mean-nonzero feature. Sibling CUDA path at discretization.py:850
    # had the same defect by design (per the now-obsolete bit-comparability
    # comment) - fixed together.
    if min_value is None or max_value is None:
        min_value, max_value = arrayMinMax(arr)
    n = arr.shape[0]
    out: np.ndarray = np.empty(n, dtype=dtype)
    # Dedicated NaN bin code (one past the top real code n_bins-1), matching the
    # categorize_dataset / quantile-path convention so NaN rows do NOT collide with real
    # bin 0 and never produce a garbage cast. ``np.clip(NaN, ...)`` is a no-op on NaN, so
    # without this gate the subsequent ``.astype(int8)`` casts NaN to a garbage code
    # (RuntimeWarning: invalid value encountered in cast) and silently poisons MI/SU/MRMR.
    nan_code = n_bins
    _rng = max_value - min_value
    if not (_rng > 0):
        # Constant column OR all-NaN min/max (max-min is NaN -> _rng>0 is False): real rows
        # -> bin 0 (honest single-bin), NaN rows -> the dedicated NaN bin.
        for i in range(n):
            v = arr[i]
            out[i] = nan_code if v != v else 0
        return out
    rev_bin_width = n_bins / _rng
    hi = n_bins - 1
    for i in range(n):
        v = arr[i]
        if v != v:  # NaN: route to the dedicated bin, never the affine map / cast
            out[i] = nan_code
            continue
        # Affine map then clip in the float domain BEFORE the (possibly narrow) cast: casting
        # first lets codes > dtype-max wrap negative (int8 modulo), and a later clip then maps
        # those wrapped negatives to bin 0 -- silently collapsing the high-value region.
        # A denormal-tiny range makes rev_bin_width overflow to inf, so (v-min)*inf is NaN at v==min
        # (0*inf). NaN passes both ``< 0`` and ``> hi`` (comparisons are False on NaN) and would cast to a
        # garbage code -- route it to bin 0 (the value sits at the column floor) alongside the negatives.
        c = (v - min_value) * rev_bin_width
        if c != c or c < 0:
            c = 0.0
        elif c > hi:
            c = float(hi)
        out[i] = c
    return out


@njit(cache=True, parallel=True)
def discretize_uniform_parallel(arr: np.ndarray, n_bins: int, min_value: float, max_value: float, dtype: type = np.int8) -> np.ndarray:
    """Column-prange twin of ``discretize_uniform`` for large single-column arrays.

    Byte-identical to ``discretize_uniform`` (same affine map + clip-before-cast in float domain), only the elementwise
    clip+cast is split across cores via ``prange``. 17.9x at n=10M / 47.9x at n=1M over the serial scan, bit-identical
    (the map is per-element independent, no reduction). The Python ``discretize_array`` uniform path size-gates to this
    above ``_UNIFORM_PAR_THRESHOLD``; below it the serial kernel wins (prange spawn overhead). ``min_value``/``max_value``
    are required here (the size-gating caller has already resolved them, avoiding a separate serial ``arrayMinMax`` scan).
    """
    n = arr.shape[0]
    out: np.ndarray = np.empty(n, dtype=dtype)
    # Dedicated NaN bin (n_bins, one past the top real code) -- mirrors discretize_uniform so NaN rows
    # do not collide with real bin 0 and never hit the NaN-producing affine map + cast.
    nan_code = n_bins
    rng = max_value - min_value
    if not (rng > 0):
        for i in prange(n):
            v = arr[i]
            out[i] = nan_code if v != v else 0
        return out
    rev_bin_width = n_bins / rng
    hi = n_bins - 1
    for i in prange(n):
        x = arr[i]
        if x != x:  # NaN: dedicated bin, skip the affine map / cast
            out[i] = nan_code
            continue
        v = (x - min_value) * rev_bin_width
        if v != v or v < 0:  # NaN from inf*0 on a denormal-tiny range -> column floor, not a garbage cast
            v = 0.0
        elif v > hi:
            v = float(hi)
        out[i] = v
    return out


# Crossover (measured 2026-06-15, n=10M float64): serial wins <~50k (prange spawn dominates), parallel wins above
# (2.2x @100k -> 47.9x @1M). Override via MLFRAME_DISCRETIZE_UNIFORM_PAR_THRESHOLD for non-dev hardware.
_UNIFORM_PAR_THRESHOLD = int(os.environ.get("MLFRAME_DISCRETIZE_UNIFORM_PAR_THRESHOLD", "50000"))


def discretize_array(
    arr: np.ndarray, n_bins: int = 10, method: str = "quantile",
    min_value: Optional[float] = None, max_value: Optional[float] = None, dtype: type = np.int8,
) -> np.ndarray:
    """Discretise a 1-D continuous array into ordinal bins.

    Single-column path uses raw numpy instead of dispatching to the ``@njit`` ``_discretize_array_impl``. Microbench at n=10000: njit ``np.percentile`` ~870us
    vs direct ``np.percentile`` ~405us (numba is ~2x slower than numpy at this size for percentile work). The FE pipeline calls this 6000+ times per fit on
    n=10000, p=200 -- the un-njit path saves ~3s. Multi-column ``discretize_2d_array`` keeps the njit chain because it parallelises columns via ``prange``.
    """
    if method not in ("uniform", "quantile"):
        raise ValueError(f"Unsupported discretization method: '{method}'. Supported methods: 'uniform', 'quantile'")
    dtype = _safe_code_dtype(n_bins, dtype)  # widen so n_bins>128 codes don't wrap negative
    arr = np.asarray(arr)
    if arr.size == 0:
        # Empty input: the uniform sibling already returns an empty array; the
        # quantile path used to raise an opaque IndexError from
        # ``np.nanpercentile([])`` slicing. Return an empty result for both so
        # siblings agree on the degenerate-input contract.
        return np.empty(0, dtype=dtype)
    if method == "uniform":
        # Size-gate the elementwise affine+clip+cast: the serial njit scan is single-threaded, but it is a real O(n)
        # cost that a column-prange twin parallelises bit-identically (17.9x @10M). Below the crossover the serial
        # kernel wins (prange spawn overhead), so only large arrays route to the parallel twin.
        if arr.shape[0] >= _UNIFORM_PAR_THRESHOLD:
            if min_value is None or max_value is None:
                min_value, max_value = arrayMinMax(arr)
            return np.asarray(discretize_uniform_parallel(arr, n_bins, float(min_value), float(max_value), dtype=dtype))
        return np.asarray(discretize_uniform(arr=arr, n_bins=n_bins, min_value=min_value, max_value=max_value, dtype=dtype))
    # quantile path -- raw numpy.
    # Wave 21 P0: nanpercentile so NaN-bearing columns don't collapse to a
    # constant via the all-NaN bin_edges trap. Same finding as the ``edges``
    # helper above.
    quantiles = np.linspace(0, 100, n_bins + 1)
    # bench-attempt-rejected (2026-06-14): routing the NaN-free 1-D case to np.percentile (skipping
    # nanpercentile's nan-mask) to mirror the 2-D batch win. Microbench at n=100k float32 x 2000 calls:
    # 9.74s vs 9.68s nanpercentile -- a WASH (the O(n) np.isfinite(arr).all() guard offsets the saved
    # nan-mask on the 1-D path; the 2-D win came from amortising dispatch over many columns, absent here).
    # Kept nanpercentile: equal speed, simpler, and NaN-correct without a guard.
    #
    # bench-attempt-rejected (2026-06-24): a FUSED 1-D njit kernel ``_quantile_codes_1d_njit``
    # (kept in _kernels.py for reuse/re-bench on other HW) that partitions the column ONCE at the
    # lerp's read indices (the same one-partition technique that WON in the 2-D ``_quantile_edges_2d_njit``)
    # and assigns codes via an inline binary search, replacing BOTH ``nanpercentile`` and ``searchsorted``.
    # BIT-IDENTICAL on a NaN-free array (verified across float32/64 x {2..300} bins x uniform/ties/heavy/const).
    # But it is a NET LOSS for the SINGLE-COLUMN 1-D case: microbench (NaN-free float32, 200 reps)
    #   n=10k/nb=10 0.74x, n=10k/nb=20 0.70x, n=100k/nb=10 0.67x, n=100k/nb=20 0.58x.
    # The 2-D batch win came from amortising dispatch over MANY columns + a column ``prange``; a single
    # serial 1-D column has neither, and numpy's vectorised C ``partition`` beats a numba scalar
    # partition-copy + python-free-but-scalar binary-search loop. Confirms the older 2026-06-14
    # np.percentile-swap rejection: the 1-D quantile path is numpy-optimal as-is. Reverted to numpy.
    bins_edges = np.nanpercentile(arr, quantiles)
    return np.searchsorted(bins_edges[1:-1], arr, side="right").astype(dtype)


# The 2-D quantile-edge + searchsorted njit kernels were carved into the sibling
# _kernels.py (LOC-budget sibling re-export). They are imported here so
# discretize_2d_quantile_batch below keeps calling them, and re-exported at the
# bottom of this module so the package public/private import surface is unchanged
# (e.g. from ...discretization import _quantile_edges_2d_njit still resolves).
from ._kernels import (
    _quantile_edges_2d_njit,
    _searchsorted_2d_right_njit,
    _searchsorted_2d_right_njit_parallel,
)


def discretize_2d_quantile_batch(arr2d: np.ndarray, n_bins: int = 10, dtype: type = np.int8, parallel: bool = False, assume_finite: bool = False) -> np.ndarray:
    """Batch (quantile-only) discretiser: bit-identical to per-column ``discretize_array(method='quantile')``.

    ``arr2d`` is ``(n_rows, n_cols)``; each column is discretised independently into ``n_bins`` ordinal codes
    and returned as a ``(n_rows, n_cols)`` array of ``dtype``.

    Why this is bit-identical to calling ``discretize_array(method='quantile')`` column-by-column (the
    FE-pair-search hot path):
      * Quantile grid: ``np.linspace(0, 100, n_bins+1)`` -- the identical grid the 1-D path uses.
      * Edges: ``np.percentile(arr2d, q, axis=0)`` partitions EACH COLUMN independently with the SAME
        linear-interpolation estimator as the 1-D path, so ``edges[:, j]`` equals the 1-D edge vector exactly.
        The 1-D path uses ``np.nanpercentile``; on a column WITHOUT NaN ``np.percentile == np.nanpercentile``
        bit-for-bit (verified across random + tie-heavy + constant-column frames). The FE caller always passes
        a post-``nan_to_num`` (NaN/inf-free) buffer, so the fast ``np.percentile`` path is exact there. If ANY
        NaN is present we fall back to ``np.nanpercentile(axis=0)`` (still bit-identical to the per-column 1-D
        ``nanpercentile``, just slower) -- so the helper stays correct for any caller.
      * Codes: ``np.searchsorted(edges[1:-1, j], arr2d[:, j], side='right')`` is the identical call the 1-D path
        makes; only the edges argument is sliced out of the 2-D edge matrix (same float64 values).
      * dtype: NO cast is applied to ``arr2d`` -- it is consumed at its native dtype (the FE buffer is float32).
        ``np.percentile``/``np.nanpercentile`` upcast internally to float64 for a float32 input regardless of
        1-D vs 2-D, so the float64 edges match; ``searchsorted(float64_edges, float32_col)`` is value-identical.

    Performance note (2026-06-04): ``np.nanpercentile(axis=0)`` routes through ``apply_along_axis`` (a
    Python-level per-column loop) and gives NO dispatch amortisation -- it is as slow as (or slower than) the
    per-column loop. ``np.percentile(axis=0)`` uses the fully vectorised C partition path (no per-column python
    loop) and is ~3-13x faster on the FE buffer shapes (e.g. n=400 x 300 cols: 3.8ms vs 48ms loop). That fast
    vectorised partition -- only legal because the FE buffer is NaN-free -- is the actual win; the NaN guard keeps
    it bit-identical for the general case. ``searchsorted`` stays per-column (each column's sliced edge vector
    differs); that loop is cheap relative to the eliminated per-column ``percentile`` + ``linspace`` dispatch.
    """
    dtype = _safe_code_dtype(n_bins, dtype)  # widen so n_bins>128 codes don't wrap negative
    n_rows, n_cols = arr2d.shape
    quantiles = np.linspace(0, 100, n_bins + 1)
    # bench-attempt-rejected (2026-06-07): FUSING searchsorted INTO the quantile sort
    # (Q1, top ceiling on the ~21% discretize hotspot). The only fusion that reuses the
    # per-column sort is ARGSORT (sorted->orig index) + an edge-pointer walk that SCATTERS
    # codes to ``out[orig]``. That scatter is random-access (cache-hostile) and replaces the
    # current cache-friendly column-strided binary search. Isolated bench on scene FE buffer
    # shapes (parallel kernels): 300 cols/f32 1.27x (only win, tiny K), 300/f64 0.78x,
    # 1000 cols 0.49x, 4000 cols 0.04x (28x REGRESSION at the K=4000-8000 the FE buffer
    # actually uses -- the scatter blows the LLC). The existing 2-pass (parallel edges-sort
    # + parallel searchsorted, the latter only 26-34% of full) is already optimal; do not
    # re-attempt argsort-fusion. (proto D:/Temp/q1_fused_proto.py)
    # Fast path: NaN-free buffer -> ``_quantile_edges_2d_njit`` (sort each column ONCE,
    # read all quantiles) is BIT-IDENTICAL to ``np.percentile(axis=0, method='linear')``
    # (verified across float32/64, ties, constant columns) and removes the numpy
    # ``ndarray.partition`` hotspot -- the FE sweep's single dominant numpy cost
    # (scene 1500x299 cProfile: 114.5s / 20% of fit in ``partition``, re-partitioning the
    # whole buffer once per quantile). NaN path keeps ``np.nanpercentile`` (the njit kernel
    # does not NaN-handle; callers that pass NaN are rare and stay correct + bit-identical).
    # ``assume_finite``: the caller already scrubbed NaN/inf out of ``arr2d`` (e.g. the FE-chunk
    # path does ``np.nan_to_num(buf, copy=False)`` on the line immediately before this call), so the
    # per-call ``np.isnan(arr2d).any()`` scan -- a full O(n_rows*n_cols) pass plus a full bool-array
    # allocation, run on every one of 1000+ discretise calls in a wide FE sweep -- is pure wasted work
    # whose result is always False. Skipping it goes straight to the fast ``_quantile_edges_2d_njit``
    # branch; BIT-IDENTICAL by construction on a NaN-free buffer (the scan would pick exactly that branch).
    # Leave the default False so any caller that has NOT scrubbed keeps the safe NaN-aware path.
    if not assume_finite and np.isnan(arr2d).any():
        edges = np.nanpercentile(arr2d, quantiles, axis=0)
    else:
        edges = np.empty((quantiles.shape[0], n_cols), dtype=np.float64)
        if n_cols > 0 and n_rows > 0:
            # Order-statistic indices the kernel's lerp actually reads (lo / lo+1 per
            # quantile, with the n-1 endpoint clamp). Partitioning at exactly these is O(n)
            # vs the full O(n log n) sort and bit-identical at the read indices.
            _vidx = (quantiles / 100.0) * (n_rows - 1)
            _lo = np.floor(_vidx).astype(np.int64)
            _kths_set = set()
            for _l in _lo.tolist():
                if _l >= n_rows - 1:
                    _kths_set.add(n_rows - 1)
                else:
                    _kths_set.add(int(_l)); _kths_set.add(int(_l) + 1)
            _kths = np.array(sorted(_kths_set), dtype=np.int64)
            _quantile_edges_2d_njit(np.ascontiguousarray(arr2d), quantiles, _kths, edges)
    out: np.ndarray = np.empty((n_rows, n_cols), dtype=dtype)
    # njit per-column searchsorted (bit-identical to the numpy loop, incl. NaN
    # -> rightmost bin; see ``_searchsorted_2d_right_njit``). ``edges`` is float64 from
    # percentile and is small ((n_bins-1) x n_cols) so we make it C-contiguous float64.
    # ``arr2d`` is passed at its NATIVE dtype (do NOT upcast to float64): the FE buffer is
    # float32 and full-width (n_rows x n_cols, often >1e8 cells on a wide pool), so a
    # ``np.ascontiguousarray(arr2d, dtype=np.float64)`` would DOUBLE it into a multi-GB
    # float64 copy and OOM -- the regression that crashed MRMR.fit on the wide canonical
    # fixture (20000 x ~19000 cols -> 2.9 GiB float64 alloc). The kernel's per-element
    # ``arr2d[r,j] < edges_inner[mid,j]`` promotes a float32 value to float64 against the
    # float64 edge in numba, EXACTLY matching numpy's ``searchsorted(float64_edges,
    # float32_col)`` (which finds the float64 common dtype and compares there) -- so the
    # codes are byte-identical to both the float64-copy path and the per-column 1-D path.
    # ``np.ascontiguousarray(arr2d)`` (no dtype) is a no-op view when arr2d is already
    # C-contiguous (the common FE case incl. column-slices ``buf[:, :k]``); a genuinely
    # non-contiguous input copies at the native float32 width, still half the float64 cost.
    # bench-attempt-rejected (2026-06-07): F-CONTIGUOUS buffers (Q4) so the kernels'
    # ``for j(cols): for r(rows): arr2d[r,j]`` inner r-loop is unit-stride instead of
    # column-strided. Bit-identical (layout-only) but a NET LOSS at the real FE width:
    # ``np.asfortranarray(arr2d)`` copies the whole (n x K) buffer, and that copy costs
    # more than the cache win once K is large. Bench (scene FE shapes, copy included):
    #   300 cols/f32 1.30-1.46x (only win, tiny K); 1000 cols 0.76-0.91x; 4000 cols 0.80-0.88x.
    # The edges kernel already copies each column into a contiguous scratch (so it gains
    # nothing from F-layout); only searchsorted reads strided, and enabling that via a
    # full-buffer transpose-copy regresses. Keep C-contiguous. (proto D:/Temp/q4_fcontig_proto.py)
    if n_cols > 0 and n_rows > 0:
        edges_inner = np.ascontiguousarray(edges[1:-1], dtype=np.float64)
        arr_c = np.ascontiguousarray(arr2d)
        # OPT-A (2026-06-07): on the SERIAL-MAIN-THREAD FE path the searchsorted kernel
        # runs single-threaded on one core while the rest sit idle (the serial kernel is
        # ``nogil`` not ``parallel`` to avoid deadlocking the joblib threading layer on the
        # >=50000-row path -- but that path is NOT active here). ``parallel=True`` selects the
        # byte-identical column-prange twin so the per-column binary searches spread across
        # cores. The caller (check_prospective_fe_pairs) passes ``parallel=True`` ONLY when it
        # knows it is on the main-thread/no-joblib branch (threaded down from _mrmr_fe_step's
        # ``len(X) < 50000`` dispatch); the joblib path keeps the serial kernel (parallel=False).
        if parallel:
            _searchsorted_2d_right_njit_parallel(edges_inner, arr_c, out)
        else:
            _searchsorted_2d_right_njit(edges_inner, arr_c, out)
    return out


@njit(cache=True)
def _discretize_array_impl(
    arr: np.ndarray, n_bins: int = 10, method: str = "quantile",
    min_value: Optional[float] = None, max_value: Optional[float] = None, dtype: type = np.int8,
) -> np.ndarray:
    """Discretize a single 1-D column via ``uniform`` or ``quantile`` binning."""
    if method == "uniform":
        return np.asarray(discretize_uniform(arr=arr, n_bins=n_bins, min_value=min_value, max_value=max_value, dtype=dtype))
    elif method == "quantile":
        bins_edges = get_binning_edges(arr=arr, n_bins=n_bins, method=method, min_value=min_value, max_value=max_value)
    return np.asarray(quantize_search(arr, bins_edges).astype(dtype))


# cache=True persists the parallel-fused artefact alongside the serial @njit kernels above.
# Pre-fix iter-366: the only cache=False kernel in this module re-paid ~7.9s LLVM compile
# (18% of a 43.5s 1M cb+MRMR train) on every fresh process. Caching reduces second-run
# fit time by the full compile budget; the parallel=True specialisation caches per CPU
# arch the same way the serial variants already did.
@njit(parallel=True, cache=True)
def _discretize_2d_array_njit(
    arr: np.ndarray,
    n_bins: int = 10,
    method: str = "quantile",
    min_ncats: int = 50,
    min_values: Optional[np.ndarray] = None,
    max_values: Optional[np.ndarray] = None,
    dtype: type = np.int8,
) -> np.ndarray:
    """CPU prange backend; one column per worker thread."""
    res: np.ndarray = np.empty_like(arr, dtype=dtype)
    for col in prange(arr.shape[1]):
        res[:, col] = _discretize_array_impl(
            arr=arr[:, col],
            n_bins=n_bins,
            method=method,
            min_value=min_values[col] if min_values is not None else None,
            max_value=max_values[col] if max_values is not None else None,
            dtype=dtype,
        )
    return res


# Size threshold for CUDA dispatch: below this the per-launch CUDA overhead
# (~50 ms H2D + first-call kernel JIT amortised across the session) dominates
# the prange wall. Measured on GTX 1050 Ti: at n_rows * n_cols = 500_000 the
# CUDA path is ~5x faster than warm CPU prange; below 100_000 cells CPU wins.
_DISCRETIZE_2D_CUDA_MIN_CELLS = 500_000
# Spans the ~500k crossover; capped at 2M (the catch-all extrapolates beyond) to
# keep the one-time cold-start sweep cheap -- discretize is a hot dispatch.
_DISCRETIZE_SWEEP_CELLS = [50_000, 200_000, 500_000, 2_000_000]
_DISCRETIZE_SALT = 1


def _make_discretize_inputs(dims: dict):
    """A 2-D float64 frame of ~``n_cells`` cells (fixed 8 columns) -- the operand
    discretize_2d_array bins per column."""
    rng = np.random.default_rng(0)
    cols = 8
    rows = max(1, int(dims["n_cells"]) // cols)
    return (rng.standard_normal((rows, cols)).astype(np.float64),)


def _run_discretize_sweep() -> list:
    """Full n_cells grid sweep -> backend_choice regions: cpu (njit prange) vs cuda
    (cupy), fastest EQUIVALENT per band. quantile method, no min/max (the cuda path
    computes its own percentiles). Inputs host-resident -> no residency axis. The
    cupy + cpu percentile binning use the same algorithm, so the int8 bins match."""
    from pyutilz.core.pythonlib import is_cuda_available
    from pyutilz.dev.benchmarking import sweep_backend_grid

    def _cpu(arr):
        """CPU-path timing: the njit-prange 2D discretize kernel."""
        return _discretize_2d_array_njit(
            arr=arr, n_bins=10, method="quantile", min_ncats=50,
            min_values=None, max_values=None, dtype=np.int8,
        )

    variants = {"cpu": _cpu}
    if is_cuda_available():
        def _cuda(arr):
            """GPU-path timing: the cupy percentile-binning discretize kernel, for crossover comparison against ``_cpu``."""
            return discretize_2d_array_cuda(arr=arr, n_bins=10, method="quantile", dtype=np.int8)
        variants["cuda"] = _cuda
    return list(sweep_backend_grid(
        variants, {"n_cells": _DISCRETIZE_SWEEP_CELLS}, _make_discretize_inputs,
        reference="cpu", repeats=2, equiv_rtol=1e-6, equiv_atol=1e-6,
    ))


def _discretize_fallback_choice(n_cells: int) -> str:
    """Pre-sweep heuristic (the spec's dynamic fallback callable): cuda above the
    hand-tuned 500k-cell breakeven, else cpu."""
    return "cuda" if int(n_cells) >= _DISCRETIZE_2D_CUDA_MIN_CELLS else "cpu"


def discretize_2d_array(
    arr: np.ndarray,
    n_bins: int = 10,
    method: str = "quantile",
    min_ncats: int = 50,
    min_values: Optional[np.ndarray] = None,
    max_values: Optional[np.ndarray] = None,
    dtype: type = np.int8,
    prefer_gpu: bool = True,
) -> np.ndarray:
    """Discretise every column of a 2-D continuous array into ordinal bins.

    Dispatcher that picks the fastest backend per call:

    * **CUDA / CuPy** (``discretize_2d_array_cuda``) -- wins at ``n_rows *
      n_cols >= 500_000`` when CUDA is available AND ``method="quantile"``
      AND ``min_values is None`` AND ``max_values is None`` (the GPU path
      computes its own per-column percentiles via ``cp.percentile``).
    * **CPU prange** (``_discretize_2d_array_njit``) -- the fallback;
      always available, optimal at small frames.

    Use ``prefer_gpu=False`` to force the CPU prange path -- the tests
    that compare GPU-vs-CPU walls rely on this knob (mirrors the
    ``mi_direct(..., prefer_gpu=False)`` API added in commit 7319f11).

    Per ``feedback_fastest_default_with_dispatch``: the public name
    routes to the fastest backend by default; manual backend selection
    is only for tests + benches.
    """
    dtype = _safe_code_dtype(n_bins, dtype)  # widen so n_bins>128 codes don't wrap negative
    # CUDA-eligibility: per-host backend_choice (cpu/cuda) from the kernel tuning
    # cache via get_or_tune (the hand-tuned 500k breakeven is the fallback), so the
    # dispatcher adapts to faster GPUs without code edits. _discretize_backend_choice
    # uses the module-singleton cache + is lru_cached, so it does NOT re-trigger
    # _build_provenance (nvidia-smi ~48ms) per call.
    #
    # 2026-05-28: uniform method gained a CUDA path (single-pass vectorised
    # arithmetic + RawKernel searchsorted). Both methods route to GPU when
    # min_values/max_values are absent (the CUDA uniform path computes col_min/max).
    if (
        prefer_gpu
        and method in ("quantile", "uniform")
        and min_values is None
        and max_values is None
        and arr.ndim == 2
        and _DISCRETIZE_SPEC.choose(n_cells=int(arr.size)) == "cuda"
    ):
        try:
            from pyutilz.core.pythonlib import is_cuda_available
            if is_cuda_available():
                # VRAM guard (2026-07-10): ``discretize_2d_array_cuda`` H2D-uploads the WHOLE ``arr``
                # unconditionally (``d_arr = cp.asarray(arr)``), then ``cp.percentile`` needs a
                # comparably-sized internal sort/partition scratch buffer on top -- at production scale
                # (millions of rows) this can consume a small card's entire VRAM. On Windows/WDDM an
                # oversized upload can transparently over-subscribe device memory via host-paging instead
                # of raising a catchable CUDA OOM, so the kernel then grinds through PCIe-paged memory for
                # minutes before the OS silently kills the process -- the try/except below never fires
                # because there is no exception to catch. Reject BEFORE attempting the upload, mirroring
                # every other GPU-FE dispatch site's ``_fe_gpu_vram.fe_gpu_has_vram_cushion`` guard.
                _bytes_needed = arr.nbytes * 2 + (arr.shape[0] * arr.shape[1] * np.dtype(dtype).itemsize)
                _vram_ok = True
                _free_gb = _total_gb = None
                _free_b: Optional[int] = None
                try:
                    import cupy as _cp_probe
                    _free_b, _total_b = _cp_probe.cuda.runtime.memGetInfo()
                    _free_gb, _total_gb = _free_b / 1024**3, _total_b / 1024**3
                except Exception as exc:
                    logger.debug("discretize_2d_array: memGetInfo probe failed (%s)", exc)
                try:
                    from mlframe.feature_selection.filters._fe_gpu_vram import fe_gpu_has_vram_cushion
                    _vram_ok = fe_gpu_has_vram_cushion(_bytes_needed)
                except Exception as exc:
                    logger.debug("discretize_2d_array: VRAM cushion probe failed (%s); permissive", exc)
                if not _vram_ok:
                    # Never silent (explicit user feedback): log the full sizing context -- requested GB,
                    # shape/dtype, and free/total VRAM -- so a production run is diagnosable from the log
                    # alone, not just "GPU skipped" with no numbers.
                    logger.warning(
                        "discretize_2d_array: GPU upload REJECTED -- requested %.2fGB upload+scratch "
                        "(n_rows=%d, n_cols=%d, in_dtype=%s, out_dtype=%s) exceeds the safe VRAM budget "
                        "(free=%s, total=%s) -- trying a row-chunked GPU path before falling back to CPU prange",
                        _bytes_needed / 1024**3, arr.shape[0], arr.shape[1], arr.dtype, np.dtype(dtype),
                        f"{_free_gb:.2f}GB" if _free_gb is not None else "unknown",
                        f"{_total_gb:.2f}GB" if _total_gb is not None else "unknown",
                    )
                    try:
                        # Thread the ALREADY-PROBED free-VRAM value through (memGetInfo is a read-only counter
                        # query -- no GPU state changes between the probe above and here) so the row-chunked
                        # fallback's OWN internal memGetInfo call is skipped: this reject-path used to call it
                        # 3x total (this probe, the cushion check's internal probe, and the row-chunked one)
                        # for one decision. ``None`` (probe failed above) keeps the row-chunked function's own
                        # self-probe unchanged.
                        result = discretize_2d_array_cuda_row_chunked(
                            arr=arr, n_bins=n_bins, method=method, dtype=dtype, free_bytes=_free_b,
                        )
                        logger.info("discretize_2d_array: completed via row-chunked CUDA (GPU speed preserved, VRAM-safe)")
                        return result
                    except Exception as exc:
                        logger.warning(
                            "discretize_2d_array: row-chunked CUDA also failed (%s: %s) -- falling back to CPU prange",
                            type(exc).__name__, exc,
                        )
                if _vram_ok:
                    try:
                        return discretize_2d_array_cuda(
                            arr=arr, n_bins=n_bins, method=method, dtype=dtype,
                        )
                    except Exception as exc:
                        logger.debug(
                            "discretize_2d_array: CUDA fastpath failed (%s: %s); " "falling back to CPU prange",
                            type(exc).__name__,
                            exc,
                        )
        except ImportError:
            pass

    return np.asarray(_discretize_2d_array_njit(
        arr=arr, n_bins=n_bins, method=method, min_ncats=min_ncats,
        min_values=min_values, max_values=max_values, dtype=dtype,
    ))


def discretize_2d_array_cuda(
    arr: np.ndarray,
    n_bins: int = 10,
    method: str = "quantile",
    dtype: type = np.int8,
) -> np.ndarray:
    """CuPy port of :func:`discretize_2d_array` for the quantile method.

    Single-launch ``cp.percentile`` computes all per-column edges at once;
    per-column ``cp.searchsorted`` produces the ordinal bins. Total H2D +
    compute + D2H on a 1M-row x 30-col frame runs in ~50 ms (vs ~880 ms
    for the CPU prange path on the same workload at fit-time on a
    GTX 1050 Ti / cc 6.1).

    Returns:
        ``np.ndarray`` of shape ``arr.shape`` with the requested ``dtype``.
        ``copy_to_host`` happens at the end -- callers see plain numpy.

    Raises:
        RuntimeError: if CuPy is not installed or CUDA is not available.
        NotImplementedError: for ``method`` other than ``"quantile"``.

    The function does NOT replace :func:`discretize_2d_array`; both stay
    available. A future dispatch path (``discretize_2d_array_dispatch``)
    can route by ``(n_rows, n_cols)`` and CUDA availability, mirroring
    the ``dispatch_batch_pair_mi`` pattern in ``batch_pair_mi_gpu``.
    """
    try:
        import cupy as cp
    except ImportError as exc:
        raise RuntimeError("cupy not installed; discretize_2d_array_cuda unavailable") from exc

    try:
        from pyutilz.core.pythonlib import is_cuda_available
        if not is_cuda_available():
            raise RuntimeError("CUDA not available on this host")
    except ImportError:
        pass  # fall through; cupy import succeeded so CUDA is likely there

    if method not in ("quantile", "uniform"):
        raise NotImplementedError(
            f"discretize_2d_array_cuda implements 'quantile' / 'uniform'; got method={method!r}",
        )

    if arr.ndim != 2:
        raise ValueError(f"expected 2-D array; got shape {arr.shape}")

    n_rows, n_cols = arr.shape
    if n_rows == 0 or n_cols == 0:
        return np.empty(arr.shape, dtype=dtype)

    # Widen the code dtype to hold ordinal codes 0..n_bins-1 BEFORE allocating the device output -- mirrors
    # the CPU discretize_2d_array (_safe_code_dtype). Without this an int8 request at n_bins>128 wrapped the
    # top bins negative on the GPU (codes 128..n_bins-1 -> negative) while the CPU path widened to int16,
    # a silent cross-backend divergence on the public API. (Verified: NaN routing already matches CPU.)
    dtype = _safe_code_dtype(n_bins, dtype)
    d_arr = cp.asarray(arr)  # H2D once for the whole frame
    # No throwaway GPU allocation just to read a dtype object -- ``np.dtype(dtype)`` produces the identical
    # ``numpy.dtype`` instance cupy's own ``.dtype`` attribute would (cupy dtypes ARE numpy dtypes), so the
    # 1-element ``cp.asarray``/upload this used to pay per call is pure overhead with no different result.
    _out_cp_dtype = cp.int8 if dtype == np.int8 else np.dtype(dtype)
    out = cp.empty((n_rows, n_cols), dtype=_out_cp_dtype)

    if method == "quantile":
        qs = cp.linspace(0.0, 100.0, n_bins + 1)
        # cp.percentile has no nanpercentile twin (unlike numpy), and a plain
        # cp.percentile over a NaN-bearing column poisons EVERY edge for that column with NaN -- searchsorted
        # against an all-NaN edges row then silently collapses the WHOLE column's real values (not just the
        # NaN rows) to a single bin, the exact bug the CPU path's edges()/get_binning_edges() were already
        # fixed for (Wave 21 P0). cupy has no NaN-aware percentile kernel to vectorise this with, so route the
        # rare NaN-bearing case through numpy's nanpercentile on the host array already available in ``arr``
        # (this function's caller has it; ``d_arr`` is just its device upload) -- the common NaN-free case
        # keeps the fully vectorised cp.percentile fast path unchanged.
        if bool(cp.isnan(d_arr).any()):
            bin_edges = cp.asarray(np.nanpercentile(arr, cp.asnumpy(qs), axis=0))
        else:
            # cp.percentile vectorises across axis=0 -> bin_edges shape: (n_bins + 1, n_cols).
            bin_edges = cp.percentile(d_arr, qs, axis=0)
        # cp.searchsorted is 1-D; loop per column. Each call is fully on-device
        # so the loop is dispatch-overhead only (~30 us per launch). For
        # n_cols=30 the total dispatch is ~1 ms vs ~50 ms compute. For
        # n_cols >= 1000 the Python-loop dispatch becomes a wall: route to the
        # fused RawKernel ``discretize_quantile_cuda_rk`` below in that regime.
        if n_cols >= 1000:
            # Per-row col-wise: ravel bin_edges to (n_cols * (n_bins+1)) and do
            # one fused 2D searchsorted via a hand-rolled RawKernel. ~10x
            # speedup vs the per-col Python loop on n_cols=10k.
            cuts = cp.ascontiguousarray(bin_edges[1:-1, :].T)  # (n_cols, n_bins-1)
            out = _discretize_quantile_rawkernel(d_arr, cuts, n_bins, _out_cp_dtype)
        else:
            for j in range(n_cols):
                out[:, j] = cp.searchsorted(bin_edges[1:-1, j], d_arr[:, j], side="right")
    else:
        # method == 'uniform': vectorised arithmetic, no percentile sort,
        # no per-column dispatch. Single GPU pass. Mirrors discretize_uniform
        # njit kernel on CPU. Fastest path for Gaussian-ish data where the
        # accuracy hit vs quantile is small (bench at info_theory module
        # docstring quotes H(X)/log(nbins) >= 0.82 for Gaussian).
        # plain cp.min/cp.max propagate NaN (a single NaN anywhere in a column
        # poisons that column's min/max to NaN), unlike the CPU discretize_uniform path's NaN-aware range.
        # cp.nanmin/cp.nanmax exist (unlike cp.nanpercentile above) so this branch stays fully on-device.
        col_min = cp.nanmin(d_arr, axis=0, keepdims=True)
        col_max = cp.nanmax(d_arr, axis=0, keepdims=True)
        # 2026-05-30 Wave 9.1 fix (loop iter 33): mirrors the CPU
        # ``discretize_uniform`` fix - canonical formula
        # ``rev_bin_width = n_bins / (max - min)`` with constant-column
        # zero fallback. The pre-fix formula
        # ``n_bins / (max - min + min/2)`` silently mis-binned positive-
        # shifted columns (e.g. linspace(1000, 1100) collapsed to 2 bins
        # instead of 10) AND broke on negative ranges via div-by-zero
        # / sign flip. Cross-backend bit-comparability still holds
        # because both backends now use the same canonical formula.
        _rng = col_max - col_min
        # Where range is zero (constant column), substitute 1 to avoid
        # div-by-zero; the resulting code is clamped to 0 below so the
        # column emits a single bin honestly.
        _rng_safe = cp.where(_rng > 0, _rng, 1.0)
        rev = n_bins / _rng_safe
        out_f = (d_arr - col_min) * rev
        out_f = cp.where(_rng > 0, out_f, 0.0)
        out_f = cp.clip(out_f, 0, n_bins - 1)
        # an individual NaN VALUE (not just a NaN-poisoned column min/max,
        # already fixed above) still produces NaN through the affine map; cp.clip is a no-op on NaN (like
        # numpy), so without this it would cast to an undefined/garbage int code. The CPU discretize_uniform
        # kernel routes NaN rows to a dedicated code one past the real range (``nan_code = n_bins``) instead
        # of colliding with a real bin -- mirror that here so NaN rows carry the same, correct, honest code
        # on both backends rather than a silent garbage cast.
        out_f = cp.where(cp.isnan(d_arr), float(n_bins), out_f)
        out = out_f.astype(_out_cp_dtype)

    # D2H the final tensor (single transfer, n_rows * n_cols bytes for int8).
    return np.asarray(cp.asnumpy(out).astype(dtype, copy=False))


def _choose_discretize_row_chunk_rows(n_cols: int, in_itemsize: int, free_bytes: int) -> int:
    """Rows of ``arr`` (``n_cols`` columns, ``in_itemsize`` bytes/element) that fit a single row-chunk
    upload within a safe VRAM budget (40% of free VRAM, leaving headroom for the output chunk + any
    quantile-edge/reduction scratch). Clamped to >=10_000 rows (a tiny chunk would need an excessive
    number of launches) and to 20M as a sane ceiling."""
    budget = max(0, int(free_bytes * 0.4))
    per_row_bytes = max(1, n_cols * (in_itemsize + 2))  # input row + a small output/scratch margin
    rows = budget // per_row_bytes
    return int(np.clip(rows, 10_000, 20_000_000))


def discretize_2d_array_cuda_row_chunked(
    arr: np.ndarray,
    n_bins: int = 10,
    method: str = "quantile",
    dtype: type = np.int8,
    quantile_subsample_rows: Optional[int] = None,
    free_bytes: Optional[int] = None,
) -> np.ndarray:
    """Row-chunked variant of :func:`discretize_2d_array_cuda` for when the FULL ``arr`` upload would not
    safely fit in free VRAM. Uploads ``arr`` in row-chunks small enough to fit; the two methods handle the
    cross-chunk statistic differently:

    * ``method="uniform"``: EXACT, no approximation. Column min/max are genuinely reducible across row-
      chunks (running min/max, pass 1), then the elementwise bin formula is applied per row-chunk (pass 2)
      using the exact global min/max -- bit-identical to :func:`discretize_2d_array_cuda`.
    * ``method="quantile"``: APPROXIMATE by construction. Exact quantiles need the full column's order
      statistics, which is NOT reducible across row-chunks without a streaming quantile algorithm. Instead,
      bin edges are computed from a GPU-resident random SUBSAMPLE (``quantile_subsample_rows``, default
      ``None`` -> ``feature_engineering.UNIFIED_FE_SUBSAMPLE_N`` = 30_000, the SAME validated MI-sweep
      subsample size used throughout MRMR's FE pipeline -- jaccard=1.0 vs full-n at 50k+, 0.88 at 5k, per
      the bench backing that constant. Quantile-edge estimation has far lower sampling variance than the
      MI estimation that constant was validated for, so 30k is comfortably sufficient here too) then
      applied via row-chunked ``searchsorted`` (exact application of approximate edges). This matches the
      project's documented FE/MRMR exception (a binning/candidate-MI speed lever's bar is SELECTION-
      equivalence, not bit-identical MI). See
      ``tests/feature_selection/discretization/test_discretize_2d_array_row_chunked.py`` for the
      closeness/selection-equivalence validation.

    Returns a plain ``np.ndarray`` (D2H happens per row-chunk, not as one giant transfer at the end).

    ``free_bytes``: an optional already-probed free-VRAM byte count (``cp.cuda.runtime.memGetInfo()``'s
    first element). When the caller (``discretize_2d_array``'s CUDA-eligibility gate) already probed
    free VRAM microseconds earlier for its own reject decision, passing it here skips this function's
    own redundant ``memGetInfo`` call -- ``memGetInfo`` is a read-only device counter query with no
    intervening GPU allocation between the two probes, so reusing the value changes no decision.
    ``None`` (the default -- direct/standalone calls) keeps the self-probe unchanged.
    """
    import cupy as cp

    if quantile_subsample_rows is None:
        from mlframe.feature_selection.filters.feature_engineering import UNIFIED_FE_SUBSAMPLE_N

        quantile_subsample_rows = UNIFIED_FE_SUBSAMPLE_N

    if method not in ("quantile", "uniform"):
        raise NotImplementedError(f"discretize_2d_array_cuda_row_chunked implements 'quantile' / 'uniform'; got method={method!r}")
    if arr.ndim != 2:
        raise ValueError(f"expected 2-D array; got shape {arr.shape}")

    n_rows, n_cols = arr.shape
    if n_rows == 0 or n_cols == 0:
        return np.empty(arr.shape, dtype=dtype)

    dtype = _safe_code_dtype(n_bins, dtype)
    # No throwaway GPU allocation just to read a dtype object -- ``np.dtype(dtype)`` produces the identical
    # ``numpy.dtype`` instance cupy's own ``.dtype`` attribute would (cupy dtypes ARE numpy dtypes), so the
    # 1-element ``cp.asarray``/upload this used to pay per call is pure overhead with no different result.
    _out_cp_dtype = cp.int8 if dtype == np.int8 else np.dtype(dtype)

    if free_bytes is not None:
        free_b = int(free_bytes)
    else:
        try:
            free_b, _total_b = cp.cuda.runtime.memGetInfo()
        except Exception:
            free_b = 512 * 1024 * 1024  # conservative fallback if the probe is unavailable
    row_chunk_rows = _choose_discretize_row_chunk_rows(n_cols, arr.dtype.itemsize, free_b)
    _quantile_subsample_note = f", quantile_subsample_rows={min(n_rows, quantile_subsample_rows)}/{n_rows}" if method == "quantile" else ""
    logger.info(
        "discretize_2d_array_cuda_row_chunked: method=%s n_rows=%d n_cols=%d in_dtype=%s -> row_chunk_rows=%d "
        "(%d chunk(s)), free_vram=%.2fGB%s",
        method, n_rows, n_cols, arr.dtype, row_chunk_rows, -(-n_rows // row_chunk_rows), free_b / 1024**3,
        _quantile_subsample_note,
    )

    out: np.ndarray = np.empty((n_rows, n_cols), dtype=dtype)
    n_chunks = 0

    if method == "uniform":
        col_min_d: Any = None
        col_max_d: Any = None
        # DISCRETIZATION-1 fix: mirrors the B-12 fix already landed on the
        # non-chunked sibling discretize_2d_array_cuda -- plain cp.min/cp.max propagate NaN (a single NaN
        # anywhere in a column poisons that column's min/max to NaN), so use cp.nanmin/cp.nanmax instead.
        for row_start in range(0, n_rows, row_chunk_rows):
            row_end = min(row_start + row_chunk_rows, n_rows)
            d_chunk = cp.asarray(arr[row_start:row_end])
            cmin = cp.nanmin(d_chunk, axis=0)
            cmax = cp.nanmax(d_chunk, axis=0)
            col_min_d = cmin if col_min_d is None else cp.minimum(col_min_d, cmin)
            col_max_d = cmax if col_max_d is None else cp.maximum(col_max_d, cmax)
            del d_chunk
        _rng = col_max_d - col_min_d
        _rng_safe = cp.where(_rng > 0, _rng, 1.0)
        rev = n_bins / _rng_safe
        for row_start in range(0, n_rows, row_chunk_rows):
            row_end = min(row_start + row_chunk_rows, n_rows)
            d_chunk = cp.asarray(arr[row_start:row_end])
            out_f = (d_chunk - col_min_d) * rev
            out_f = cp.where(_rng > 0, out_f, 0.0)
            out_f = cp.clip(out_f, 0, n_bins - 1)
            # DISCRETIZATION-1 fix: route individual NaN VALUES to the dedicated
            # NaN bin code (n_bins), matching the CPU discretize_uniform kernel and the fixed non-chunked
            # sibling -- without this, cp.clip is a no-op on NaN and it would cast to a garbage int code.
            out_f = cp.where(cp.isnan(d_chunk), float(n_bins), out_f)
            out[row_start:row_end] = cp.asnumpy(out_f.astype(_out_cp_dtype))
            del d_chunk, out_f
            n_chunks += 1
    else:  # quantile
        sub_n = min(n_rows, quantile_subsample_rows)
        if sub_n < n_rows:
            sub_idx = np.sort(np.random.default_rng(0).choice(n_rows, size=sub_n, replace=False))
            sub_arr = arr[sub_idx]
        else:
            sub_arr = arr
        d_sub = cp.asarray(sub_arr)
        qs = cp.linspace(0.0, 100.0, n_bins + 1)
        # DISCRETIZATION-1 fix: mirrors the B-12 fix on the non-chunked sibling --
        # cp.percentile has no nanpercentile twin, and a plain cp.percentile over a NaN-bearing subsample
        # poisons EVERY edge for that column with NaN, collapsing the whole column to one degenerate bin.
        if bool(cp.isnan(d_sub).any()):
            bin_edges = cp.asarray(np.nanpercentile(sub_arr, cp.asnumpy(qs), axis=0))
        else:
            bin_edges = cp.percentile(d_sub, qs, axis=0)
        del d_sub
        # Cut points are derived from ``bin_edges`` ONCE here (fit-constant across every row-chunk below)
        # instead of being re-transposed inside ``_discretize_quantile_rawkernel`` on every chunk call --
        # that re-derivation was an O(n_cols * n_bins) transpose+copy repeated per chunk for identical output.
        cuts = cp.ascontiguousarray(bin_edges[1:-1, :].T) if n_cols >= 1000 else None  # (n_cols, n_bins-1)
        for row_start in range(0, n_rows, row_chunk_rows):
            row_end = min(row_start + row_chunk_rows, n_rows)
            d_chunk = cp.asarray(arr[row_start:row_end])
            if n_cols >= 1000:
                chunk_out = _discretize_quantile_rawkernel(d_chunk, cuts, n_bins, _out_cp_dtype)
            else:
                chunk_out = cp.empty((row_end - row_start, n_cols), dtype=_out_cp_dtype)
                for j in range(n_cols):
                    chunk_out[:, j] = cp.searchsorted(bin_edges[1:-1, j], d_chunk[:, j], side="right")
            out[row_start:row_end] = cp.asnumpy(chunk_out)
            del d_chunk, chunk_out
            n_chunks += 1

    logger.debug(
        "discretize_2d_array_cuda_row_chunked: method=%s n_rows=%d n_cols=%d row_chunk_rows=%d n_chunks=%d",
        method, n_rows, n_cols, row_chunk_rows, n_chunks,
    )
    return out


_SEARCHSORTED_RIGHT_2D_SRC = r"""
extern "C" __global__ void searchsorted_right_2d(
    const double* __restrict__ arr,    // (n_rows, n_cols) C-order
    const double* __restrict__ cuts,    // (n_cols, n_cuts) C-order
    int* __restrict__ out,              // (n_rows, n_cols)
    const int n_rows, const int n_cols, const int n_cuts
){
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n_rows * n_cols;
    if (gid >= total) return;
    int row = gid / n_cols;
    int col = gid % n_cols;
    double v = arr[row * n_cols + col];
    // searchsorted side='right': bin = first index i s.t. cuts[i] > v,
    // OR n_cuts if every cut <= v.
    int lo = 0, hi = n_cuts;
    while (lo < hi) {
        int mid = (lo + hi) >> 1;
        if (cuts[col * n_cuts + mid] > v) hi = mid;
        else lo = mid + 1;
    }
    out[row * n_cols + col] = lo;
}
"""
_searchsorted_right_2d_cuda = None
_DISCRETIZE_KERNEL_LOCK = threading.Lock()


def _get_searchsorted_right_2d_kernel():
    """Build (idempotently) and return the fused per-column searchsorted RawKernel.

    Mirrors ``info_theory._cmi_cuda._get_kernel``'s module-level-singleton pattern. ``cp.RawKernel`` used
    to be rebuilt from CUDA source text on EVERY call to ``_discretize_quantile_rawkernel`` -- which
    ``discretize_2d_array_cuda_row_chunked``'s quantile branch calls once per row-chunk (up to 10-50+
    times per large discretize call) -- so the source was recompiled that many times per fit instead of
    once for the whole process lifetime.
    """
    global _searchsorted_right_2d_cuda
    if _searchsorted_right_2d_cuda is not None:
        return _searchsorted_right_2d_cuda
    import cupy as cp

    with _DISCRETIZE_KERNEL_LOCK:
        if _searchsorted_right_2d_cuda is not None:
            return _searchsorted_right_2d_cuda
        module = sys.modules[__name__]
        module._searchsorted_right_2d_cuda = cp.RawKernel(  # type: ignore[attr-defined]
            _SEARCHSORTED_RIGHT_2D_SRC, "searchsorted_right_2d",
        )
        return module._searchsorted_right_2d_cuda


def _discretize_quantile_rawkernel(d_arr, cuts, n_bins, out_cp_dtype):
    """Fused per-column searchsorted via cupy RawKernel.

    Replaces the Python-loop calling ``cp.searchsorted`` once per column,
    which becomes dispatch-bound at n_cols >= 1000 (~30us launch * 1000 cols
    = 30ms wasted on dispatch alone). The fused kernel does ``n_rows*n_cols``
    binary searches in parallel; for n=1M / p=1000 / n_bins=10 measured ~7ms
    vs ~70ms for the per-col loop on cc 6.1.

    ``cuts`` is the caller's PRE-TRANSPOSED, contiguous ``(n_cols, n_bins-1)`` cut-point matrix (its
    ``bin_edges[1:-1, :].T``) -- hoisted out of this function because ``bin_edges``/``cuts`` are fit-
    constant across every row-chunk of one ``discretize_2d_array_cuda_row_chunked`` call, so re-deriving
    them here on every call re-paid an O(n_cols * n_bins) transpose+copy per chunk for identical output.
    """
    import cupy as cp
    n_rows, n_cols = d_arr.shape
    out_int32 = cp.empty((n_rows, n_cols), dtype=cp.int32)
    kernel = _get_searchsorted_right_2d_kernel()
    threads = 256
    blocks = (n_rows * n_cols + threads - 1) // threads
    kernel((blocks,), (threads,), (
        d_arr.astype(cp.float64, copy=False), cuts.astype(cp.float64, copy=False),
        out_int32, np.int32(n_rows), np.int32(n_cols), np.int32(n_bins - 1),
    ))
    return out_int32.astype(out_cp_dtype, copy=False)


# Register with the kernel-tuner registry so retune_all / mlframe-tune-kernels
# discover + batch-tune discretize_2d_array (GPU-capable; cpu njit vs cuda cupy).
from pyutilz.performance.kernel_tuning.registry import kernel_tuner

_DISCRETIZE_SPEC = kernel_tuner(
    kernel_name="discretize_2d_array",
    variant_fns=(_discretize_2d_array_njit, discretize_2d_array_cuda),  # both -> auto-invalidate
    tuner=_run_discretize_sweep,
    axes={"n_cells": list(_DISCRETIZE_SWEEP_CELLS)},
    fallback=_discretize_fallback_choice,  # callable (n_cells) -> str
    gpu_capable=True,
    salt=_DISCRETIZE_SALT,
    cli_label="discretize_2d_array",
)


from ._discretization_dataset import (
    categorize_dataset,
    clear_numeric_code_cache,
    create_redundant_continuous_factor,
)
