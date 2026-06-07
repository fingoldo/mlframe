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
import warnings

import numpy as np
import pandas as pd
from numba import njit, prange
# 2026-05-28: sklearn / astropy removed from categorize_1d_array hot path.
# Pure-numpy + numba kernels are ~10x faster than KBinsDiscretizer / OrdinalEncoder
# (single-threaded estimator-API overhead) and ~12x faster than astropy.histogram
# for the supported bin schemes. The legacy methods 'astropy' and 'discretizer'
# still resolve via thin compat shims below.
def _native_ordinal_encode_2d(vals: np.ndarray) -> np.ndarray:
    """Drop-in pure-numpy replacement for sklearn OrdinalEncoder().fit_transform on a (n, 1) array.

    Returns float64 ordinals so downstream digitize / dtype-promotion logic stays bit-for-bit
    identical to the sklearn path. ``pd.factorize`` is asymptotically the same numpy unique
    + inverse-index lookup but skips estimator-validation overhead (~6x faster at n=10k).
    """
    flat = vals.reshape(-1)
    codes, _ = pd.factorize(flat, use_na_sentinel=True)
    return codes.astype(np.float64).reshape(vals.shape)


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
        if len(needs_factorize) <= 1:
            for _j, _c in needs_factorize:
                _codes, _ = pd.factorize(categorical_df[_c], use_na_sentinel=True)
                out[:, _j] = _codes.astype(np.float64)
        else:
            # joblib threading. pd.factorize releases the GIL on the hash build,
            # so threads parallelise. prefer='threads' avoids the pickling cost
            # of process workers on a categorical DF view.
            from joblib import Parallel, delayed as _delayed
            _results = Parallel(n_jobs=min(8, len(needs_factorize)), prefer="threads")(
                _delayed(lambda c: pd.factorize(categorical_df[c], use_na_sentinel=True)[0].astype(np.float64))(_c)
                for _j, _c in needs_factorize
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

from ._discretization_edges import (  # noqa: F401
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


def _handle_missing(arr: np.ndarray, *, strategy: str = "fillna_zero") -> np.ndarray:
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
    re-routes NaN positions to the dedicated NaN bin.
    Private -- external callers should use the public ``discretize_*`` family.
    """
    if not np.isnan(arr).any():
        return arr
    if strategy == "fillna_zero":
        return np.where(np.isnan(arr), 0.0, arr)
    if strategy == "separate_bin":
        # The actual bin re-routing happens in categorize_dataset after
        # discretization. Here we replace NaN with column median so np.percentile
        # produces clean bin edges; the original NaN positions are preserved
        # via the caller's nan-mask and overwritten back to max_bin+1 below.
        # nanmedian emits RuntimeWarning("All-NaN slice encountered") on all-NaN
        # columns, but the np.where below handles that case explicitly by
        # falling back to 0.0; suppress the noise to keep test output clean.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            col_medians = np.nanmedian(arr, axis=0)
        # Empty / all-NaN columns: median is NaN; fall back to 0.0 for the
        # discretize edges (the column will be all-NaN-bin anyway).
        col_medians = np.where(np.isnan(col_medians), 0.0, col_medians)
        # Broadcast-fill (rows, cols) where row is NaN.
        filled = np.where(np.isnan(arr), col_medians, arr)
        return filled
    if strategy == "propagate":
        # 2026-05-30 Wave 9.1 fix (loop iter 11): propagate USED to return
        # the NaN-bearing array unchanged, but downstream ``np.searchsorted``
        # routes NaN to ``ej.size`` -- the same code as the column's top
        # real bin -- silently merging NaN rows with the highest-value
        # real category. Net effect: any column whose NaN-ness carried
        # signal scored near zero MI (verified: column where NaN-ness IS
        # the target dropped from MI=0.69 nats under separate_bin to
        # MI=0.38 under propagate). Fix: behave like ``separate_bin``
        # at the categorize_dataset level - the actual NaN-bin reassignment
        # happens in the categorize_dataset post-discretize block, but
        # here we still need to median-fill so np.percentile gets clean
        # edges. The caller's ``_nan_mask`` capture at categorize_dataset
        # line 1027 was also extended to include 'propagate'.
        # nanmedian emits RuntimeWarning on all-NaN columns; suppress (the
        # subsequent np.where handles the NaN-median path explicitly).
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            col_medians = np.nanmedian(arr, axis=0)
        col_medians = np.where(np.isnan(col_medians), 0.0, col_medians)
        return np.where(np.isnan(arr), col_medians, arr)
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
        logger.warning(
            "MRMR autocollecting LazyFrame at boundary. Pass a materialised DataFrame to skip this copy."
        )
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
            raise ValueError(
                "categorize_1d_array: input contains NaN and nan_filler=None; "
                "drop NaN upstream or pick a non-colliding sentinel."
            )
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
                bin_edges = np.histogram_bin_edges(vals, bins=bins)
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
                raise ValueError(
                    f"categorize_1d_array: unknown method={method!r}; expected one of "
                    "'discretizer', 'numpy', 'astropy'."
                )

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
            raise ValueError(
                f"categorize_1d_array: cardinality {out_max} exceeds int64 max; cannot encode."
            )
    return out.astype(dtype)


# =============================================================================
# Low-level numba kernels (pure functions; no module-level side-effects)
# =============================================================================


@njit(cache=True)
def digitize(arr: np.ndarray, bins: np.ndarray, dtype=np.int32) -> np.ndarray:
    res = np.empty(len(arr), dtype=dtype)
    for i, val in enumerate(arr):
        for j, bin_edge in enumerate(bins):
            if val <= bin_edge:
                res[i] = j
                break
    return res



@njit(cache=True)
def quantize_dig(arr, bins):
    return np.digitize(arr, bins[1:-1], right=True)


@njit(cache=True)
def quantize_search(arr, bins):
    return np.searchsorted(bins[1:-1], arr, side="right")


@njit(cache=True)
def discretize_uniform(arr: np.ndarray, n_bins: int, min_value: float = None, max_value: float = None, dtype: object = np.int8) -> np.ndarray:
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
    _rng = max_value - min_value
    if _rng <= 0:
        # Constant column: every row -> bin 0; honest single-bin code.
        return np.zeros_like(arr, dtype=dtype)
    rev_bin_width = n_bins / _rng
    result = ((arr - min_value) * rev_bin_width).astype(dtype)
    return np.clip(result, 0, n_bins - 1)


def discretize_array(
    arr: np.ndarray, n_bins: int = 10, method: str = "quantile",
    min_value: float = None, max_value: float = None, dtype: object = np.int8,
) -> np.ndarray:
    """Discretise a 1-D continuous array into ordinal bins.

    Single-column path uses raw numpy instead of dispatching to the ``@njit`` ``_discretize_array_impl``. Microbench at n=10000: njit ``np.percentile`` ~870us
    vs direct ``np.percentile`` ~405us (numba is ~2x slower than numpy at this size for percentile work). The FE pipeline calls this 6000+ times per fit on
    n=10000, p=200 -- the un-njit path saves ~3s. Multi-column ``discretize_2d_array`` keeps the njit chain because it parallelises columns via ``prange``.
    """
    if method not in ("uniform", "quantile"):
        raise ValueError(f"Unsupported discretization method: '{method}'. Supported methods: 'uniform', 'quantile'")
    if method == "uniform":
        return discretize_uniform(arr=arr, n_bins=n_bins, min_value=min_value, max_value=max_value, dtype=dtype)
    # quantile path -- raw numpy.
    # Wave 21 P0: nanpercentile so NaN-bearing columns don't collapse to a
    # constant via the all-NaN bin_edges trap. Same finding as the ``edges``
    # helper above.
    quantiles = np.linspace(0, 100, n_bins + 1)
    bins_edges = np.nanpercentile(arr, quantiles)
    return np.searchsorted(bins_edges[1:-1], arr, side="right").astype(dtype)


@njit(parallel=True, nogil=True, cache=True)
def _quantile_edges_2d_njit(arr2d: np.ndarray, quantiles: np.ndarray, edges_out: np.ndarray) -> None:
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
        col.sort()
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


def discretize_2d_quantile_batch(arr2d: np.ndarray, n_bins: int = 10, dtype: object = np.int8) -> np.ndarray:
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
    n_rows, n_cols = arr2d.shape
    quantiles = np.linspace(0, 100, n_bins + 1)
    # Fast path: NaN-free buffer -> ``_quantile_edges_2d_njit`` (sort each column ONCE,
    # read all quantiles) is BIT-IDENTICAL to ``np.percentile(axis=0, method='linear')``
    # (verified across float32/64, ties, constant columns) and removes the numpy
    # ``ndarray.partition`` hotspot -- the FE sweep's single dominant numpy cost
    # (scene 1500x299 cProfile: 114.5s / 20% of fit in ``partition``, re-partitioning the
    # whole buffer once per quantile). NaN path keeps ``np.nanpercentile`` (the njit kernel
    # does not NaN-handle; callers that pass NaN are rare and stay correct + bit-identical).
    if np.isnan(arr2d).any():
        edges = np.nanpercentile(arr2d, quantiles, axis=0)
    else:
        edges = np.empty((quantiles.shape[0], n_cols), dtype=np.float64)
        if n_cols > 0 and n_rows > 0:
            _quantile_edges_2d_njit(np.ascontiguousarray(arr2d), quantiles, edges)
    out = np.empty((n_rows, n_cols), dtype=dtype)
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
    if n_cols > 0 and n_rows > 0:
        edges_inner = np.ascontiguousarray(edges[1:-1], dtype=np.float64)
        arr_c = np.ascontiguousarray(arr2d)
        _searchsorted_2d_right_njit(edges_inner, arr_c, out)
    return out


@njit(cache=True)
def _discretize_array_impl(
    arr: np.ndarray, n_bins: int = 10, method: str = "quantile",
    min_value: float = None, max_value: float = None, dtype: object = np.int8,
) -> np.ndarray:
    if method == "uniform":
        return discretize_uniform(arr=arr, n_bins=n_bins, min_value=min_value, max_value=max_value, dtype=dtype)
    elif method == "quantile":
        bins_edges = get_binning_edges(arr=arr, n_bins=n_bins, method=method, min_value=min_value, max_value=max_value)
    return quantize_search(arr, bins_edges).astype(dtype)


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
    min_values: float = None,
    max_values: float = None,
    dtype: object = np.int8,
) -> np.ndarray:
    """CPU prange backend; one column per worker thread."""
    res = np.empty_like(arr, dtype=dtype)
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
        return _discretize_2d_array_njit(
            arr=arr, n_bins=10, method="quantile", min_ncats=50,
            min_values=None, max_values=None, dtype=np.int8,
        )

    variants = {"cpu": _cpu}
    if is_cuda_available():
        def _cuda(arr):
            return discretize_2d_array_cuda(arr=arr, n_bins=10, method="quantile", dtype=np.int8)
        variants["cuda"] = _cuda
    return sweep_backend_grid(
        variants, {"n_cells": _DISCRETIZE_SWEEP_CELLS}, _make_discretize_inputs,
        reference="cpu", repeats=2, equiv_rtol=1e-6, equiv_atol=1e-6,
    )


def _discretize_fallback_choice(n_cells: int) -> str:
    """Pre-sweep heuristic (the spec's dynamic fallback callable): cuda above the
    hand-tuned 500k-cell breakeven, else cpu."""
    return "cuda" if int(n_cells) >= _DISCRETIZE_2D_CUDA_MIN_CELLS else "cpu"


def discretize_2d_array(
    arr: np.ndarray,
    n_bins: int = 10,
    method: str = "quantile",
    min_ncats: int = 50,
    min_values: float = None,
    max_values: float = None,
    dtype: object = np.int8,
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
                try:
                    return discretize_2d_array_cuda(
                        arr=arr, n_bins=n_bins, method=method, dtype=dtype,
                    )
                except Exception as exc:
                    logger.debug(
                        "discretize_2d_array: CUDA fastpath failed (%s: %s); "
                        "falling back to CPU prange",
                        type(exc).__name__, exc,
                    )
        except ImportError:
            pass

    return _discretize_2d_array_njit(
        arr=arr, n_bins=n_bins, method=method, min_ncats=min_ncats,
        min_values=min_values, max_values=max_values, dtype=dtype,
    )


def discretize_2d_array_cuda(
    arr: np.ndarray,
    n_bins: int = 10,
    method: str = "quantile",
    dtype: object = np.int8,
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

    d_arr = cp.asarray(arr)  # H2D once for the whole frame
    _out_cp_dtype = cp.int8 if dtype == np.int8 else cp.asarray(np.zeros(1, dtype=dtype)).dtype
    out = cp.empty((n_rows, n_cols), dtype=_out_cp_dtype)

    if method == "quantile":
        qs = cp.linspace(0.0, 100.0, n_bins + 1)
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
            out = _discretize_quantile_rawkernel(d_arr, bin_edges, n_bins, _out_cp_dtype)
        else:
            for j in range(n_cols):
                out[:, j] = cp.searchsorted(bin_edges[1:-1, j], d_arr[:, j], side="right")
    else:
        # method == 'uniform': vectorised arithmetic, no percentile sort,
        # no per-column dispatch. Single GPU pass. Mirrors discretize_uniform
        # njit kernel on CPU. Fastest path for Gaussian-ish data where the
        # accuracy hit vs quantile is small (bench at info_theory module
        # docstring quotes H(X)/log(nbins) >= 0.82 for Gaussian).
        col_min = cp.min(d_arr, axis=0, keepdims=True)
        col_max = cp.max(d_arr, axis=0, keepdims=True)
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
        out = out_f.astype(_out_cp_dtype)

    # D2H the final tensor (single transfer, n_rows * n_cols bytes for int8).
    return cp.asnumpy(out).astype(dtype, copy=False)


def _discretize_quantile_rawkernel(d_arr, bin_edges, n_bins, out_cp_dtype):
    """Fused per-column searchsorted via cupy RawKernel.

    Replaces the Python-loop calling ``cp.searchsorted`` once per column,
    which becomes dispatch-bound at n_cols >= 1000 (~30us launch * 1000 cols
    = 30ms wasted on dispatch alone). The fused kernel does ``n_rows*n_cols``
    binary searches in parallel; for n=1M / p=1000 / n_bins=10 measured ~7ms
    vs ~70ms for the per-col loop on cc 6.1.

    bin_edges shape: (n_bins+1, n_cols); we use rows [1, n_bins-1] inclusive
    (i.e. n_bins-1 right-side cut points per column) and use searchsorted-right
    semantics.
    """
    import cupy as cp
    n_rows, n_cols = d_arr.shape
    # Cut points: shape (n_bins-1, n_cols). Contiguous in column-major so each
    # column's edges are adjacent in memory after .T.copy().
    cuts = cp.ascontiguousarray(bin_edges[1:-1, :].T)  # (n_cols, n_bins-1)
    out_int32 = cp.empty((n_rows, n_cols), dtype=cp.int32)
    src = r'''
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
    '''
    kernel = cp.RawKernel(src, "searchsorted_right_2d")
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


from ._discretization_dataset import (  # noqa: E402,F401
    categorize_dataset,
    create_redundant_continuous_factor,
)
