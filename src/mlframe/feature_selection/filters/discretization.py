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
from typing import Sequence

import numpy as np
import pandas as pd
from joblib import delayed
from numba import jit, njit, prange
from sklearn.preprocessing import KBinsDiscretizer, OrdinalEncoder

try:
    from astropy.stats import histogram as _astropy_histogram
except (ImportError, AttributeError):
    # astropy may be wedged by transitive numpy-API removal (e.g. np.in1d
    # gone in numpy 2.x while older astropy still imports it). Fall back
    # to np.histogram — same (hist, edges) contract for string-rule and
    # integer ``bins`` values; the astropy-specific features (Bayesian
    # blocks etc.) aren't used in this module.
    _astropy_histogram = None


def histogram(a, bins="auto", **kwargs):
    """Astropy histogram with np.histogram fallback. See
    ``mlframe.feature_engineering.numerical.histogram`` for the rationale.
    """
    if _astropy_histogram is not None:
        return _astropy_histogram(a, bins=bins, **kwargs)
    return np.histogram(a, bins=bins, **kwargs)

from mlframe.core.arrays import arrayMinMax
from pyutilz.parallel import parallel_run
from pyutilz.system import tqdmu

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
    ``"propagate"``: leave NaN in place (legacy polars behaviour); the
    numba kernel will route to the lowest bin or raise depending on
    bounds-checking.
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
        col_medians = np.nanmedian(arr, axis=0)
        # Empty / all-NaN columns: median is NaN; fall back to 0.0 for the
        # discretize edges (the column will be all-NaN-bin anyway).
        col_medians = np.where(np.isnan(col_medians), 0.0, col_medians)
        # Broadcast-fill (rows, cols) where row is NaN.
        filled = np.where(np.isnan(arr), col_medians, arr)
        return filled
    if strategy == "propagate":
        return arr
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
    """Per-column ordinal encoder used by ad-hoc external pipelines. Inside MRMR proper we use ``categorize_dataset`` below."""
    ordinal_encoder = OrdinalEncoder()

    if vals.dtype.name != "category" and np.issubdtype(vals.dtype, np.bool_):
        vals = vals.astype(np.int8)

    if pd.isna(vals).any():
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
                discretizer = KBinsDiscretizer(**method_kwargs, encode="ordinal")
                new_vals = discretizer.fit_transform(vals)
            else:
                new_vals = ordinal_encoder.fit_transform(vals)
        else:
            if method == "numpy":
                bin_edges = np.histogram_bin_edges(vals, bins=bins)
            elif method == "astropy":
                if bins == "blocks" and len(vals) >= astropy_sample_size:
                    _, bin_edges = histogram(np.random.choice(vals.ravel(), size=astropy_sample_size, replace=False), bins=bins)
                elif bins == "knuth" and len(vals) >= astropy_sample_size:
                    _, bin_edges = histogram(np.random.choice(vals.ravel(), size=astropy_sample_size, replace=False), bins=bins)
                else:
                    _, bin_edges = histogram(vals, bins=bins)

            if bin_edges[0] <= vals.min():
                bin_edges = bin_edges[1:]

            new_vals = ordinal_encoder.fit_transform(np.digitize(vals, bins=bin_edges, right=True))
    else:
        new_vals = ordinal_encoder.fit_transform(vals)

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


def edges(arr, quantiles):
    # Wave 21 P0: use nanpercentile so NaN in arr doesn't poison every
    # bin edge. ``discretize_array`` calls this 6000+ times per FS fit
    # (per the module docstring); pre-fix any NaN-bearing column made
    # bin_edges all-NaN, then digitize / searchsorted silently bucketed
    # every row to bin 0 -- the entire discretised feature collapsed to
    # a constant with no upstream signal.
    bin_edges = np.asarray(np.nanpercentile(arr, quantiles))
    return bin_edges


@njit(cache=True)
def quantize_dig(arr, bins):
    return np.digitize(arr, bins[1:-1], right=True)


@njit(cache=True)
def quantize_search(arr, bins):
    return np.searchsorted(bins[1:-1], arr, side="right")


@njit(cache=True)
def discretize_uniform(arr: np.ndarray, n_bins: int, min_value: float = None, max_value: float = None, dtype: object = np.int8) -> np.ndarray:
    if min_value is None or max_value is None:
        min_value, max_value = arrayMinMax(arr)
    rev_bin_width = n_bins / (max_value - min_value + min_value / 2)
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
    # CUDA-eligibility gate. ``min_cells`` comes from the per-host kernel
    # tuning cache (pyutilz.system.kernel_tuning_cache + auto_tune sweep)
    # when available; else the hand-tuned 500k default. Lets the dispatcher
    # adapt to faster GPUs (cc 8+ wins at smaller sizes) without code edits.
    # Uses the module-singleton cache; building a fresh KernelTuningCache here
    # would re-trigger _load + _build_provenance (nvidia-smi subprocess) on
    # every call (~48ms each, observed 6x in fuzz combo c0143 profile).
    min_cells = _DISCRETIZE_2D_CUDA_MIN_CELLS
    from ._kernel_tuning import get_kernel_tuning_cache
    _cache = get_kernel_tuning_cache()
    if _cache is not None:
        try:
            _entry = _cache.lookup(
                "discretize_2d_array",
                arr_size=int(arr.size) if hasattr(arr, "size") else 0,
            )
            if _entry is not None and "min_cells" in _entry:
                min_cells = int(_entry["min_cells"])
        except Exception:
            pass  # lookup error -> hand-tuned default

    if (
        prefer_gpu
        and method == "quantile"
        and min_values is None
        and max_values is None
        and arr.ndim == 2
        and arr.size >= min_cells
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

    if method != "quantile":
        raise NotImplementedError(
            f"discretize_2d_array_cuda only implements 'quantile'; got method={method!r}",
        )

    if arr.ndim != 2:
        raise ValueError(f"expected 2-D array; got shape {arr.shape}")

    n_rows, n_cols = arr.shape
    if n_rows == 0 or n_cols == 0:
        return np.empty(arr.shape, dtype=dtype)

    d_arr = cp.asarray(arr)  # H2D once for the whole frame
    qs = cp.linspace(0.0, 100.0, n_bins + 1)
    # cp.percentile vectorises across axis=0 -> bin_edges shape: (n_bins + 1, n_cols).
    bin_edges = cp.percentile(d_arr, qs, axis=0)

    # cp.searchsorted is 1-D; loop per column. Each call is fully on-device
    # so the loop is dispatch-overhead only (~30 us per launch). For
    # n_cols=30 the total dispatch is ~1 ms vs ~50 ms compute.
    out = cp.empty((n_rows, n_cols), dtype=cp.int8 if dtype == np.int8 else cp.asarray(np.zeros(1, dtype=dtype)).dtype)
    # bin_edges[1:-1, j] is the right-side cut points for column j (n_bins - 1 edges).
    for j in range(n_cols):
        out[:, j] = cp.searchsorted(bin_edges[1:-1, j], d_arr[:, j], side="right")

    # D2H the final tensor (single transfer, n_rows * n_cols bytes for int8).
    return cp.asnumpy(out).astype(dtype, copy=False)


@njit(cache=True)
def get_binning_edges(arr: np.ndarray, n_bins: int = 10, method: str = "uniform",
                       min_value: float = None, max_value: float = None):
    """Numba-jitted binning-edge calculator. Used by ``discretize_2d_array`` (itself ``@njit(parallel=True)`` and cannot dispatch to object-mode helpers).

    Outside an njit context (single-column path via ``discretize_array``) prefer the inlined raw-numpy version -- ``np.percentile`` beats numba's njit
    equivalent at n >= ~5000.
    """
    if method == "uniform":
        if min_value is None or max_value is None:
            min_value, max_value = arrayMinMax(arr)
        bin_edges = np.linspace(min_value, max_value, n_bins + 1)
    elif method == "quantile":
        # Wave 21 P0: numba's njit doesn't expose np.nanpercentile, so we
        # filter NaN inline before delegating to np.percentile. Pre-fix any
        # NaN in arr poisoned every edge -> downstream digitize silently
        # bucketed all rows to bin 0. The mask path is array-allocate +
        # one pass, cheaper than the percentile sort that follows.
        _mask = ~np.isnan(arr)
        if _mask.all():
            arr_finite = arr
        else:
            arr_finite = arr[_mask]
        quantiles = np.linspace(0, 100, n_bins + 1)
        bin_edges = np.asarray(np.percentile(arr_finite, quantiles))
    return bin_edges


def discretize_sklearn(
    arr: np.ndarray, n_bins: int = 10, method: str = "uniform",
    min_value: float = None, max_value: float = None, dtype: object = np.int8,
) -> np.ndarray:
    """Lightweight numpy port of sklearn's ``KBinsDiscretizer``.
    ``np.searchsorted`` is faster un-jitted on contemporary numpy."""
    bins_edges = get_binning_edges(arr=arr, n_bins=n_bins, method=method, min_value=min_value, max_value=max_value)
    return np.searchsorted(bins_edges[1:-1], arr, side="right").astype(dtype)


# =============================================================================
# Categorisation of arbitrary value tables (continuous random factors)
# =============================================================================


def create_redundant_continuous_factor(
    df: pd.DataFrame,
    factors: Sequence[str],
    agg_func: object = np.sum,
    noise_percent: float = 5.0,
    dist: object = None,
    dist_args: tuple = (),
    name: str = None,
    sep: str = "_",
) -> None:
    """Out of a few continuous factors, craft a new factor with known relationship and amount of redundancy. Used by tests / benchmark harnesses, not by ``MRMR`` directly."""
    if dist:
        rvs = dist.rvs
        # Wave 31 (2026-05-20): assert -> AttributeError.
        if not callable(rvs):
            raise AttributeError(
                f"dist must have a callable .rvs method; got {dist!r}."
            )
        noise = rvs(*dist_args, size=len(df))
    else:
        noise = np.random.random(len(df))

    val_min, val_max = noise.min(), noise.max()
    if np.isclose(val_max, val_min):
        noise = np.zeros(len(noise), dtype=np.float32)
    else:
        noise = (noise - val_min) / (val_max - val_min)

    if not name:
        name = sep.join(factors) + sep + f"{noise_percent:.0f}%{dist.name if dist else ''}noise"

    df[name] = agg_func(df[factors].values, axis=1) * (1 + (noise - 0.5) * noise_percent / 100)


# =============================================================================
# Top-level entry
# =============================================================================


def categorize_dataset(
    df,
    method: str = "quantile",
    n_bins: int = 4,
    min_ncats: int = 50,
    dtype=np.int16,
    missing_strategy: str = "fillna_zero",
):
    """Convert a DataFrame into an ordinal-encoded ``(n_samples, n_features)`` array. Accepts pandas or polars (DataFrame or LazyFrame -- materialised at the
    boundary). ``missing_strategy`` controls NaN handling: see :func:`_handle_missing`."""
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
            return (
                dt == pl.Utf8
                or dt == pl.String
                or dt == pl.Categorical
                or dt == pl.Boolean
                or (hasattr(pl, "Enum") and isinstance(dt, pl.Enum))
            )
        numerical_cols = [name for name, dt in df.schema.items() if not _is_pl_cat(dt)]
        categorical_cols_detected = [name for name, dt in df.schema.items() if _is_pl_cat(dt)]
    else:
        numerical_cols = df.head(5).select_dtypes(exclude=("category", "object", "bool")).columns.values.tolist()
        categorical_cols_detected = None

    if _is_polars:
        _num_frame = df.select(numerical_cols)
        arr = _num_frame.to_numpy().astype(np.float64, copy=False)
    else:
        arr = df[numerical_cols].to_numpy(dtype=np.float64, na_value=np.nan)

    # Snapshot the NaN positions BEFORE _handle_missing rewrites them: the
    # "separate_bin" strategy fills NaN with the column median so np.percentile
    # produces clean edges, then we overwrite the same positions in the
    # discretized output with bin=n_bins (max+1 per column). Net effect: NaN
    # gets its own honest category that MI estimators see correctly.
    _nan_mask = np.isnan(arr) if (missing_strategy == "separate_bin" and arr.size > 0) else None

    # Unified NaN handling for both pandas and polars.
    arr = _handle_missing(arr, strategy=missing_strategy)

    data = discretize_2d_array(
        arr=arr, n_bins=n_bins, method=method, min_ncats=min_ncats,
        min_values=None, max_values=None, dtype=dtype,
    )

    if _nan_mask is not None and _nan_mask.any():
        # Verify dtype width before overwriting: max bin index after assignment
        # is n_bins (one past the regular [0, n_bins-1] range). Most callers
        # use int16/int32 which handle this trivially; raise if int8 would
        # overflow (n_bins>=127 would also overflow regular bins, so practical
        # n_bins is well below int8 max anyway).
        max_bin_after = n_bins
        if max_bin_after > np.iinfo(data.dtype).max:
            raise ValueError(
                f"separate_bin strategy needs dtype able to hold {max_bin_after}; "
                f"current dtype {data.dtype} max is {np.iinfo(data.dtype).max}. "
                "Pass a wider dtype to categorize_dataset."
            )
        # Overwrite NaN-row, NaN-col positions with the dedicated bin index.
        data[_nan_mask] = max_bin_after

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
        else:
            categorical_cols = []
            new_vals = None
    else:
        categorical_factors = df.select_dtypes(include=("category", "object", "bool"))
        categorical_cols = []
        if categorical_factors.shape[1] > 0:
            categorical_cols = categorical_factors.columns.values.tolist()
            ordinal_encoder = OrdinalEncoder()
            new_vals = ordinal_encoder.fit_transform(categorical_factors)
        else:
            new_vals = None
    if categorical_cols and new_vals is not None:
        # Wave 40 (2026-05-20): the prior log-warn-then-truncate pattern silently wrapped
        # category codes past the target dtype's max (e.g. id 128 -> -128 for int8),
        # then nbins below read the post-wrap max and the joint-histogram in mi.py was
        # sized to the wrapped value. Auto-promote dtype to fit the actual max code so
        # high-cardinality categoricals (user_id / product_sku / hash-encoded targets)
        # produce honest codes; fall back to the original log+truncate only if the
        # caller explicitly forbids promotion via _force_dtype=True kwarg (not exposed
        # at this signature -> always promote).
        max_cats = new_vals.max(axis=0)
        global_max = int(max_cats.max())
        if global_max > np.iinfo(dtype).max:
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
                raise ValueError(
                    f"categorize_dataset: category cardinality {global_max} exceeds int64 max; cannot encode."
                )
        new_vals = new_vals.astype(dtype)

        if data is None:
            data = new_vals
        else:
            data = np.append(data, new_vals, axis=1)

    nbins = data.max(axis=0).astype(np.int64) + 1

    return data, numerical_cols + categorical_cols, nbins
