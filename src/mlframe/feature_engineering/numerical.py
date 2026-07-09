"""Numerical feature engineering for ML. Optimized & rich set of aggregates for 1d vectors."""

from __future__ import annotations

__all__ = [
    "compute_simple_stats_numba",
    "compute_simple_stats_numba_arr",
    "get_simple_stats_names",
    "compute_numerical_aggregates_numba",
    "get_basic_feature_names",
    "compute_nunique_modes_quantiles_numpy",
    "compute_ncrossings",
    "compute_nunique_mode_quantiles_numba",
    "compute_numaggs",
    "get_numaggs_names",
    "get_moments_slope_mi_feature_names",
    "rolling_moving_average",
    "numaggs_over_matrix_rows",
    "compute_numaggs_parallel",
    "cont_entropy",
]

import logging
import warnings
from contextlib import contextmanager
from typing import Any, Optional, Sequence, cast

import numba
import numpy as np
import pandas as pd
import psutil
from joblib import delayed
# antropy is imported lazily (see _get_entropy_funcs below): the top-level
# ``import antropy`` eagerly JIT-compiles its numba kernels, costing ~5s at
# import time. These six fractal/entropy features are only touched when
# return_entropy=True, so we defer that cost to first actual use.

# astropy is imported lazily on first histogram() call (see _resolve below):
# the top-level import costs ~0.6s and this module is on the eager MRMR
# import path, yet most fits never call histogram(). The try/except still
# guards the case where astropy is wedged by a transitive numpy-API
# deprecation (e.g. numpy 2.x removing np.in1d) -- we fall back to
# np.histogram, equivalent for the bins="scott"/"auto" string-rule cases.
_astropy_histogram = None
_astropy_resolved = False


def _resolve_astropy_histogram():
    """Lazily import and cache ``astropy.stats.histogram``, resolving it only once per process."""
    global _astropy_histogram, _astropy_resolved
    if not _astropy_resolved:
        try:
            from astropy.stats import histogram as _h
            _astropy_histogram = _h
        except (ImportError, AttributeError):
            _astropy_histogram = None
        _astropy_resolved = True
    return _astropy_histogram


def histogram(a, bins="auto", **kwargs):
    """Thin shim over astropy.stats.histogram with numpy fallback.

    Keeps the module importable when astropy is wedged by a transitive
    numpy-API deprecation; downstream callers see the same (hist, edges)
    contract for the string-rule bins values cont_entropy / extras rely on.
    """
    _h = _resolve_astropy_histogram()
    if _h is not None:
        return _h(a, bins=bins, **kwargs)
    return np.histogram(a, bins=bins, **kwargs)
from scipy.stats import kstest, rv_continuous
from sklearn.feature_selection import mutual_info_regression

from mlframe.feature_engineering.hurst import compute_hurst_exponent
from pyutilz.parallel import parallel_run

logger = logging.getLogger(__name__)


@contextmanager
def _suppress_numeric_warnings():
    """Scoped replacement for the old module-level ``warnings.simplefilter`` calls.

    Previously this module registered global filters at import time, silently swallowing
    FutureWarning / RuntimeWarning everywhere in the process (including in unrelated caller
    code). Callers that really need suppression can use this context manager.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="nperseg =")
        warnings.simplefilter(action="ignore", category=FutureWarning)
        warnings.simplefilter(action="ignore", category=RuntimeWarning)
        yield

# ----------------------------------------------------------------------------------------------------------------------------
# Inits
# ----------------------------------------------------------------------------------------------------------------------------

from ._numerical_constants import (
    NUMBA_NJIT_PARAMS,
    distributions,
    default_quantiles,
)


def cont_entropy(arr: np.ndarray, bins: str = "scott") -> float:
    """Shannon entropy of a continuous distribution (binned).

    Wave 25 bonus fix (2026-05-20): pre-fix computed
    ``-(hist * log(hist + eps)).sum()`` where ``hist`` was raw integer
    counts, not normalised probabilities. That formula is NOT Shannon
    entropy; it scales with bin count and total sample size, giving
    arbitrary values like 12_345 nats instead of the expected
    (0, log(n_bins)) range. The commented hint
    ``# np.histogram(arr, bins=bins, density=True)`` shows the author
    spotted the missing normalisation but didn't apply it.

    Post-fix: compute probabilities from counts (``p = counts / counts.sum()``)
    then apply the Shannon formula ``-sum(p * log(p))`` over the
    nonzero-probability bins (``p == 0`` contributions are zero by the
    ``0 * log(0) = 0`` convention; we filter them to avoid log(0)
    instead of adding an epsilon that biases the result).
    """
    try:
        hist, _bin_edges = histogram(arr, bins=bins)
        total = float(hist.sum())
        if total <= 0:
            return float("nan")
        p = np.asarray(hist, dtype=np.float64) / total
        nonzero = p > 0.0
        if not np.any(nonzero):
            return 0.0
        ent = -float(np.sum(p[nonzero] * np.log(p[nonzero])))
    except Exception:
        # NaN sentinel is the documented degenerate result; log at DEBUG so a genuine histogram/log-coding
        # bug is diagnosable rather than silently returned as a NaN feature value.
        logger.debug("cont_entropy: returning NaN after an unexpected error", exc_info=True)
        return float("nan")
    return ent


_entropy_funcs_cache = None

# Static names of the entropy callables, in the exact order _get_entropy_funcs
# builds them. Kept as literals so feature-name construction never needs to
# import antropy (whose import JIT-compiles for ~5s); only actual entropy
# *computation* materialises the callables.
_ENTROPY_FUNC_NAMES = ("cont_entropy", "svd_entropy", "sample_entropy", "petrosian_fd", "perm_entropy", "katz_fd", "detrended_fluctuation")


def _get_entropy_funcs() -> tuple:
    """Lazily import antropy and build the entropy-feature callable tuple.

    Deferring the ``antropy`` import to here (rather than module top-level)
    avoids the ~5s of numba JIT that antropy runs on import for callers that
    never request entropy features. Result is cached after the first call.
    """
    global _entropy_funcs_cache
    if _entropy_funcs_cache is None:
        from antropy import detrended_fluctuation, katz_fd, perm_entropy, petrosian_fd, sample_entropy, svd_entropy
        _entropy_funcs_cache = (cont_entropy, svd_entropy, sample_entropy, petrosian_fd, perm_entropy, katz_fd, detrended_fluctuation)  # continuous.get_h, app_entropy
    return _entropy_funcs_cache


def __getattr__(name):
    # PEP 562: preserve the historical module-level ``entropy_funcs`` /
    # ``entropy_funcs_names`` attributes for any external importer, while
    # keeping the antropy import lazy (they are only materialised on access).
    if name == "entropy_funcs":
        return _get_entropy_funcs()
    if name == "entropy_funcs_names":
        return list(_ENTROPY_FUNC_NAMES)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

default_dist_responses = dict(levy_l=(np.nan, np.nan), logistic=(np.nan, np.nan), pareto=(np.nan, np.nan, np.nan))


def get_distributions_features_names() -> list:
    """Return the flat list of feature names emitted by the distribution-fitting features (per-distribution params + KS stat/pvalue)."""
    distributions_features_names = []
    for dist in distributions:
        for i in range(len(default_dist_responses[dist.name])):
            distributions_features_names.append(dist.name + str(i + 1))
        distributions_features_names.append(dist.name + "_kss")
        distributions_features_names.append(dist.name + "_kspval")
    return distributions_features_names


distributions_features_names = get_distributions_features_names()


# ----------------------------------------------------------------------------------------------------------------------------
# Factory: macro-equivalent in Python via numba dead-code-elimination.
#
# Numba treats closure-captured Python bools as COMPILE-TIME CONSTANTS and DCE'es the unused
# branch. This lets us write ONE function body containing `if KAHAN:` checkpoints and
# specialize it into TWO compiled kernels (compensated/fast) at module import time, with
# zero runtime dispatch overhead. Single source of truth, dual implementation.
# ----------------------------------------------------------------------------------------------------------------------------


def _make_compute_simple_stats(use_kahan: bool, use_fastmath: bool):
    """Factory producing an njit-compiled simple-stats kernel; pass ``use_kahan=False`` +
    ``use_fastmath=True`` for the fast variant.

    The fast variant uses full ``fastmath=True`` (LLVM ``nnan``/``ninf`` flags ON) - it assumes
    the caller has pre-filtered NaN/inf. The Python wrapper enforces this contract via an
    ``np.isfinite(arr).all()`` gate that routes NaN-containing input to the compensated kernel.
    """
    KAHAN = use_kahan
    njit_kwargs = dict(NUMBA_NJIT_PARAMS)
    njit_kwargs["fastmath"] = use_fastmath

    @numba.njit(**njit_kwargs)
    def kernel(arr: np.ndarray) -> tuple:  # pragma: no cover
        """Compiled reduction: single pass min/max/argmin/argmax + moment accumulators, honoring the KAHAN/fastmath settings closed over above."""
        minval, maxval, argmin, argmax = 0.0, 0.0, 0, 0
        for i, next_value in enumerate(arr):
            if np.isfinite(next_value):
                minval, maxval, argmin, argmax = arr[i], arr[i], i, i
                break
        size = 0
        total, std_val = 0.0, 0.0

        for i, next_value in enumerate(arr):
            if np.isfinite(next_value):
                size += 1
                total += next_value
                # Independent if/if (not if/elif): on a flat array like [5, 5, 5, 5] the `elif`
                # branch would keep argmax at 0 (first scan) while argmin stayed at 0 too - but
                # a value equal to minval could never reach the max-update branch.
                if next_value < minval:
                    minval = next_value
                    argmin = i
                if next_value > maxval:
                    maxval = next_value
                    argmax = i

        if size == 0:
            size = len(arr)

        if size:
            mean_value = total / size
        else:
            mean_value = 0.0
            minval, maxval = 0.0, 0.0

        std_compensation = 0.0
        for _i, next_value in enumerate(arr):
            if np.isfinite(next_value):
                d = next_value - mean_value
                summand = d * d
                if KAHAN:
                    # Kahan-Babuska-Neumaier compensated accumulator. DCE'd when KAHAN=False.
                    t = std_val + summand
                    if abs(std_val) >= abs(summand):
                        std_compensation += (std_val - t) + summand
                    else:
                        std_compensation += (summand - t) + std_val
                    std_val = t
                else:
                    std_val += summand
        if size:
            std_val = np.sqrt((std_val + std_compensation) / size)
        else:
            std_val = 0.0
        return minval, maxval, argmin, argmax, mean_value, std_val

    return kernel


# Private specializations: njit-callable from other numba kernels in this module. The fast path
# is the default in the public wrapper because two-pass deviation form already gives ~14 digits
# of accuracy on float64 N<=1e7 - Kahan buys nothing observable there for ~1.4x extra cost.
_compute_simple_stats_compensated = _make_compute_simple_stats(use_kahan=True, use_fastmath=False)
_compute_simple_stats_fast = _make_compute_simple_stats(use_kahan=False, use_fastmath=True)


def compute_simple_stats_numba(arr: np.ndarray, compensated: bool = False) -> tuple:
    """Return ``(min, max, argmin, argmax, mean, std)`` over the finite elements of ``arr``.

    Parameters
    ----------
    arr
        1D numeric array. Non-finite values (NaN, +/-inf) are skipped.
    compensated
        ``False`` (default) prefers the fastmath+no-Kahan kernel, but falls back to the
        compensated kernel when ``arr`` contains any NaN/inf (the fastmath kernel sets LLVM
        ``nnan``/``ninf`` flags and would silently mishandle non-finite inputs). On all-finite
        float64 N<=1e7 the fast path is 1.3-1.5x faster and matches Kahan to 1e-12.
        ``True`` forces the Kahan kernel unconditionally - use for float32 with N>=1e6 or
        known ill-conditioned (uncentered, large-magnitude) data.

    Returns an all-zero tuple when ``arr`` contains no finite element. ``std`` is biased (ddof=0).
    """
    if compensated:
        return cast(tuple, _compute_simple_stats_compensated(arr))
    # Pre-flight finiteness check: O(N) vectorised in C, ~2us per 100k. Cheap insurance against
    # the LLVM `nnan`/`ninf` assumptions in the fast kernel.
    if not np.isfinite(arr).all():
        return cast(tuple, _compute_simple_stats_compensated(arr))
    return cast(tuple, _compute_simple_stats_fast(arr))


def compute_simple_stats_numba_arr(arr: np.ndarray, dtype=np.float32, compensated: bool = False) -> np.ndarray:
    """``compute_simple_stats_numba`` packed into an ndarray of ``dtype`` for column-stacking."""
    return np.array(compute_simple_stats_numba(arr, compensated=compensated), dtype=dtype)


def get_simple_stats_names() -> list:
    """Return the feature names matching the tuple order returned by ``compute_simple_stats_numba``."""
    return "min,max,argmin,argmax,mean,std".split(",")


# Wave 107 (2026-05-21): compute_numerical_aggregates_numba moved to
# _numerical_numba.py (along with its @numba.njit(cache=False) decorator,
# which is needed because numba's AOT cache corrupts on the function's
# bool-kwarg-heavy signature on Windows + Python 3.11 + numba 0.59).
from ._numerical_numba import compute_numerical_aggregates_numba  # noqa: F401, E402

def get_basic_feature_names(
    weights: Optional[np.ndarray] = None,
    whiten_means: bool = True,
    return_drawdown_stats: bool = False,
    return_profit_factor: bool = False,
    return_n_zer_pos_int: bool = True,
    return_exotic_means: bool = True,
    return_unsorted_stats: bool = True,
):
    """Feature names produced by ``compute_numerical_aggregates_numba`` under the same kwargs.

    Length and order are guaranteed to match the kernel's return tuple. ``has_weights`` is
    conveyed via ``weights is not None`` only - the actual weight values are not inspected here.
    """
    basic_fields = ["arimean"]
    if weights is not None:
        basic_fields.append("warimean")
    basic_fields.extend("min,max,arimean_to_first,first_to_max,min_to_first,last_to_first".split(","))

    if return_unsorted_stats:  # must be false for arrays known to be sorted
        basic_fields.append("minr")
        basic_fields.append("maxr")
        basic_fields.append("nmaxupdates")
        basic_fields.append("nminupdates")
        basic_fields.append("lastcross")
        basic_fields.append("lasttouch")

    if return_exotic_means:
        basic_fields.extend(("quadmean,qubmean,geomean,harmmean" if not whiten_means else "quadmeanw,qubmeanw,geomeanw,harmmeanw").split(","))

    if return_n_zer_pos_int:
        basic_fields.append("nonzero")
        basic_fields.append("npos")
        basic_fields.append("nint")

    if weights is not None:
        if return_exotic_means:
            basic_fields.extend(("wquadmean,wqubmean,wgeomean,wharmmean" if not whiten_means else "wquadmeanw,wqubmeanw,wgeomeanw,wharmmeanw").split(","))

    res = basic_fields.copy()

    if return_profit_factor:
        res.append("profit_factor")

    if return_drawdown_stats:
        for var in "pos_dd pos_dd_dur neg_dd neg_dd_dur".split():
            res.extend([var + "_" + field for field in basic_fields])

    return res


from ._numerical_counts import (  # noqa: F401  re-export: count / crossing / quantile kernels carved to a sibling
    _fused_nunique_modes_quantiles_kernel,
    _fused_nunique_modes_quantiles,
    compute_nunique_modes_quantiles_numpy,
    _compute_ncrossings_serial,
    _compute_ncrossings_marks_prange,
    compute_ncrossings,
    compute_nunique_mode_quantiles_numba,
)


from ._numerical_numba import (  # noqa: F401
    _make_compute_moments_slope_mi,
    compute_moments_slope_mi,
    _compute_moments_slope_mi_compensated,
    _compute_moments_slope_mi_fast,
    _EMPTY_FLOAT32,
)

def compute_mutual_info_regression(arr: np.ndarray, xvals: np.ndarray = _EMPTY_FLOAT32) -> float:
    """KNN-based mutual information between ``arr`` and ``xvals`` (or ``arr``'s row index when ``xvals`` is empty)."""
    if len(xvals):
        mi = mutual_info_regression(xvals.reshape(-1, 1), arr, n_neighbors=2)
    else:
        mi = mutual_info_regression(np.arange(len(arr)).reshape(-1, 1), arr, n_neighbors=2)

    return float(mi[0])


def compute_entropy_features(arr: np.ndarray, sampling_frequency: int = 100, spectral_method: str = "welch") -> tuple:
    """Apply every function in ``entropy_funcs`` to ``arr`` and return their outputs as a tuple.

    Non-finite values are stripped first; if fewer than 2 finite samples remain, returns zeros.
    Inf/NaN outputs are collapsed to 0 via ``np.nan_to_num``.
    """
    # hjorth_mobility, hjorth_complexity = hjorth_params(arr)
    # hjorth_mobility,
    # hjorth_complexity,
    # spectral_entropy(arr, sf=sampling_frequency, method=spectral_method),)
    # num_zerocross(arr),

    _entropy_funcs = _get_entropy_funcs()
    nonzero = (~np.isnan(arr)).sum()
    if nonzero < 10:
        return (0.0,) * len(_entropy_funcs)
    else:
        safe_arr = np.nan_to_num(arr[~np.isnan(arr)], posinf=0, neginf=0)
        return tuple(np.nan_to_num([f(safe_arr) for f in _entropy_funcs], posinf=0, neginf=0))


def fit_distribution(dist: rv_continuous, data: np.ndarray, method: str = "mle"):
    """Fit a scipy distribution to ``data`` and return parameters + Kolmogorov-Smirnov fit quality.

    Returns the distribution's fitted parameters concatenated with ``(ks_stat, ks_pval)``. On any
    exception during fitting, returns the predefined fallback tuple from ``default_dist_responses``.
    """
    try:
        params = dist.fit(data, method=method)
    except Exception:
        # Documented fallback for a failed scipy fit; log at DEBUG so a genuine bug (vs an unfittable sample)
        # is diagnosable rather than silently returned as the NaN fallback.
        logger.debug("fit_distribution: %s fit failed; returning fallback", getattr(dist, "name", "?"), exc_info=True)
        return default_dist_responses[dist.name] + (np.nan, np.nan)
    else:
        dist_fitted = dist(*params)
        ks_stat, ks_pval = kstest(data, dist_fitted.cdf)

        return tuple(np.nan_to_num([*params, ks_stat, ks_pval], posinf=0, neginf=0))


def compute_distributional_features(arr: np.ndarray) -> tuple:
    """Concatenate ``fit_distribution`` outputs across the module-level ``distributions`` tuple.

    Order and length must match ``get_distributions_features_names()``.
    """
    # Build via list+extend then convert once - O(n) vs the previous tuple-concat-in-loop which
    # is O(n^2) over the number of distributions. Current `distributions` has 1 entry so the
    # impact is moot today, but the pattern stays correct as the roster grows.
    parts: list = []
    for dist in distributions:
        parts.extend(fit_distribution(dist=dist, data=arr))
    return tuple(parts)


def compute_numaggs(
    arr: np.ndarray,
    xvals: Optional[np.ndarray] = None,
    weights: Optional[np.ndarray] = None,
    geomean_log_mode: bool = False,
    q: Sequence[float] = default_quantiles,
    quantile_method: str = "median_unbiased",
    max_modes: int = 10,
    sampling_frequency: int = 100,
    spectral_method: str = "welch",
    hurst_kwargs: Optional[dict] = None,
    directional_only: bool = False,
    whiten_means: bool = True,
    return_distributional: bool = False,
    return_entropy: bool = True,
    return_hurst: bool = True,
    return_float32: bool = True,
    return_profit_factor: bool = False,
    return_drawdown_stats: bool = False,
    return_n_zer_pos_int: bool = True,
    return_exotic_means: bool = True,
    return_unsorted_stats: bool = True,
    return_lintrend_approx_stats: bool = False,
    compensated: bool = False,
):
    """Compute a plethora of numerical aggregates for all values in an array.

    Converts an arbitrarily-length array into a fixed number of aggregates.

    ``compensated=False`` (default) routes the inner moment kernels through the fast (no Kahan,
    fastmath=True) path - identical to the compensated path within 1e-5 on well-conditioned
    float64. Set ``True`` for float32 with N>=1e6 or known ill-conditioned data.
    """
    if hurst_kwargs is None:
        hurst_kwargs = dict(min_window=10, max_window=None, windows_log_step=0.25, take_diffs=False)
    if len(arr) <= 1:
        return (np.nan,) * len(
            get_numaggs_names(
                weights=weights,
                q=q,
                directional_only=directional_only,
                whiten_means=whiten_means,
                return_distributional=return_distributional,
                return_entropy=return_entropy,
                return_hurst=return_hurst,
                return_profit_factor=return_profit_factor,
                return_drawdown_stats=return_drawdown_stats,
                return_n_zer_pos_int=return_n_zer_pos_int,
                return_exotic_means=return_exotic_means,
                return_unsorted_stats=return_unsorted_stats,
                return_lintrend_approx_stats=return_lintrend_approx_stats,
            )
        )

    res = compute_numerical_aggregates_numba(
        arr,
        weights=weights,
        geomean_log_mode=geomean_log_mode,
        directional_only=directional_only,
        whiten_means=whiten_means,
        return_profit_factor=return_profit_factor,
        return_drawdown_stats=return_drawdown_stats,
        return_n_zer_pos_int=return_n_zer_pos_int,
        return_exotic_means=return_exotic_means,
        return_unsorted_stats=return_unsorted_stats,
    )

    arithmetic_mean = res[0]
    if weights is not None:
        weighted_arithmetic_mean = res[1]
    else:
        weighted_arithmetic_mean = 0.0

    res = tuple(res)
    if not directional_only:
        res = res + compute_nunique_modes_quantiles_numpy(
            arr=arr, q=q, quantile_method=quantile_method, max_modes=max_modes, return_unsorted_stats=return_unsorted_stats
        )

    moments_slope_mi, lintrend_data_diffs = compute_moments_slope_mi(
        arr=arr,
        weights=weights,
        mean_value=arithmetic_mean,
        weighted_mean_value=weighted_arithmetic_mean,
        xvals=xvals,
        directional_only=directional_only,
        return_lintrend_approx_stats=return_lintrend_approx_stats,
        compensated=compensated,
    )
    res = res + tuple(moments_slope_mi)

    if return_hurst:
        res = res + compute_hurst_exponent(arr=arr, **hurst_kwargs)

    if not (directional_only or not return_entropy):
        res = res + compute_entropy_features(arr=arr, sampling_frequency=sampling_frequency, spectral_method=spectral_method)
    if return_distributional:
        res = res + tuple(compute_distributional_features(arr=arr))
    if return_lintrend_approx_stats:
        params: dict[str, Any] = dict(
            weights=weights,
            geomean_log_mode=geomean_log_mode,
            q=q,
            quantile_method=quantile_method,
            max_modes=max_modes,
            sampling_frequency=sampling_frequency,
            spectral_method=spectral_method,
            hurst_kwargs=hurst_kwargs,
            directional_only=directional_only,
            whiten_means=whiten_means,
            return_distributional=return_distributional,
            return_entropy=return_entropy,
            return_hurst=return_hurst,
            return_float32=True,
            return_profit_factor=False,
            return_drawdown_stats=False,
            return_n_zer_pos_int=return_n_zer_pos_int,
            return_exotic_means=return_exotic_means,
            return_unsorted_stats=return_unsorted_stats,
            return_lintrend_approx_stats=False,
        )
        res = res + tuple(compute_numaggs(arr=lintrend_data_diffs, xvals=xvals, **params))

    if return_float32:
        return np.array(res, dtype=np.float32)
    else:
        return res


def get_numaggs_names(
    weights: Optional[np.ndarray] = None,
    q: Sequence[float] = default_quantiles,
    directional_only: bool = False,
    whiten_means: bool = True,
    return_distributional: bool = False,
    return_entropy: bool = True,
    return_hurst: bool = True,
    return_profit_factor: bool = False,
    return_drawdown_stats: bool = False,
    return_n_zer_pos_int: bool = True,
    return_exotic_means: bool = True,
    return_unsorted_stats: bool = True,
    return_lintrend_approx_stats: bool = False,
    **kwargs,
) -> tuple:
    """Feature names produced by ``compute_numaggs`` under the same kwargs.

    Length and order are guaranteed to match the values tuple from ``compute_numaggs``. Excess
    ``**kwargs`` are accepted (but ignored) so callers can pass the same dict to both functions
    without filtering it.
    """
    res = tuple(
        (
            ["arimean", "ratio"]
            if directional_only
            else get_basic_feature_names(
                weights=weights,
                whiten_means=whiten_means,
                return_profit_factor=return_profit_factor,
                return_drawdown_stats=return_drawdown_stats,
                return_n_zer_pos_int=return_n_zer_pos_int,
                return_exotic_means=return_exotic_means,
                return_unsorted_stats=return_unsorted_stats,
            )
        )
        + ([] if (directional_only or not return_unsorted_stats) else "nuniques,modmin,modmax,modmean,modqty".split(","))
        + ([] if directional_only else (["q" + str(q) for q in q]))
        + ([] if (directional_only or not return_unsorted_stats) else ["ncrs" + str(q) for q in q])
        + get_moments_slope_mi_feature_names(weights=weights, directional_only=directional_only, return_lintrend_approx_stats=return_lintrend_approx_stats)
        + (["hursth", "hurstc"] if return_hurst else [])
        + ([] if (directional_only or not return_entropy) else list(_ENTROPY_FUNC_NAMES))
        + (distributions_features_names if return_distributional else [])
    )

    if return_lintrend_approx_stats:
        params: dict[str, Any] = dict(
            weights=weights,
            q=q,
            directional_only=directional_only,
            whiten_means=whiten_means,
            return_distributional=return_distributional,
            return_entropy=return_entropy,
            return_hurst=return_hurst,
            return_float32=True,
            return_profit_factor=False,
            return_drawdown_stats=False,
            return_n_zer_pos_int=return_n_zer_pos_int,
            return_exotic_means=return_exotic_means,
            return_unsorted_stats=return_unsorted_stats,
            return_lintrend_approx_stats=False,
        )
        res += tuple(["ltrappr_" + el for el in get_numaggs_names(**params)])
    return res


def get_moments_slope_mi_feature_names(weights: Optional[np.ndarray] = None, directional_only: bool = False, return_lintrend_approx_stats: bool = True):
    """Feature names produced by ``compute_moments_slope_mi``. ``weights is not None`` toggles weighted variants."""
    res = []
    if not directional_only:
        res.extend("mad,std,skew,kurt".split(","))
        if weights is not None:
            res.extend("wmad,wstd,wskew,wkurt".split(","))
    res.extend("slope,intercept,r,meancross,trendcross".split(","))
    return res


# fastmath=False is REQUIRED for the Kahan path: the compensator relies on the exact identity
# `c = (t - sum_window) - y` where `t = sum_window + y`. With fastmath, LLVM is allowed to
# reassociate `(t - sum_window) - y` -> `t - (sum_window + y)` -> `t - t` -> `0`, nullifying
# the compensator and reverting the loop to naive sliding-window drift (~n*eps*max_value, several
# decimal digits on float32 windows). MEASURED cost of disabling fastmath here: ~260% slowdown
# on Windows / numba 0.59 (LLVM also SIMD-vectorises the cumsum once Kahan is folded out).
@numba.njit(fastmath=False, cache=True)
def _rolling_moving_average_compensated(arr: np.ndarray, n: int) -> np.ndarray:  # pragma: no cover
    """Kahan-Babuska-Neumaier compensated rolling mean. Slow but precision-stable."""
    result = np.empty(len(arr) - n + 1, dtype=arr.dtype)
    sum_window = np.sum(arr[:n])
    mult = 1 / n
    result[0] = sum_window * mult
    kahan_c = 0.0
    for i in range(1, len(arr) - n + 1):
        y = (arr[i + n - 1] - arr[i - 1]) - kahan_c
        t = sum_window + y
        kahan_c = (t - sum_window) - y
        sum_window = t
        result[i] = sum_window * mult
    return result


@numba.njit(fastmath=True, cache=True)
def _rolling_moving_average_fast(arr: np.ndarray, n: int) -> np.ndarray:  # pragma: no cover
    """Plain sliding-window rolling mean. ~3.5x faster than the compensated variant.

    Use when the input is well-conditioned (magnitudes within ~6 orders of magnitude on float64,
    ~3 on float32) and accumulated rounding drift is acceptable. Both variants have identical
    output to within ~n*eps*max(|x|), but on long ill-conditioned windows (e.g. 1e9 + N(0,1))
    the fast path can lose 7+ digits while the compensated path stays at 1 ULP.
    """
    result = np.empty(len(arr) - n + 1, dtype=arr.dtype)
    sum_window = np.sum(arr[:n])
    mult = 1 / n
    result[0] = sum_window * mult
    for i in range(1, len(arr) - n + 1):
        sum_window = sum_window + (arr[i + n - 1] - arr[i - 1])
        result[i] = sum_window * mult
    return result


def rolling_moving_average(arr: np.ndarray, n: int = 2, compensated: bool = True) -> np.ndarray:
    """Rolling mean over a 1D array. Length of the return is ``len(arr) - n + 1``.

    Parameters
    ----------
    arr
        1D numeric array. dtype is preserved on output.
    n
        Window size (>= 1, <= len(arr)).
    compensated
        ``True`` (default) uses Kahan-Babuska-Neumaier compensated summation: precision-stable
        on ill-conditioned and float32 inputs, ~3.5x slower (measured on Windows / numba 0.59).
        ``False`` uses a plain sliding-window sum with fastmath enabled: faster but accumulates
        ~n*eps*max(|x|) rounding drift, which can be ~7 digits on long ill-conditioned windows.
        Pick False when downstream features are robust to that drift (and you're not on float32
        with a large additive offset).

    Raises
    ------
    ValueError
        If ``n <= 0`` or ``n > len(arr)``.
    """
    if n <= 0:
        raise ValueError("n must be greater than 0")
    if n > len(arr):
        raise ValueError("n must be less than or equal to the length of the array")
    if compensated:
        return cast(np.ndarray, _rolling_moving_average_compensated(arr, n))
    return cast(np.ndarray, _rolling_moving_average_fast(arr, n))


def numaggs_over_matrix_rows(vals: np.ndarray, numagg_params: dict, rolling_ma: int = 0, use_diffs: bool = False, dtype=np.float32) -> np.ndarray:
    """Compute numaggs over rows of a 2D matrix and stack into a (n_rows, n_features) array."""

    # Local copy so the caller's dict is not mutated by the `return_float32` override below.
    numagg_params = dict(numagg_params)
    numagg_params["return_float32"] = True
    feature_names = get_numaggs_names(**numagg_params)

    res = np.empty(shape=(len(vals), len(feature_names)), dtype=dtype)

    for i in range(vals.shape[0]):
        vals_to_compute = vals[i, :]

        if rolling_ma:
            if use_diffs:
                vals_to_compute = np.nan_to_num(vals_to_compute[rolling_ma - 1 :] / rolling_moving_average(vals_to_compute, rolling_ma) - 1, posinf=0, neginf=0)
            else:
                vals_to_compute = vals_to_compute[rolling_ma - 1 :] - rolling_moving_average(vals_to_compute, rolling_ma)
        else:
            if use_diffs:
                vals_to_compute = np.nan_to_num((vals_to_compute[1:] / vals_to_compute[:-1] - 1), posinf=0, neginf=0)

        res[i, :] = compute_numaggs(vals_to_compute, **numagg_params)
    return res


def compute_numaggs_parallel(
    df: pd.DataFrame = None,
    cols: Optional[Sequence] = None,
    values: Optional[np.ndarray] = None,
    use_diffs: bool = False,
    rolling_ma: int = 0,
    numagg_params: Optional[dict] = None,
    dtype=np.float32,
    n_jobs=-1,
    prefetch_factor: int = 2,
    **parallel_kwargs,
) -> np.ndarray:
    """Computes numaggs over columns of a dataframe, in parallel fashion.

    Example of parallel_kwargs: ``compute_numaggs_parallel(df=X, cols=cols, temp_folder=<temp>)``.
    Exactly one of ``values`` or ``(df, cols)`` must be supplied.
    """
    if numagg_params is None:
        numagg_params = {}
    if values is None and (df is None or cols is None):
        raise ValueError("compute_numaggs_parallel: must provide either `values` or both `df` and `cols`.")
    if n_jobs <= 0:
        # psutil.cpu_count can return None on container/restricted hosts; fall back to logical.
        n_jobs = psutil.cpu_count(logical=False) or psutil.cpu_count(logical=True) or 1

    if values is None:
        values = df.loc[:, cols].values

    res = parallel_run(
        [
            delayed(numaggs_over_matrix_rows)(vals=chunk, use_diffs=use_diffs, rolling_ma=rolling_ma, numagg_params=numagg_params)
            for chunk in np.array_split(values, n_jobs * prefetch_factor)
        ],
        max_nbytes=0,
        n_jobs=n_jobs,
        **parallel_kwargs,
    )
    res = np.vstack(res).astype(dtype)

    return res
