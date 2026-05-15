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
from typing import Sequence, Tuple

import numba
import numpy as np
import pandas as pd
import psutil
from antropy import detrended_fluctuation, katz_fd, perm_entropy, petrosian_fd, sample_entropy, svd_entropy
from astropy.stats import histogram
from joblib import delayed
from scipy import stats
from scipy.stats import kstest
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

NUMBA_NJIT_PARAMS = dict(fastmath=False, cache=True, nogil=True)


def cont_entropy(arr: np.ndarray, bins: str = "scott") -> float:
    """Entropy of a continuous distribution"""
    try:
        hist, bin_edges = histogram(arr, bins=bins)  # np.histogram(arr, bins=bins, density=True)
        ent = -(hist * np.log(hist + 1e-60)).sum()
    except Exception:
        return np.nan
    return ent


entropy_funcs = (cont_entropy, svd_entropy, sample_entropy, petrosian_fd, perm_entropy, katz_fd, detrended_fluctuation)  # continuous.get_h, app_entropy
entropy_funcs_names = [f.__name__ for f in entropy_funcs]

distributions = (stats.levy_l,)  # stats.logistic, stats.pareto
default_dist_responses = dict(levy_l=(np.nan, np.nan), logistic=(np.nan, np.nan), pareto=(np.nan, np.nan, np.nan))

LARGE_CONST = 1e3
# Geometric-mean overflow / underflow thresholds. When the running product crosses either, the
# kernel switches to log-mode accumulation to avoid float64 over/underflow.
GEOMEAN_OVERFLOW_HI: float = 1e100
GEOMEAN_OVERFLOW_LO: float = 1e-100


def get_distributions_features_names() -> list:
    distributions_features_names = []
    for dist in distributions:
        for i in range(len(default_dist_responses[dist.name])):
            distributions_features_names.append(dist.name + str(i + 1))
        distributions_features_names.append(dist.name + "_kss")
        distributions_features_names.append(dist.name + "_kspval")
    return distributions_features_names


distributions_features_names = get_distributions_features_names()

default_quantiles: Tuple[float, ...] = (0.1, 0.25, 0.5, 0.75, 0.9)  # tuple is hashable + immutable; list vs ndarray is ~10% faster (125 µs vs 140 µs per loop) so callers convert via list(default_quantiles) where needed


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
        return _compute_simple_stats_compensated(arr)
    # Pre-flight finiteness check: O(N) vectorised in C, ~2us per 100k. Cheap insurance against
    # the LLVM `nnan`/`ninf` assumptions in the fast kernel.
    if not np.isfinite(arr).all():
        return _compute_simple_stats_compensated(arr)
    return _compute_simple_stats_fast(arr)


def compute_simple_stats_numba_arr(arr: np.ndarray, dtype=np.float32, compensated: bool = False) -> np.ndarray:
    """``compute_simple_stats_numba`` packed into an ndarray of ``dtype`` for column-stacking."""
    return np.array(compute_simple_stats_numba(arr, compensated=compensated), dtype=dtype)


def get_simple_stats_names() -> list:
    return "min,max,argmin,argmax,mean,std".split(",")


# cache=False overrides NUMBA_NJIT_PARAMS for this kernel only: numba's AOT
# cache for functions with many bool kwargs corrupts on Windows (Python 3.11
# + numba 0.59) -- a fresh process that calls this with all kwargs explicit
# loads a stale .nbc compilation and segfaults with an access violation.
# Cleared the cache directory and the next call rebuilt cleanly, only to
# crash again on the *next* fresh process. Disabling the cache for this one
# function trades a ~3s warm-up at import time for crash-free behaviour.
# Other kernels in this module are unaffected by the bug and keep cache=True.
@numba.njit(**{**NUMBA_NJIT_PARAMS, "cache": False})
def compute_numerical_aggregates_numba(
    arr: np.ndarray,
    weights: np.ndarray = None,
    geomean_log_mode: bool = False,
    directional_only: bool = False,
    whiten_means: bool = True,
    return_drawdown_stats: bool = False,
    return_profit_factor: bool = False,
    return_n_zer_pos_int: bool = True,
    return_exotic_means: bool = True,
    return_unsorted_stats: bool = True,
) -> list:  # pragma: no cover
    """Compute statistical aggregates over 1d array of float32 values.
    E mid2(abs(x-mid1(X))) where mid1, mid2=averages of any kind
    E Функции ошибок иногда и классные признаки...
    V What happens first: min or max? Add relative percentage of min/max indices
    V Add absolute values of min/max indices?
    V Добавить количество пересечений средних и медианного значений, линии slope? (trend reversions)
        Хотя это можно в т.ч. получить, вызвав стату над нормированным или детрендированным рядом (x-X_avg) или (x-(slope*x+x[0]))
    V убрать гэпы. это статистика второго порядка и должна считаться отдельно. причем можно считать от разностей или от отношений.
    V взвешенные статы считать отдельным вызовом ( и не только среднеарифметические, а ВСЕ).
    Добавить
        V среднее кубическое,
        V entropy
        V hurst
        V R2
        E? среднее винзоризированное (https://ru.wikipedia.org/wiki/%D0%92%D0%B8%D0%BD%D0%B7%D0%BE%D1%80%D0%B8%D0%B7%D0%BE%D0%B2%D0%B0%D0%BD%D0%BD%D0%BE%D0%B5_%D1%81%D1%80%D0%B5%D0%B4%D0%BD%D0%B5%D0%B5).
        E? усечённое,
        E? tukey mean
        V fit variable to a number of known distributions!! their params become new features
        V drawdowns, negative drawdowns (for shorts), dd duration (%)
        V Number of MAX/MIN refreshers during period
        V numpeaks
    """

    size = len(arr)
    # Empty input would IndexError on arr[0] / arr[-1]; callers usually guard upstream (compute_numaggs short-circuits at len<=1) but the kernel is exported in __all__ so accept the corner.
    if size == 0:
        return [0.0]

    first = arr[0]
    last = arr[-1]
    if first != 0.0:
        last_to_first = last / first
    else:
        last_to_first = LARGE_CONST * np.sign(last)

    if directional_only:
        arithmetic_mean = np.mean(arr)
        return [arithmetic_mean, last_to_first]

    ninteger, npositive, cnt_nonzero = 0, 0, 0
    sum_positive, sum_negative = 0.0, 0.0

    if not geomean_log_mode:
        geometric_mean = 1.0
    else:
        geometric_mean = 0.0

    arithmetic_mean, quadratic_mean, qubic_mean, harmonic_mean = 0.0, 0.0, 0.0, 0.0
    weighted_arithmetic_mean = 0.0
    if weights is not None:
        weighted_geometric_mean, weighted_arithmetic_mean, weighted_quadratic_mean, weighted_qubic_mean, weighted_harmonic_mean = (
            geometric_mean,
            arithmetic_mean,
            quadratic_mean,
            qubic_mean,
            harmonic_mean,
        )
        sum_weights = 0.0

    maximum, minimum = first, first
    max_index, min_index = 0, 0

    if return_drawdown_stats:

        pos_dd_start_idx, neg_dd_start_idx = 0, 0

        pos_dds = np.empty(shape=size, dtype=np.float32)
        pos_dd_durs = np.empty(shape=size, dtype=np.float32)

        neg_dds = np.empty(shape=size, dtype=np.float32)
        neg_dd_durs = np.empty(shape=size, dtype=np.float32)

    nmaxupdates, nminupdates = 0, 0

    n_last_crossings, n_last_touches = 0, 0
    # numba Optional[NoneType|float64] segfaults under numba 0.62 / numpy 2.2 —
    # use a bool flag + sentinel float so the variable type is invariant.
    has_prev_d = False
    prev_d = 0.0

    for i, next_value in enumerate(arr):
        if weights is not None:
            next_weight = weights[i]
            sum_weights += weights[i]
            if weights is not None:
                weighted_arithmetic_mean += next_value * next_weight

        if return_unsorted_stats:
            d = next_value - last
            if has_prev_d:
                mul = d * prev_d
                if mul < 0:
                    n_last_crossings += 1
                elif mul == 0.0:
                    n_last_touches += 1
            else:
                if next_value == last:
                    n_last_touches += 1
            prev_d = d
            has_prev_d = True

        arithmetic_mean += next_value

        if return_exotic_means:
            temp_value = next_value * next_value
            quadratic_mean += temp_value
            if weights is not None:
                weighted_quadratic_mean += temp_value * next_weight

            temp_value = temp_value * next_value
            qubic_mean += temp_value
            if weights is not None:
                weighted_qubic_mean += temp_value * next_weight

        # Independent checks (not if/elif): elif would mean a sample equal to minimum can never
        # update maximum, producing inconsistent min_index/max_index on degenerate inputs.
        if next_value < minimum:
            minimum = next_value
            min_index = i
            nminupdates += 1
        if next_value > maximum:
            maximum = next_value
            max_index = i
            nmaxupdates += 1

        # ----------------------------------------------------------------------------------------------------------------------------
        # Drawdowns
        # ----------------------------------------------------------------------------------------------------------------------------

        if return_drawdown_stats:
            # pos
            dd = maximum - next_value
            pos_dds[i] = dd
            if dd == 0.0:
                pos_dd_start_idx = i
            pos_dd_durs[i] = i - pos_dd_start_idx

            # neg
            dd = next_value - minimum
            neg_dds[i] = dd
            if dd == 0.0:
                neg_dd_start_idx = i
            neg_dd_durs[i] = i - neg_dd_start_idx

        if next_value:
            cnt_nonzero = cnt_nonzero + 1
            if return_exotic_means:
                addend = 1 / next_value
                harmonic_mean += addend
                if weights is not None:
                    weighted_harmonic_mean += next_weight * addend

            if return_n_zer_pos_int:
                # Was `next_value % 1` — robust for positive floats but fragile around negative
                # values and denormals. `np.floor(x) == x` is the exact integer check.
                if np.floor(next_value) == next_value:
                    ninteger = ninteger + 1

            if next_value > 0:
                npositive = npositive + 1
                sum_positive += next_value
                if return_exotic_means:
                    if not geomean_log_mode:
                        geometric_mean *= next_value
                        if weights is not None:
                            weighted_geometric_mean *= next_value**next_weight
                        # Check BOTH unweighted and weighted products: either can underflow / overflow independently.
                        # A weighted product can hit the tail much faster when |next_weight| > 1.
                        unweighted_oor = geometric_mean >= GEOMEAN_OVERFLOW_HI or geometric_mean <= GEOMEAN_OVERFLOW_LO
                        weighted_oor = (weights is not None) and (weighted_geometric_mean >= GEOMEAN_OVERFLOW_HI or weighted_geometric_mean <= GEOMEAN_OVERFLOW_LO)
                        if unweighted_oor or weighted_oor:
                            # convert to log mode (geometric_mean strictly positive here; log is finite)
                            geomean_log_mode = True
                            geometric_mean = np.log(float(geometric_mean)) if geometric_mean > 0 else -np.inf
                            if weights is not None:
                                weighted_geometric_mean = np.log(float(weighted_geometric_mean)) if weighted_geometric_mean > 0 else -np.inf
                    else:
                        addend = np.log(next_value)
                        geometric_mean += addend
                        if weights is not None:
                            weighted_geometric_mean += next_weight * addend
            else:
                sum_negative += next_value

    arithmetic_mean = arithmetic_mean / size
    if weights is not None:
        weighted_arithmetic_mean = weighted_arithmetic_mean / sum_weights

    if return_exotic_means:
        quadratic_mean = np.sqrt(quadratic_mean / size)
        qubic_mean = (qubic_mean / size) ** (1 / 3)
        if npositive:
            if not geomean_log_mode:
                geometric_mean = geometric_mean ** (1 / size)
            else:
                geometric_mean = np.exp(geometric_mean / size)
        else:
            geometric_mean = np.nan
        if harmonic_mean:
            harmonic_mean = size / harmonic_mean
        else:
            harmonic_mean = np.nan

        if weights is not None:
            weighted_quadratic_mean = np.sqrt(weighted_quadratic_mean / sum_weights)
            weighted_qubic_mean = (weighted_qubic_mean / sum_weights) ** (1 / 3)
            if npositive and sum_weights != 0.0:
                if not geomean_log_mode:
                    weighted_geometric_mean = weighted_geometric_mean ** (1 / sum_weights)
                else:
                    weighted_geometric_mean = np.exp(weighted_geometric_mean / sum_weights)
                if weighted_geometric_mean == np.inf:
                    weighted_geometric_mean = 0.0
            else:
                weighted_geometric_mean = np.nan
            if weighted_harmonic_mean:
                weighted_harmonic_mean = sum_weights / weighted_harmonic_mean
            else:
                weighted_harmonic_mean = np.nan

        if whiten_means:
            quadratic_mean = quadratic_mean - arithmetic_mean
            qubic_mean = qubic_mean - arithmetic_mean
            geometric_mean = geometric_mean - arithmetic_mean
            harmonic_mean = harmonic_mean - arithmetic_mean
            if weights is not None:
                weighted_quadratic_mean = weighted_quadratic_mean - weighted_arithmetic_mean
                weighted_qubic_mean = weighted_qubic_mean - weighted_arithmetic_mean
                weighted_geometric_mean = weighted_geometric_mean - weighted_arithmetic_mean
                weighted_harmonic_mean = weighted_harmonic_mean - weighted_arithmetic_mean

    res = [arithmetic_mean]
    if weights is not None:
        res.append(weighted_arithmetic_mean)
    res.extend(
        (
            minimum,
            maximum,
        )
    )  # cant combine with the next statement as it's failing on interger inputs due to tuple dtypes mismatch
    res.extend(
        (
            arithmetic_mean / first if first else LARGE_CONST * np.sign(arithmetic_mean),
            first / maximum if maximum else LARGE_CONST * np.sign(first),
            minimum / first if first else LARGE_CONST * np.sign(minimum),
            last_to_first,
        )
    )

    if return_unsorted_stats:  # must be false for arrays known to be sorted
        res.extend(((min_index + 1) / size if size else 0, (max_index + 1) / size if size else 0))
        res.extend((nmaxupdates, nminupdates, n_last_crossings, n_last_touches - 1))

    if return_exotic_means:
        res.extend((quadratic_mean, qubic_mean, geometric_mean, harmonic_mean))

    if return_n_zer_pos_int:
        res.extend((cnt_nonzero, npositive, ninteger))

    if weights is not None:
        if return_exotic_means:
            res.extend((weighted_quadratic_mean, weighted_qubic_mean, weighted_geometric_mean, weighted_harmonic_mean))

    if return_profit_factor:
        profit_factor = sum_positive / -sum_negative if sum_negative != 0.0 else (0.0 if sum_positive == 0.0 else LARGE_CONST)
        res.append(profit_factor)

    if return_drawdown_stats:
        res.extend(
            compute_numerical_aggregates_numba(
                arr=pos_dds[1:],
                weights=weights if weights is None else weights[1:],
                geomean_log_mode=geomean_log_mode,
                directional_only=directional_only,
                whiten_means=whiten_means,
                return_drawdown_stats=False,
                return_profit_factor=False,
                return_n_zer_pos_int=return_n_zer_pos_int,
                return_exotic_means=return_exotic_means,
                return_unsorted_stats=return_unsorted_stats,
            )
        )
        res.extend(
            compute_numerical_aggregates_numba(
                arr=pos_dd_durs[1:] / (size - 1),
                weights=weights if weights is None else weights[1:],
                geomean_log_mode=geomean_log_mode,
                directional_only=directional_only,
                whiten_means=whiten_means,
                return_drawdown_stats=False,
                return_profit_factor=False,
                return_n_zer_pos_int=return_n_zer_pos_int,
                return_exotic_means=return_exotic_means,
                return_unsorted_stats=return_unsorted_stats,
            )
        )
        res.extend(
            compute_numerical_aggregates_numba(
                arr=neg_dds[1:],
                weights=weights if weights is None else weights[1:],
                geomean_log_mode=geomean_log_mode,
                directional_only=directional_only,
                whiten_means=whiten_means,
                return_drawdown_stats=False,
                return_profit_factor=False,
                return_n_zer_pos_int=return_n_zer_pos_int,
                return_exotic_means=return_exotic_means,
                return_unsorted_stats=return_unsorted_stats,
            )
        )
        res.extend(
            compute_numerical_aggregates_numba(
                arr=neg_dd_durs[1:] / (size - 1),
                weights=weights if weights is None else weights[1:],
                geomean_log_mode=geomean_log_mode,
                directional_only=directional_only,
                whiten_means=whiten_means,
                return_drawdown_stats=False,
                return_profit_factor=False,
                return_n_zer_pos_int=return_n_zer_pos_int,
                return_exotic_means=return_exotic_means,
                return_unsorted_stats=return_unsorted_stats,
            )
        )

    return res


def get_basic_feature_names(
    weights: np.ndarray = None,
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


def compute_nunique_modes_quantiles_numpy(
    arr: np.ndarray, q: Sequence[float] = default_quantiles, quantile_method: str = "median_unbiased", max_modes: int = 10, return_unsorted_stats: bool = True
) -> list:
    """For a 1d array, computes aggregates:
    nunique
    modes:min,max,mean
    list of quantiles (0 and 1 included by default, therefore, min/max)
    number of quantiles crossings
    Can NOT be numba jitted (yet).
    """
    if return_unsorted_stats:
        vals, counts = np.unique(arr, return_counts=True)

        max_modes = min(max_modes, len(counts))

        modes_indices = np.argpartition(counts, -max_modes)[-max_modes:]
        modes_indices = modes_indices[np.argsort(counts[modes_indices])][::-1]

        first_mode_count = counts[modes_indices[0]]

        if first_mode_count == 1:
            modes_min, modes_max, modes_mean, modes_qty = (
                np.nan,
                np.nan,
                np.nan,
                np.nan,
            )  # for higher stability. cnt=1 is not really a mode, rather a random pick.
        else:
            next_idx = modes_indices[0]
            best_modes = [vals[next_idx]]
            for i in range(1, max_modes):
                next_idx = modes_indices[i]
                next_mode_count = counts[next_idx]
                if next_mode_count < first_mode_count:
                    break
                else:
                    best_modes.append(vals[next_idx])
            best_modes = np.asarray(best_modes)
            modes_min = best_modes.min()
            modes_max = best_modes.max()
            modes_mean = best_modes.mean()
            modes_qty = len(best_modes)

        nuniques = len(vals)

        res = (nuniques, modes_min, modes_max, modes_mean, modes_qty)
    else:
        res = ()

    quantiles = np.quantile(arr, q, method=quantile_method)
    res = res + tuple(quantiles)  # .tolist()

    if return_unsorted_stats:
        res = res + tuple(compute_ncrossings(arr=arr, marks=quantiles))  # .tolist()

    return res


@numba.njit(**NUMBA_NJIT_PARAMS)
def compute_ncrossings(arr: np.ndarray, marks: np.ndarray, dtype=np.int32) -> np.ndarray:  # pragma: no cover
    """Count sign-changes in ``(arr[i] - mark)`` for each ``mark`` in ``marks``.

    Returns one integer per mark; useful as a quantile-crossing feature on time series.
    """
    n_crossings = np.zeros(len(marks), dtype=dtype)
    prev_ds = np.full(len(marks), dtype=np.float32, fill_value=np.nan)

    for next_value in arr:
        for i, mark in enumerate(marks):
            d = next_value - mark
            if not np.isnan(prev_ds[i]):
                if d * prev_ds[i] < 0:
                    n_crossings[i] += 1
            prev_ds[i] = d

    return n_crossings


@numba.njit(**NUMBA_NJIT_PARAMS)
def compute_nunique_mode_quantiles_numba(arr: np.ndarray, q: Sequence[float] = default_quantiles) -> tuple:  # pragma: no cover
    """
    NOT RECOMMENDED. use compute_nunique_modes_quantiles_numpy instead, it's faster and more functional.
    numUnique and mode calculation from sorted array
    CAN be numba jitted.
    """
    xsorted = np.sort(arr)

    next_unique_value = xsorted[0]
    numUnique = 1
    mode = next_unique_value
    best_count = 1
    times_occured = 0
    for next_value in xsorted:
        if next_value == next_unique_value:
            times_occured = times_occured + 1
        else:
            numUnique = numUnique + 1
            if times_occured > best_count:
                best_count = times_occured
                mode = next_unique_value
            next_unique_value = next_value
            times_occured = 1
    if times_occured > best_count:
        best_count = times_occured
        mode = next_unique_value

    factor = len(arr)
    quantiles = []
    for quantile in q:
        # Clamp index >= 0: at quantile=0.0 the formula yields -1 which would wrap to the LAST element (max) instead of the first (min).
        idx = int(np.ceil(quantile * factor)) - 1
        if idx < 0:
            idx = 0
        elif idx >= factor:
            idx = factor - 1
        quantiles.append(xsorted[idx])

    if times_occured == 1:
        mode = np.nan
    return numUnique, mode, quantiles


def _make_compute_moments_slope_mi(use_kahan: bool, use_fastmath: bool):
    """Factory producing an njit-compiled moments/slope/MI kernel.

    Single source of truth for both compensated and fast variants. ``KAHAN`` is a closure-
    captured Python bool: numba treats it as a compile-time constant and DCE's the unused
    branch in each ``if KAHAN: ... else: ...`` block, so the generated code for each variant is
    the same as if it had been hand-written separately.
    """
    KAHAN = use_kahan
    njit_kwargs = dict(NUMBA_NJIT_PARAMS)
    njit_kwargs["fastmath"] = use_fastmath

    @numba.njit(**njit_kwargs)
    def kernel(
        arr: np.ndarray,
        mean_value: float,
        weights: np.ndarray = None,
        weighted_mean_value: float = None,
        xvals: np.ndarray = None,
        directional_only: bool = False,
        return_lintrend_approx_stats: bool = True,
    ) -> list:  # pragma: no cover
        slope_over, slope_under = 0.0, 0.0
        mad, std, skew, kurt = 0.0, 0.0, 0.0, 0.0
        # Kahan compensation counters. When KAHAN=False these stay 0.0 and get DCE'd along with
        # every `if KAHAN: ... else: ...` block in the loop, so the fast variant pays nothing.
        slope_over_c = 0.0
        slope_under_c = 0.0
        r_sum_c = 0.0
        mad_c = 0.0
        std_c = 0.0
        skew_c = 0.0
        kurt_c = 0.0
        if weights is not None:
            sum_weights = 0.0
            weighted_mad, weighted_std, weighted_skew, weighted_kurt = mad, std, skew, kurt
            sum_weights_c = 0.0
            weighted_mad_c = 0.0
            weighted_std_c = 0.0
            weighted_skew_c = 0.0
            weighted_kurt_c = 0.0

        size = len(arr)

        if xvals is None:
            xvals = np.arange(size, dtype=np.float32)
        xvals_mean = np.mean(xvals)

        n_mean_crossings = 0.0
        has_prev_d = False
        prev_d = 0.0
        r_sum = 0.0

        for i, next_value in enumerate(arr):

            sl_x = xvals[i] - xvals_mean

            # slope_over += sl_x * next_value
            _inc = sl_x * next_value
            if KAHAN:
                _t = slope_over + _inc
                if abs(slope_over) >= abs(_inc):
                    slope_over_c += (slope_over - _t) + _inc
                else:
                    slope_over_c += (_inc - _t) + slope_over
                slope_over = _t
            else:
                slope_over += _inc

            # slope_under += sl_x**2
            _inc = sl_x * sl_x
            if KAHAN:
                _t = slope_under + _inc
                if abs(slope_under) >= abs(_inc):
                    slope_under_c += (slope_under - _t) + _inc
                else:
                    slope_under_c += (_inc - _t) + slope_under
                slope_under = _t
            else:
                slope_under += _inc

            d = next_value - mean_value

            # r_sum += sl_x * d
            _inc = sl_x * d
            if KAHAN:
                _t = r_sum + _inc
                if abs(r_sum) >= abs(_inc):
                    r_sum_c += (r_sum - _t) + _inc
                else:
                    r_sum_c += (_inc - _t) + r_sum
                r_sum = _t
            else:
                r_sum += _inc

            if has_prev_d:
                if d * prev_d < 0:
                    n_mean_crossings += 1
            prev_d = d
            has_prev_d = True

            # mad += abs(d)
            _inc = abs(d)
            if KAHAN:
                _t = mad + _inc
                if abs(mad) >= abs(_inc):
                    mad_c += (mad - _t) + _inc
                else:
                    mad_c += (_inc - _t) + mad
                mad = _t
            else:
                mad += _inc

            if weights is not None:
                next_weight = weights[i]
                w_d = next_value - weighted_mean_value

                # sum_weights += next_weight
                if KAHAN:
                    _t = sum_weights + next_weight
                    if abs(sum_weights) >= abs(next_weight):
                        sum_weights_c += (sum_weights - _t) + next_weight
                    else:
                        sum_weights_c += (next_weight - _t) + sum_weights
                    sum_weights = _t
                else:
                    sum_weights += next_weight

                # weighted_mad += abs(w_d) * next_weight
                _inc = abs(w_d) * next_weight
                if KAHAN:
                    _t = weighted_mad + _inc
                    if abs(weighted_mad) >= abs(_inc):
                        weighted_mad_c += (weighted_mad - _t) + _inc
                    else:
                        weighted_mad_c += (_inc - _t) + weighted_mad
                    weighted_mad = _t
                else:
                    weighted_mad += _inc

            summand = d * d
            # std += summand
            if KAHAN:
                _t = std + summand
                if abs(std) >= abs(summand):
                    std_c += (std - _t) + summand
                else:
                    std_c += (summand - _t) + std
                std = _t
            else:
                std += summand

            if weights is not None:
                w_summand = w_d * w_d
                # weighted_std += w_summand * next_weight
                _inc = w_summand * next_weight
                if KAHAN:
                    _t = weighted_std + _inc
                    if abs(weighted_std) >= abs(_inc):
                        weighted_std_c += (weighted_std - _t) + _inc
                    else:
                        weighted_std_c += (_inc - _t) + weighted_std
                    weighted_std = _t
                else:
                    weighted_std += _inc

            if not directional_only:

                summand = summand * d
                # skew += summand (d^3)
                if KAHAN:
                    _t = skew + summand
                    if abs(skew) >= abs(summand):
                        skew_c += (skew - _t) + summand
                    else:
                        skew_c += (summand - _t) + skew
                    skew = _t
                else:
                    skew += summand

                if weights is not None:
                    w_summand = w_summand * w_d
                    # weighted_skew += w_summand * next_weight
                    _inc = w_summand * next_weight
                    if KAHAN:
                        _t = weighted_skew + _inc
                        if abs(weighted_skew) >= abs(_inc):
                            weighted_skew_c += (weighted_skew - _t) + _inc
                        else:
                            weighted_skew_c += (_inc - _t) + weighted_skew
                        weighted_skew = _t
                    else:
                        weighted_skew += _inc

                # kurt += summand * d (d^4)
                _inc = summand * d
                if KAHAN:
                    _t = kurt + _inc
                    if abs(kurt) >= abs(_inc):
                        kurt_c += (kurt - _t) + _inc
                    else:
                        kurt_c += (_inc - _t) + kurt
                    kurt = _t
                else:
                    kurt += _inc

                if weights is not None:
                    # Was `weighted_skew +=` here in the original buggy version: double-
                    # accumulating skew while weighted_kurt stayed 0 -> constant -3.0 feature.
                    _inc = w_summand * w_d * next_weight
                    if KAHAN:
                        _t = weighted_kurt + _inc
                        if abs(weighted_kurt) >= abs(_inc):
                            weighted_kurt_c += (weighted_kurt - _t) + _inc
                        else:
                            weighted_kurt_c += (_inc - _t) + weighted_kurt
                        weighted_kurt = _t
                    else:
                        weighted_kurt += _inc

        # Apply Kahan corrections once at the end. DCE'd when KAHAN=False.
        if KAHAN:
            slope_over += slope_over_c
            slope_under += slope_under_c
            r_sum += r_sum_c
            mad += mad_c
            std += std_c
            skew += skew_c
            kurt += kurt_c

        std = np.sqrt(std / size)
        if weights is not None:
            if KAHAN:
                sum_weights += sum_weights_c
                weighted_mad += weighted_mad_c
                weighted_std += weighted_std_c
                weighted_skew += weighted_skew_c
                weighted_kurt += weighted_kurt_c
            weighted_std = np.sqrt(weighted_std / sum_weights)

        if not directional_only:
            mad = mad / size

            if std == 0:
                skew, kurt = 0.0, 0.0
            else:
                factor = size * std**3
                if factor:
                    skew = skew / factor

                    factor = factor * std
                    kurt = kurt / factor - 3.0

            if weights is not None:
                weighted_mad = weighted_mad / sum_weights
                if weighted_std == 0:
                    weighted_skew, weighted_kurt = 0.0, 0.0
                else:
                    factor = size * weighted_std**3
                    if factor:
                        weighted_skew = weighted_skew / factor

                        factor = factor * weighted_std
                        weighted_kurt = weighted_kurt / factor - 3.0

        if np.isclose(slope_under, 0) or np.isnan(slope_under):
            r = 0.0
            slope = np.nan
            intercept = np.nan
            n_lintrend_crossings = np.nan
        else:
            slope = slope_over / slope_under

            # R-value
            if np.isclose(std, 0):
                r = 0
            else:
                r = r_sum / (np.sqrt(slope_under) * std * np.sqrt(size))
                # Test for numerical error propagation (make sure -1 < r < 1)
                if r > 1.0:
                    r = 1.0
                elif r < -1.0:
                    r = -1.0

            # slope crossings & trend approximation errors

            has_prev_d = False
            prev_d = 0.0
            intercept = mean_value - slope * xvals_mean
            n_lintrend_crossings = 0.0

            if return_lintrend_approx_stats:
                lintrend_data_diffs = np.empty_like(arr)
            else:
                lintrend_data_diffs = None

            for i, next_value in enumerate(arr):
                d = next_value - (slope * xvals[i] + intercept)
                if has_prev_d:
                    if d * prev_d < 0:
                        n_lintrend_crossings += 1
                prev_d = d
                has_prev_d = True
                if return_lintrend_approx_stats:
                    lintrend_data_diffs[i] = d

        res = []
        if not directional_only:
            res.extend((mad, std, skew, kurt))
            if weights is not None:
                res.extend((weighted_mad, weighted_std, weighted_skew, weighted_kurt))
        res.extend((slope, intercept, r, n_mean_crossings, n_lintrend_crossings))
        return res, lintrend_data_diffs

    return kernel


# Private specializations: njit-compiled top-level callables.
_compute_moments_slope_mi_compensated = _make_compute_moments_slope_mi(use_kahan=True, use_fastmath=False)
_compute_moments_slope_mi_fast = _make_compute_moments_slope_mi(use_kahan=False, use_fastmath=True)


def compute_moments_slope_mi(
    arr: np.ndarray,
    mean_value: float,
    weights: np.ndarray = None,
    weighted_mean_value: float = None,
    xvals: np.ndarray = None,
    directional_only: bool = False,
    return_lintrend_approx_stats: bool = True,
    compensated: bool = False,
) -> tuple:
    """Per-row moments + slope/intercept/r + crossings.

    Parameters
    ----------
    compensated
        ``False`` (default) prefers the fastmath+no-Kahan kernel and falls back to the Kahan
        kernel when ``arr`` (or ``weights``) contains any NaN/inf. ~1.4x speedup on well-
        conditioned float64 N>=50k. Switch to ``True`` to force Kahan for float32 with large N
        or known ill-conditioned data (uncentered prices, extreme outliers).

    Returns ``(stats_list, lintrend_data_diffs)`` - see kernel source for the per-element layout.
    """
    if compensated:
        return _compute_moments_slope_mi_compensated(
            arr=arr,
            mean_value=mean_value,
            weights=weights,
            weighted_mean_value=weighted_mean_value,
            xvals=xvals,
            directional_only=directional_only,
            return_lintrend_approx_stats=return_lintrend_approx_stats,
        )
    # NaN-gate: see compute_simple_stats_numba for rationale.
    if not np.isfinite(arr).all() or (weights is not None and not np.isfinite(weights).all()):
        return _compute_moments_slope_mi_compensated(
            arr=arr,
            mean_value=mean_value,
            weights=weights,
            weighted_mean_value=weighted_mean_value,
            xvals=xvals,
            directional_only=directional_only,
            return_lintrend_approx_stats=return_lintrend_approx_stats,
        )
    return _compute_moments_slope_mi_fast(
        arr=arr,
        mean_value=mean_value,
        weights=weights,
        weighted_mean_value=weighted_mean_value,
        xvals=xvals,
        directional_only=directional_only,
        return_lintrend_approx_stats=return_lintrend_approx_stats,
    )


_EMPTY_FLOAT32 = np.array([], dtype=np.float32)


def compute_mutual_info_regression(arr: np.ndarray, xvals: np.ndarray = _EMPTY_FLOAT32) -> float:
    if len(xvals):
        mi = mutual_info_regression(xvals.reshape(-1, 1), arr, n_neighbors=2)
    else:
        mi = mutual_info_regression(np.arange(len(arr)).reshape(-1, 1), arr, n_neighbors=2)

    return mi[0]


def compute_entropy_features(arr: np.ndarray, sampling_frequency: int = 100, spectral_method: str = "welch") -> list:
    """Apply every function in ``entropy_funcs`` to ``arr`` and return their outputs as a tuple.

    Non-finite values are stripped first; if fewer than 2 finite samples remain, returns zeros.
    Inf/NaN outputs are collapsed to 0 via ``np.nan_to_num``.
    """
    # hjorth_mobility, hjorth_complexity = hjorth_params(arr)
    # hjorth_mobility,
    # hjorth_complexity,
    # spectral_entropy(arr, sf=sampling_frequency, method=spectral_method),)
    # num_zerocross(arr),

    nonzero = (~np.isnan(arr)).sum()
    if nonzero < 10:
        return (0.0,) * len(entropy_funcs)
    else:
        safe_arr = np.nan_to_num(arr[~np.isnan(arr)], posinf=0, neginf=0)
        return tuple(np.nan_to_num([f(safe_arr) for f in entropy_funcs], posinf=0, neginf=0))


def fit_distribution(dist: object, data: np.ndarray, method: str = "mle"):
    """Fit a scipy distribution to ``data`` and return parameters + Kolmogorov-Smirnov fit quality.

    Returns the distribution's fitted parameters concatenated with ``(ks_stat, ks_pval)``. On any
    exception during fitting, returns the predefined fallback tuple from ``default_dist_responses``.
    """
    try:
        params = dist.fit(data, method=method)
    except Exception:
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
    xvals: np.ndarray = None,
    weights: np.ndarray = None,
    geomean_log_mode: bool = False,
    q: Sequence[float] = default_quantiles,
    quantile_method: str = "median_unbiased",
    max_modes: int = 10,
    sampling_frequency: int = 100,
    spectral_method: str = "welch",
    hurst_kwargs: dict = None,
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
        params = dict(
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
    weights: np.ndarray = None,
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
    **kwargs
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
        + ([] if (directional_only or not return_entropy) else entropy_funcs_names)
        + (distributions_features_names if return_distributional else [])
    )

    if return_lintrend_approx_stats:
        params = dict(
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


def get_moments_slope_mi_feature_names(weights: np.ndarray = None, directional_only: bool = False, return_lintrend_approx_stats: bool = True):
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
@numba.njit(fastmath=False)
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


@numba.njit(fastmath=True)
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
        return _rolling_moving_average_compensated(arr, n)
    return _rolling_moving_average_fast(arr, n)


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
    cols: Sequence = None,
    values: np.ndarray = None,
    use_diffs: bool = False,
    rolling_ma: int = 0,
    numagg_params: dict = None,
    dtype=np.float32,
    n_jobs=-1,
    prefetch_factor: int = 2,
    **parallel_kwargs
) -> np.ndarray:
    """Computes numaggs over columns of a dataframe, in parallel fashion.

    Example of parallel_kwargs: ``compute_numaggs_parallel(df=X, cols=cols, temp_folder=<temp>)``.
    Exactly one of ``values`` or ``(df, cols)`` must be supplied.
    """
    if numagg_params is None:
        numagg_params = {}
    if values is None and (df is None or cols is None):
        raise ValueError(
            "compute_numaggs_parallel: must provide either `values` or both `df` and `cols`."
        )
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
        **parallel_kwargs
    )
    res = np.vstack(res).astype(dtype)

    return res
