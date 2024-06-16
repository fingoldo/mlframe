"""Numerical feature engineering for ML. Optimized & rich set of aggregates for 1d vectors."""

# pylint: disable=wrong-import-order,wrong-import-position,unidiomatic-typecheck,pointless-string-statement

# ----------------------------------------------------------------------------------------------------------------------------
# LOGGING
# ----------------------------------------------------------------------------------------------------------------------------

import logging

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------------------------------------------------------------
# Packages
# ----------------------------------------------------------------------------------------------------------------------------

from pyutilz.pythonlib import (
    ensure_installed,
)  # lint: disable=ungrouped-imports,disable=wrong-import-order

ensure_installed("numpy numba sklearn antropy astropy entropy_estimators")  # npeet?

# ----------------------------------------------------------------------------------------------------------------------------
# Normal Imports
# ----------------------------------------------------------------------------------------------------------------------------

from typing import *

import numba
import numpy as np, pandas as pd
from antropy import *
from astropy.stats import histogram
from entropy_estimators import continuous
from sklearn.feature_selection import mutual_info_regression
from mlframe.feature_engineering.hurst import compute_hurst_exponent

import psutil
from joblib import delayed
from pyutilz.parallel import parallel_run

from scipy.stats import entropy, kstest
from scipy import stats

import warnings

warnings.filterwarnings("ignore", message="nperseg =")
warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=RuntimeWarning)

# ----------------------------------------------------------------------------------------------------------------------------
# Inits
# ----------------------------------------------------------------------------------------------------------------------------

fastmath = False
empty_float32_array = np.array([], dtype=np.float32)


def cont_entropy(arr: np.ndarray, bins: str = "scott") -> float:
    """Entropy of a continuous distribution"""
    try:
        hist, bin_edges = histogram(arr, bins=bins)  # np.histogram(arr, bins=bins, density=True)
        ent = -(hist * np.log(hist + 1e-60)).sum()
    except Exception as e:
        return np.nan
    return ent


entropy_funcs = (cont_entropy, continuous.get_h, app_entropy, svd_entropy, sample_entropy, petrosian_fd, perm_entropy, katz_fd, detrended_fluctuation)  #
entropy_funcs_names = [f.__name__ for f in entropy_funcs]

distributions = (stats.levy_l,)  # stats.logistic, stats.pareto
default_dist_responses = dict(levy_l=(np.nan, np.nan), logistic=(np.nan, np.nan), pareto=(np.nan, np.nan, np.nan))

LARGE_CONST = 1e3


def get_distributions_features_names() -> list:
    distributions_features_names = []
    for dist in distributions:
        for i in range(len(default_dist_responses[dist.name])):
            distributions_features_names.append(dist.name + str(i + 1))
        distributions_features_names.append(dist.name + "_kss")
        distributions_features_names.append(dist.name + "_kspval")
    return distributions_features_names


distributions_features_names = get_distributions_features_names()

default_quantiles: list = [0.1, 0.25, 0.5, 0.75, 0.9]  # list vs ndarray gives advantage 125 µs ± 2.79 µs per loop vs 140 µs ± 8.11 µs per loop


@numba.njit(fastmath=fastmath)
def compute_simple_stats_numba(arr: np.ndarray) -> tuple:
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
            if next_value < minval:
                minval = next_value
                argmin = i
            elif next_value > maxval:
                maxval = next_value
                argmax = i

    if size == 0:
        size = len(arr)

    if size:
        mean_value = total / size
    else:
        mean_value = 0.0
        minval, maxval = 0.0, 0.0

    for i, next_value in enumerate(arr):
        if np.isfinite(next_value):
            d = next_value - mean_value
            summand = d * d
            std_val = std_val + summand
    if size:
        std_val = np.sqrt(std_val / size)
    else:
        std_val = 0.0
    return minval, maxval, argmin, argmax, mean_value, std_val


def get_simple_stats_names() -> list:
    return "min,max,argmin,argmax,mean,std".split(",")


@numba.njit(fastmath=fastmath)
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
) -> list:
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

    first = arr[0]
    last = arr[-1]
    if first:
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
    prev_d = None

    for i, next_value in enumerate(arr):
        if weights is not None:
            next_weight = weights[i]
            sum_weights += weights[i]
            if weights is not None:
                weighted_arithmetic_mean += next_value * next_weight

        if return_unsorted_stats:
            d = next_value - last
            if prev_d is not None:
                mul = d * prev_d
                if mul < 0:
                    n_last_crossings += 1
                elif mul == 0.0:
                    n_last_touches += 1
            else:
                if next_value == last:
                    n_last_touches += 1
            prev_d = d

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

        if next_value > maximum:
            maximum = next_value
            max_index = i
            nmaxupdates += 1
        elif next_value < minimum:
            minimum = next_value
            min_index = i
            nminupdates += 1

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
                frac = next_value % 1
                if not frac:
                    ninteger = ninteger + 1

            if next_value > 0:
                npositive = npositive + 1
                sum_positive += next_value
                if return_exotic_means:
                    if not geomean_log_mode:
                        geometric_mean *= next_value
                        if weights is not None:
                            weighted_geometric_mean *= next_value**next_weight
                        if geometric_mean > 1e100 or geometric_mean < 1e-100:
                            # convert to log mode
                            geomean_log_mode = True
                            geometric_mean = np.log(float(geometric_mean))
                            if weights is not None:
                                weighted_geometric_mean = np.log(float(weighted_geometric_mean))
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
    arr: np.ndarray, q: list = default_quantiles, quantile_method: str = "median_unbiased", max_modes: int = 10, return_unsorted_stats: bool = True
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


def compute_ncrossings(arr: np.ndarray, marks: np.ndarray, dtype=np.int32) -> np.ndarray:
    n_crossings = np.zeros(len(marks), dtype=dtype)
    prev_ds = np.full(len(marks), dtype=np.float32, fill_value=np.nan)

    for next_value in arr:
        for i, mark in enumerate(marks):
            d = next_value - mark
            if prev_ds[i] is not None:
                if d * prev_ds[i] < 0:
                    n_crossings[i] += 1
            prev_ds[i] = d

    return n_crossings


@numba.njit(fastmath=fastmath)
def compute_nunique_mode_quantiles_numba(arr: np.ndarray, q: list = default_quantiles) -> tuple:
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
    for q in q:
        quantiles.append(xsorted[int(np.ceil(q * factor)) - 1])

    # if size % 2 == 0:
    #    sent = int(size / 2)
    #    median = xsorted[sent - 1] + xsorted[sent]
    # else:
    #    median = xsorted[int(size // 2)]

    if times_occured == 1:
        mode = np.nan
    return numUnique, mode, quantiles


@numba.njit(fastmath=fastmath)
def compute_moments_slope_mi(
    arr: np.ndarray,
    mean_value: float,
    weights: np.ndarray = None,
    weighted_mean_value: float = None,
    xvals: np.ndarray = None,  # empty_float32_array,
    directional_only: bool = False,
    return_lintrend_approx_stats: bool = True,
) -> list:
    """Добавить:
    ? RANSAC-регрессию или что-то подобное, устойчивое к выбросам?
    V Количество пересечений не просто среднего, а ещё и квантилей. Для финансовых приложений это мб полезно, тк есть гипотеза,
        что если цена тестирует уровень много раз в течение дня, она его в итоге пробъёт (https://youtu.be/2DrBc35VLvE?t=129).
    ? Можно для начала добавить отношение крайних квантилей к макс/мин.
    """
    slope_over, slope_under = 0.0, 0.0
    mad, std, skew, kurt = 0.0, 0.0, 0.0, 0.0
    if weights is not None:
        sum_weights = 0.0
        weighted_mad, weighted_std, weighted_skew, weighted_kurt = mad, std, skew, kurt

    size = len(arr)

    if xvals is None:  # len(xvals)==0:
        xvals = np.arange(size, dtype=np.float32)
        # xvals_mean = 1 / 2 * (2 * 0 + (size - 1) * 1)
    xvals_mean = np.mean(xvals)

    n_mean_crossings = 0.0
    prev_d = None

    r_sum = 0.0

    for i, next_value in enumerate(arr):

        sl_x = xvals[i] - xvals_mean

        slope_over += sl_x * next_value
        slope_under += sl_x**2

        d = next_value - mean_value
        r_sum += sl_x * d

        if prev_d is not None:
            if d * prev_d < 0:
                n_mean_crossings += 1
        prev_d = d

        mad += abs(d)
        if weights is not None:
            next_weight = weights[i]
            w_d = next_value - weighted_mean_value
            sum_weights += next_weight
            weighted_mad += abs(w_d) * next_weight

        summand = d * d
        std += summand
        if weights is not None:
            w_summand = w_d * w_d
            weighted_std += w_summand * next_weight

        if not directional_only:

            summand = summand * d
            skew += summand
            if weights is not None:
                w_summand = w_summand * w_d
                weighted_skew += w_summand * next_weight

            kurt += summand * d
            if weights is not None:
                weighted_skew += w_summand * w_d * next_weight

    # mi = mutual_info_regression(xvals.reshape(-1, 1), arr, n_neighbors=2)  # n_neighbors=2 is strictly needed for short sequences

    std = np.sqrt(std / size)
    if weights is not None:
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

        prev_d = None
        intercept = mean_value - slope * xvals_mean
        n_lintrend_crossings = 0.0

        if return_lintrend_approx_stats:
            lintrend_data_diffs = np.empty_like(arr)
        else:
            lintrend_data_diffs = None

        for i, next_value in enumerate(arr):
            d = next_value - (slope * xvals[i] + intercept)
            if prev_d is not None:
                if d * prev_d < 0:
                    n_lintrend_crossings += 1
            prev_d = d
            if return_lintrend_approx_stats:
                lintrend_data_diffs[i] = d

    res = []
    if not directional_only:
        res.extend((mad, std, skew, kurt))
        if weights is not None:
            res.extend((weighted_mad, weighted_std, weighted_skew, weighted_kurt))
    res.extend((slope, intercept, r, n_mean_crossings, n_lintrend_crossings))
    return res, lintrend_data_diffs


def compute_mutual_info_regression(arr: np.ndarray, xvals: np.ndarray = np.array([], dtype=np.float32)) -> float:
    if len(xvals):
        mi = mutual_info_regression(xvals.reshape(-1, 1), arr, n_neighbors=2)
    else:
        mi = mutual_info_regression(np.arange(len(arr)).reshape(-1, 1), arr, n_neighbors=2)

    return mi[0]


def compute_entropy_features(arr: np.ndarray, sampling_frequency: int = 100, spectral_method: str = "welch") -> list:
    # hjorth_mobility, hjorth_complexity = hjorth_params(arr)
    # hjorth_mobility,
    # hjorth_complexity,
    # spectral_entropy(arr, sf=sampling_frequency, method=spectral_method),)
    # num_zerocross(arr),

    nonzero = (~np.isnan(arr)).sum()
    if nonzero < 10:
        return (0.0,) * len(entropy_funcs)
    else:
        safe_arr = arr[~np.isnan(arr)]
        return tuple(np.nan_to_num([f(safe_arr) for f in entropy_funcs], posinf=0, neginf=0))


def fit_distribution(dist: object, data: np.ndarray, method: str = "mle"):
    try:
        params = dist.fit(data, method=method)
    except Exception as e:
        return default_dist_responses[dist.name] + (np.nan, np.nan)
    else:
        dist_fitted = dist(*params)
        ks_stat, ks_pval = kstest(data, dist_fitted.cdf)

        return tuple(np.nan_to_num([*params, ks_stat, ks_pval], posinf=0, neginf=0))


def compute_distributional_features(arr: np.ndarray) -> tuple:
    res = tuple()
    for dist in distributions:
        res = res + fit_distribution(dist=dist, data=arr)
    return res


def compute_numaggs(
    arr: np.ndarray,
    xvals: np.ndarray = None,
    weights: np.ndarray = None,
    geomean_log_mode: bool = False,
    q: list = default_quantiles,
    quantile_method: str = "median_unbiased",
    max_modes: int = 10,
    sampling_frequency: int = 100,
    spectral_method: str = "welch",
    hurst_kwargs: dict = dict(min_window=10, max_window=None, windows_log_step=0.25, take_diffs=False),
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
):
    """Compute a plethora of numerical aggregates for all values in an array.
    Converts an arbitrarily length array into fixed number of aggregates.
    """
    if len(arr) == 0:
        return (np.nan,) * len(
            get_numaggs_names(
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
            return_float32=False,
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
    q: list = default_quantiles,
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
        + ([] if not return_unsorted_stats else ["ncrs" + str(q) for q in q])
        + get_moments_slope_mi_feature_names(weights=weights, directional_only=directional_only, return_lintrend_approx_stats=return_lintrend_approx_stats)
        # + ["mutual_info_regression",]
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
            return_float32=False,
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
    res = []
    if not directional_only:
        res.extend("mad,std,skew,kurt".split(","))
        if weights is not None:
            res.extend("wmad,wstd,wskew,wkurt".split(","))
    res.extend("slope,intercept,r,meancross,trendcross".split(","))
    return res


@numba.njit(fastmath=True)
def rolling_moving_average(arr: np.ndarray, n: int = 2):
    if n <= 0:
        raise ValueError("n must be greater than 0")
    if n > len(arr):
        raise ValueError("n must be less than or equal to the length of the array")

    result = np.empty(len(arr) - n + 1, dtype=arr.dtype)
    sum_window = np.sum(arr[:n])

    mult = 1 / n

    result[0] = sum_window * mult

    for i in range(1, len(arr) - n + 1):
        sum_window += arr[i + n - 1] - arr[i - 1]
        result[i] = sum_window * mult

    return result


def numaggs_over_matrix_rows(vals: np.ndarray, numagg_params: dict, rolling_ma: int = 0, use_diffs: bool = False, dtype=np.float32) -> np.ndarray:
    """Computes numaggs on a 2d matrix"""

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
    numagg_params: dict = {},
    dtype=np.float32,
    n_jobs=-1,
    prefetch_factor: int = 2,
    **parallel_kwargs
) -> np.ndarray:
    """Computes numaggs over columns of a dataframe, in parallel fashion.
    Example of parallel_kwargs: numaggs_over_df_columns_parallel(df=X, cols=cols, temp_folder=r'R:\Temp')
    """
    if n_jobs <= 0:
        n_jobs = psutil.cpu_count(logical=False)

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
    res = np.vstack(res)

    return res
