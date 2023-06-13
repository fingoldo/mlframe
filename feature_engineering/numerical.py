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

ensure_installed("numpy numba sklearn antropy entropy_estimators")  # npeet?

# ----------------------------------------------------------------------------------------------------------------------------
# Normal Imports
# ----------------------------------------------------------------------------------------------------------------------------

from typing import *

import numba
import numpy as np
from antropy import *
from entropy_estimators import continuous
from sklearn.feature_selection import mutual_info_regression
from mlframe.feature_engineering.hurst import compute_hurst_exponent

import warnings

warnings.filterwarnings("ignore", message="nperseg =")
warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=RuntimeWarning)

# ----------------------------------------------------------------------------------------------------------------------------
# Inits
# ----------------------------------------------------------------------------------------------------------------------------


def cont_entropy(arr: np.ndarray, bins: str = "auto") -> float:
    """Entropy of a continuous distribution"""
    try:
        hist, bin_edges = np.histogram(arr, bins=bins, density=True)
        ent = -(hist * np.log(hist + 1e-60)).sum()
    except Exception as e:
        return np.nan
    return ent


entropy_funcs = (cont_entropy, continuous.get_h, app_entropy, svd_entropy, sample_entropy, petrosian_fd, perm_entropy, katz_fd, detrended_fluctuation)
entropy_funcs_names = [f.__name__ for f in entropy_funcs]

default_quantiles: list = [0.1, 0.25, 0.5, 0.75, 0.9]  # list vs ndarray gives advantage 125 µs ± 2.79 µs per loop vs 140 µs ± 8.11 µs per loop


@numba.njit(fastmath=True)
def compute_numerical_aggregates_numba(
    arr: np.ndarray,
    geomean_log_mode: bool = False,
    directional_only: bool = False,
) -> list:
    """Compute statistical aggregates over 1d array of float32 values.
    E mid2(abs(x-mid1(X))) where mid1, mid2=averages of any kind
    E Функции ошибок иногда и классные признаки...
    V What happens first: min or max? Add relative percentage of min/max indices
    E Add absolute values of min/max indices?
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
        E fit variable to a number of known distributions!! their params become new features
        E drawdowns, negative drawdowns (for shorts), dd duration (%)
        E numpeaks
    """

    size = len(arr)

    first = arr[0]
    last = arr[-1]
    if first:
        ratio = last / first
    else:
        ratio = np.nan

    if directional_only:
        arithmetic_mean = np.mean(arr)
        return [arithmetic_mean, ratio]

    ninteger, npositive, cnt_nonzero = 0, 0, 0

    if not geomean_log_mode:
        geometric_mean = 1.0
    else:
        geometric_mean = 0.0

    arithmetic_mean, quadratic_mean, qubic_mean, harmonic_mean = 0.0, 0.0, 0.0, 0.0

    maximum, minimum = first, first
    max_index, min_index = 0, 0
    max_pos_dd, max_pos_dd_duration = 0.0, 0
    max_neg_dd, max_neg_dd_duration = 0.0, 0

    pos_dd_start_idx, neg_dd_start_idx = 0, 0

    for i, next_value in enumerate(arr):
        arithmetic_mean += next_value

        temp_value = next_value * next_value
        quadratic_mean += temp_value

        temp_value = temp_value * next_value
        qubic_mean += temp_value

        if next_value > maximum:
            maximum = next_value
            max_index = i
        elif next_value < minimum:
            minimum = next_value
            min_index = i

        # Drawdowns

        # pos
        dd = maximum - next_value
        if dd > 0.0:
            if dd > max_pos_dd:
                max_pos_dd = dd
            if not pos_dd_start_idx:
                pos_dd_start_idx = i
        else:
            if pos_dd_start_idx:
                dd_dur = i - pos_dd_start_idx
                pos_dd_start_idx = 0
                if dd_dur > max_pos_dd_duration:
                    max_pos_dd_duration = dd_dur
        # neg
        dd = next_value - minimum
        if dd > 0.0:
            if dd > max_neg_dd:
                max_neg_dd = dd
            if not neg_dd_start_idx:
                neg_dd_start_idx = i
        else:
            if neg_dd_start_idx:
                dd_dur = i - neg_dd_start_idx
                neg_dd_start_idx = 0
                if dd_dur > max_neg_dd_duration:
                    max_neg_dd_duration = dd_dur

        if next_value:
            cnt_nonzero = cnt_nonzero + 1
            harmonic_mean += 1 / next_value

            frac = next_value % 1
            if not frac:
                ninteger = ninteger + 1

            if next_value > 0:
                npositive = npositive + 1
                if not geomean_log_mode:
                    geometric_mean *= next_value
                    if geometric_mean > 1e100 or geometric_mean < 1e-100:
                        # convert to log mode
                        geomean_log_mode = True
                        geometric_mean = np.log(float(geometric_mean))
                else:
                    geometric_mean += np.log(next_value)

    if pos_dd_start_idx:
        dd_dur = i - pos_dd_start_idx
        if dd_dur > max_pos_dd_duration:
            max_pos_dd_duration = dd_dur
    if neg_dd_start_idx:
        dd_dur = i - neg_dd_start_idx
        if dd_dur > max_neg_dd_duration:
            max_neg_dd_duration = dd_dur

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

    arithmetic_mean = arithmetic_mean / size
    quadratic_mean = np.sqrt(quadratic_mean / size)
    qubic_mean = (qubic_mean / size) ** (1 / 3)

    return [
        arithmetic_mean,
        quadratic_mean,
        qubic_mean,
        geometric_mean,
        harmonic_mean,
        cnt_nonzero,
        ratio,
        npositive,
        ninteger,
        minimum,
        maximum,
        (min_index + 1) / size,
        (max_index + 1) / size,
        max_pos_dd,
        max_neg_dd,
        max_pos_dd_duration / size,
        max_neg_dd_duration / size,
    ]


def compute_nunique_modes_quantiles_numpy(arr: np.ndarray, q: list = default_quantiles, quantile_method: str = "median_unbiased", max_modes: int = 10) -> list:
    """For a 1d array, computes aggregates:
    nunique
    modes:min,max,mean
    list of quantiles (0 and 1 included by default, therefore, min/max)
    Can NOT be numba jitted.
    """
    size = len(arr)
    vals, counts = np.unique(arr, return_counts=True)

    max_modes = min(max_modes, len(counts))

    modes_indices = np.argpartition(counts, -max_modes)[-max_modes:]
    modes_indices = modes_indices[np.argsort(counts[modes_indices])][::-1]

    first_mode_count = counts[modes_indices[0]]

    if first_mode_count == 1:
        modes_min, modes_max, modes_mean, modes_qty = np.nan, np.nan, np.nan, np.nan  # for higher stability. cnt=1 is not really a mode, rather a random pick.
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

    return [nuniques, modes_min, modes_max, modes_mean, modes_qty] + np.quantile(arr, q, method=quantile_method).tolist()


@numba.njit(fastmath=True)
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


# @numba.njit(fastmath=True)
def compute_moments_slope_mi(
    arr: np.ndarray,
    mean_value: float,
    xvals: np.ndarray = np.array([], dtype=np.float32),
    directional_only: bool = False,
) -> list:
    """Добавить:
    V RANSAC-регрессию или что-то подобное, устойчивое к выбросам?
    V Количество пересечений не просто среднего, а ещё и квантилей. Для финансовых приложений это мб полезно, тк есть гипотеза,
        что если цена тестирует уровень много раз в течение дня, она его в итоге пробъёт (https://youtu.be/2DrBc35VLvE?t=129).
    V Можно для начала добавить отношение крайних квантилей к макс/мин.
    """
    slope_over, slope_under = 0.0, 0.0
    mad, std, skew, kurt = 0.0, 0.0, 0.0, 0.0

    size = len(arr)

    if not len(xvals):
        xvals = np.array(np.arange(size))
        # xvals_mean = 1 / 2 * (2 * 0 + (size - 1) * 1)
    xvals_mean = np.mean(xvals)

    n_mean_crossings = 0
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

        mad = mad + abs(d)

        summand = d * d
        std = std + summand

        if not directional_only:

            summand = summand * d
            skew = skew + summand

            kurt = kurt + summand * d

    # mi = mutual_info_regression(xvals.reshape(-1, 1), arr, n_neighbors=2)  # n_neighbors=2 is strictly needed for short sequences

    std = np.sqrt(std / size)

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

    if np.isclose(slope_under, 0) or np.isnan(slope_under):
        r = 0.0
        slope = np.nan
        n_slope_crossings = np.nan
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

        # slope crossings

        prev_d = None
        intercept = mean_value - slope * xvals_mean
        n_slope_crossings = 0

        for i, next_value in enumerate(arr):
            d = next_value - (slope * xvals[i] + intercept)
            if prev_d is not None:
                if d * prev_d < 0:
                    n_slope_crossings += 1
            prev_d = d
    if not directional_only:
        return [mad, std, skew, kurt, slope, r, n_mean_crossings, n_slope_crossings]  # , mi
    else:
        return [slope, r, n_mean_crossings, n_slope_crossings]


def compute_mutual_info_regression(arr: np.ndarray, xvals: np.ndarray = np.array([], dtype=np.float32)) -> float:
    if len(xvals):
        mi = mutual_info_regression(xvals.reshape(-1, 1), arr, n_neighbors=2)
    else:
        mi = mutual_info_regression(np.arange(len(arr)).reshape(-1, 1), arr, n_neighbors=2)

    return mi[0]


def compute_entropy_fetures(arr: np.ndarray, nonzero: int, sampling_frequency: int = 100, spectral_method: str = "welch") -> list:
    # hjorth_mobility, hjorth_complexity = hjorth_params(arr)
    # hjorth_mobility,
    # hjorth_complexity,
    # spectral_entropy(arr, sf=sampling_frequency, method=spectral_method),)
    # num_zerocross(arr),
    if nonzero < 10:
        return [np.nan] * len(entropy_funcs)
    else:
        return [f(arr) for f in entropy_funcs]


def compute_numaggs(
    arr: np.ndarray,
    xvals: np.ndarray = np.array([], dtype=np.float32),
    geomean_log_mode: bool = False,
    q: list = default_quantiles,
    quantile_method: str = "median_unbiased",
    max_modes: int = 10,
    sampling_frequency: int = 100,
    spectral_method: str = "welch",
    hurst_kwargs: dict = dict(min_window=10, max_window=None, windows_log_step=0.25, take_diffs=False),
    directional_only: bool = False,
):
    """Compute a plethora of numerical aggregates for all values in an array.
    Converts an arbitrarily length array into fixed number of aggregates.
    """
    if len(arr) == 0:
        return [np.nan] * len(get_numaggs_names(q=q, directional_only=directional_only))
    res = compute_numerical_aggregates_numba(arr, geomean_log_mode=geomean_log_mode, directional_only=directional_only)
    arithmetic_mean = res[0]
    if directional_only:
        nonzero = 0
    else:
        nonzero = res[5]
    return (
        res
        + ([] if directional_only else compute_nunique_modes_quantiles_numpy(arr=arr, q=q, quantile_method=quantile_method, max_modes=max_modes))
        + compute_moments_slope_mi(arr=arr, mean_value=arithmetic_mean, xvals=xvals, directional_only=directional_only)
        # + [compute_mutual_info_regression(arr=arr, xvals=xvals)]
        + [*compute_hurst_exponent(arr=arr, **hurst_kwargs)]
        + (
            []
            if directional_only
            else compute_entropy_fetures(arr=arr, sampling_frequency=sampling_frequency, spectral_method=spectral_method, nonzero=nonzero)
        )
    )


def get_numaggs_names(q: list = default_quantiles, directional_only: bool = False, **kwargs) -> tuple:
    return tuple(
        (
            ["arimean", "ratio"]
            if directional_only
            else "arimean,quadmean,qubmean,geomean,harmmean,nonzero,ratio,npos,nint,min,max,minr,maxr,max_pos_dd,max_neg_dd,max_pos_dd_durationr,max_neg_dd_durationr".split(
                ","
            )
        )
        + ([] if directional_only else "nuniques,modmin,modmax,modmean,modqty".split(","))
        + ([] if directional_only else ["q" + str(q) for q in q])
        + ("slope,r,meancross,slopecross".split(",") if directional_only else "mad,std,skew,kurt,slope,r,meancross,slopecross".split(","))  # ,mi
        # + ["mutual_info_regression",]
        + ["hursth", "hurstc"]
        + ([] if directional_only else entropy_funcs_names)
    )
