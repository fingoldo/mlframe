"""Heavy numba-builder functions for numerical aggregates.

Wave 107 (2026-05-21): split out from ``feature_engineering/numerical.py``
to keep that file below the 1k-line monolith threshold. Behaviour preserved
bit-for-bit; every moved symbol is re-exported from ``numerical`` so
existing imports continue to work.

(Originally part of the "Numerical feature engineering for ML --
optimized & rich set of aggregates for 1d vectors" module.)
"""

from __future__ import annotations

__all__ = [
    "compute_numerical_aggregates_numba",
    "_make_compute_moments_slope_mi",
    "compute_moments_slope_mi",
]

# 2026-05-21: NUMBA_NJIT_PARAMS + the numeric constants used inside the
# njit kernels (LARGE_CONST, GEOMEAN_OVERFLOW_HI/LO, distributions) live
# in the parent ``numerical`` module. That module imports us from L272
# AFTER it has defined all of these at L83-128, so by the time Python
# resolves the names below the parent is partially loaded and the
# bindings are already in place -- single source of truth, no duplication
# drift. Numba reads these as module globals at @njit-decoration time
# (which happens BELOW this import block), so the bindings are visible
# in time for kernel compilation.
from .numerical import (  # noqa: E402
    NUMBA_NJIT_PARAMS,
    LARGE_CONST,
    GEOMEAN_OVERFLOW_HI,
    GEOMEAN_OVERFLOW_LO,
    distributions,
)

import logging
import warnings
from contextlib import contextmanager
from typing import Sequence, Tuple

import numba
import numpy as np
import pandas as pd
import psutil
from antropy import detrended_fluctuation, katz_fd, perm_entropy, petrosian_fd, sample_entropy, svd_entropy
from joblib import delayed

try:
    from astropy.stats import histogram as _astropy_histogram
except (ImportError, AttributeError):
    # astropy may be broken (e.g. numpy 2.x removed np.in1d while older astropy
    # still imports it at module level). Fall back to np.histogram вЂ” equivalent
    # for the bins="scott"/"auto" string-rule cases this module actually uses.
    _astropy_histogram = None



# cache=False overrides NUMBA_NJIT_PARAMS for this kernel only: numba's AOT
# cache for functions with many bool kwargs corrupts on Windows (Python 3.11
# + numba 0.59) -- a fresh process that calls this with all kwargs explicit
# loads a stale .nbc compilation and segfaults with an access violation.
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
    E Р¤СѓРЅРєС†РёРё РѕС€РёР±РѕРє РёРЅРѕРіРґР° Рё РєР»Р°СЃСЃРЅС‹Рµ РїСЂРёР·РЅР°РєРё...
    V What happens first: min or max? Add relative percentage of min/max indices
    V Add absolute values of min/max indices?
    V Р”РѕР±Р°РІРёС‚СЊ РєРѕР»РёС‡РµСЃС‚РІРѕ РїРµСЂРµСЃРµС‡РµРЅРёР№ СЃСЂРµРґРЅРёС… Рё РјРµРґРёР°РЅРЅРѕРіРѕ Р·РЅР°С‡РµРЅРёР№, Р»РёРЅРёРё slope? (trend reversions)
        РҐРѕС‚СЏ СЌС‚Рѕ РјРѕР¶РЅРѕ РІ С‚.С‡. РїРѕР»СѓС‡РёС‚СЊ, РІС‹Р·РІР°РІ СЃС‚Р°С‚Сѓ РЅР°Рґ РЅРѕСЂРјРёСЂРѕРІР°РЅРЅС‹Рј РёР»Рё РґРµС‚СЂРµРЅРґРёСЂРѕРІР°РЅРЅС‹Рј СЂСЏРґРѕРј (x-X_avg) РёР»Рё (x-(slope*x+x[0]))
    V СѓР±СЂР°С‚СЊ РіСЌРїС‹. СЌС‚Рѕ СЃС‚Р°С‚РёСЃС‚РёРєР° РІС‚РѕСЂРѕРіРѕ РїРѕСЂСЏРґРєР° Рё РґРѕР»Р¶РЅР° СЃС‡РёС‚Р°С‚СЊСЃСЏ РѕС‚РґРµР»СЊРЅРѕ. РїСЂРёС‡РµРј РјРѕР¶РЅРѕ СЃС‡РёС‚Р°С‚СЊ РѕС‚ СЂР°Р·РЅРѕСЃС‚РµР№ РёР»Рё РѕС‚ РѕС‚РЅРѕС€РµРЅРёР№.
    V РІР·РІРµС€РµРЅРЅС‹Рµ СЃС‚Р°С‚С‹ СЃС‡РёС‚Р°С‚СЊ РѕС‚РґРµР»СЊРЅС‹Рј РІС‹Р·РѕРІРѕРј ( Рё РЅРµ С‚РѕР»СЊРєРѕ СЃСЂРµРґРЅРµР°СЂРёС„РјРµС‚РёС‡РµСЃРєРёРµ, Р° Р’РЎР•).
    Р”РѕР±Р°РІРёС‚СЊ
        V СЃСЂРµРґРЅРµРµ РєСѓР±РёС‡РµСЃРєРѕРµ,
        V entropy
        V hurst
        V R2
        E? СЃСЂРµРґРЅРµРµ РІРёРЅР·РѕСЂРёР·РёСЂРѕРІР°РЅРЅРѕРµ (https://ru.wikipedia.org/wiki/%D0%92%D0%B8%D0%BD%D0%B7%D0%BE%D1%80%D0%B8%D0%B7%D0%BE%D0%B2%D0%B0%D0%BD%D0%BD%D0%BE%D0%B5_%D1%81%D1%80%D0%B5%D0%B4%D0%BD%D0%B5%D0%B5).
        E? СѓСЃРµС‡С‘РЅРЅРѕРµ,
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
    # numba Optional[NoneType|float64] segfaults under numba 0.62 / numpy 2.2 вЂ”
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
                # Was `next_value % 1` вЂ” robust for positive floats but fragile around negative
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
        # Wave 47 (2026-05-20): zero-sum weights vector (all-zero weight column or
        # entirely filtered-out fold) divides by 0 in the njit kernel and aborts.
        if sum_weights == 0.0:
            weighted_arithmetic_mean = np.nan
        else:
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
            # Wave 47 (2026-05-20): same sum_weights==0 guard as above.
            if sum_weights == 0.0:
                weighted_quadratic_mean = np.nan
            else:
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
            # Wave 47 (2026-05-20): sum_weights==0 (all-zero weight column)
            # used to crash the njit kernel here.
            if sum_weights == 0.0:
                weighted_std = np.nan
            else:
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
                # Wave 47 (2026-05-20): same sum_weights==0 guard.
                if sum_weights == 0.0:
                    weighted_mad = np.nan
                else:
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
