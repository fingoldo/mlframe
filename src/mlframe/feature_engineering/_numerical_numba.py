"""Heavy numba-builder functions for numerical aggregates.

Split out from ``feature_engineering/numerical.py`` to keep that file below the
1k-line monolith threshold. Behaviour preserved bit-for-bit; every moved symbol
is re-exported from ``numerical`` so existing imports continue to work.

(Originally part of the "Numerical feature engineering for ML --
optimized & rich set of aggregates for 1d vectors" module.)
"""

from __future__ import annotations

from typing import Optional, cast

__all__ = [
    "compute_numerical_aggregates_numba",
    "_make_compute_moments_slope_mi",
    "compute_moments_slope_mi",
]

# NUMBA_NJIT_PARAMS + the numeric constants used inside the
# njit kernels (LARGE_CONST, GEOMEAN_OVERFLOW_HI/LO, distributions) live
# in the parent ``numerical`` module. That module imports us from L272
# AFTER it has defined all of these at L83-128, so by the time Python
# resolves the names below the parent is partially loaded and the
# bindings are already in place -- single source of truth, no duplication
# drift. Numba reads these as module globals at @njit-decoration time
# (which happens BELOW this import block), so the bindings are visible
# in time for kernel compilation.
from ._numerical_constants import (
    NUMBA_NJIT_PARAMS,
    LARGE_CONST,
)


import numba
import numpy as np

from ._numerical_numba_steps import (
    _finish_power_means,
    _finish_weighted_power_means,
    _geometric_mean_step,
    _finish_geometric_mean,
    _crossing_step,
    _drawdown_step,
    _whiten,
    _nonzero_step,
    _finish_harmonic_mean,
    _finish_weighted_geometric_mean,
    _assemble_aggregates,
    _extrema_step,
    _power_sums_step,
    _finish_moments,
    _kahan_add,
    _linear_trend_stats,
)


@numba.njit(**{**NUMBA_NJIT_PARAMS, "cache": False})
def _aggregate_scan(  # pragma: no cover
    arr, weights, first, last, geomean_log_mode, return_drawdown_stats, return_unsorted_stats, return_exotic_means, return_n_zer_pos_int
):
    """One pass over ``arr`` accumulating every running sum / count / extremum that ``compute_numerical_aggregates_numba`` reports, plus the drawdown arrays.

    ``cache=False`` for the same reason as the public kernel (numba's on-disk cache corrupts for functions with many bool arguments on Windows).
    """
    size = len(arr)
    ninteger, npositive, cnt_nonzero = 0, 0, 0
    sum_positive, sum_negative = 0.0, 0.0

    if not geomean_log_mode:
        geometric_mean = 1.0
    else:
        geometric_mean = 0.0

    arithmetic_mean, quadratic_mean, qubic_mean, harmonic_mean = 0.0, 0.0, 0.0, 0.0
    sum_weights = 0.0
    weighted_arithmetic_mean = 0.0
    weighted_geometric_mean = weighted_quadratic_mean = weighted_qubic_mean = weighted_harmonic_mean = 0.0
    if weights is not None:
        weighted_geometric_mean, weighted_arithmetic_mean, weighted_quadratic_mean, weighted_qubic_mean, weighted_harmonic_mean = (
            geometric_mean,
            arithmetic_mean,
            quadratic_mean,
            qubic_mean,
            harmonic_mean,
        )

    maximum, minimum = first, first
    max_index, min_index = 0, 0

    if return_drawdown_stats:

        pos_dd_start_idx, neg_dd_start_idx = 0, 0

        pos_dds = np.empty(shape=size, dtype=np.float32)
        pos_dd_durs = np.empty(shape=size, dtype=np.float32)

        neg_dds = np.empty(shape=size, dtype=np.float32)
        neg_dd_durs = np.empty(shape=size, dtype=np.float32)

    else:
        pos_dd_start_idx, neg_dd_start_idx = 0, 0
        pos_dds = np.empty(shape=0, dtype=np.float32)
        pos_dd_durs = np.empty(shape=0, dtype=np.float32)
        neg_dds = np.empty(shape=0, dtype=np.float32)
        neg_dd_durs = np.empty(shape=0, dtype=np.float32)

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
            weighted_arithmetic_mean += next_value * next_weight

        if return_unsorted_stats:
            n_last_crossings, n_last_touches = _crossing_step(next_value, last, has_prev_d, prev_d, n_last_crossings, n_last_touches)
            prev_d = next_value - last
            has_prev_d = True

        arithmetic_mean += next_value

        if return_exotic_means:
            if weights is not None:
                quadratic_mean, qubic_mean, weighted_quadratic_mean, weighted_qubic_mean = _power_sums_step(
                    next_value, next_weight, True, quadratic_mean, qubic_mean, weighted_quadratic_mean, weighted_qubic_mean
                )
            else:
                quadratic_mean, qubic_mean, _unused_wq, _unused_wc = _power_sums_step(next_value, 0.0, False, quadratic_mean, qubic_mean, 0.0, 0.0)

        # Independent checks (not if/elif): elif would mean a sample equal to minimum can never
        # update maximum, producing inconsistent min_index/max_index on degenerate inputs.
        minimum, min_index, nminupdates, maximum, max_index, nmaxupdates = _extrema_step(
            i, next_value, minimum, min_index, nminupdates, maximum, max_index, nmaxupdates
        )

        # ----------------------------------------------------------------------------------------------------------------------------
        # Drawdowns
        # ----------------------------------------------------------------------------------------------------------------------------

        if return_drawdown_stats:
            pos_dd_start_idx = _drawdown_step(i, maximum - next_value, pos_dd_start_idx, pos_dds, pos_dd_durs)
            neg_dd_start_idx = _drawdown_step(i, next_value - minimum, neg_dd_start_idx, neg_dds, neg_dd_durs)

        if next_value:
            cnt_nonzero = cnt_nonzero + 1
            if weights is not None:
                harmonic_mean, weighted_harmonic_mean, ninteger = _nonzero_step(
                    next_value, next_weight, True, return_exotic_means, return_n_zer_pos_int, harmonic_mean, weighted_harmonic_mean, ninteger
                )
            else:
                harmonic_mean, _unused_whm, ninteger = _nonzero_step(
                    next_value, 0.0, False, return_exotic_means, return_n_zer_pos_int, harmonic_mean, 0.0, ninteger
                )

            if next_value > 0:
                npositive = npositive + 1
                sum_positive += next_value
                if return_exotic_means:
                    if weights is not None:
                        geometric_mean, weighted_geometric_mean, geomean_log_mode = _geometric_mean_step(
                            geometric_mean, weighted_geometric_mean, next_value, next_weight, True, geomean_log_mode
                        )
                    else:
                        geometric_mean, _unused_wgm, geomean_log_mode = _geometric_mean_step(geometric_mean, 0.0, next_value, 0.0, False, geomean_log_mode)
            else:
                sum_negative += next_value

    return (
        arithmetic_mean, weighted_arithmetic_mean, quadratic_mean, qubic_mean, geometric_mean, harmonic_mean,
        weighted_quadratic_mean, weighted_qubic_mean, weighted_geometric_mean, weighted_harmonic_mean, sum_weights,
        minimum, maximum, min_index, max_index, nmaxupdates, nminupdates, n_last_crossings, n_last_touches,
        cnt_nonzero, npositive, ninteger, sum_positive, sum_negative, geomean_log_mode, pos_dds, pos_dd_durs, neg_dds, neg_dd_durs,
    )


@numba.njit(**NUMBA_NJIT_PARAMS)
def _finish_aggregate_means(  # pragma: no cover
    arr, weights, size, whiten_means, return_exotic_means, geomean_log_mode, arithmetic_mean, weighted_arithmetic_mean, sum_weights, quadratic_mean,
    qubic_mean, geometric_mean, harmonic_mean, weighted_quadratic_mean, weighted_qubic_mean, weighted_geometric_mean, weighted_harmonic_mean,
    minimum, maximum, npositive,
):
    """Turn the accumulated sums into the (weighted) arithmetic and exotic means; returns the ten means in the order the caller unpacks them."""
    arithmetic_mean = arithmetic_mean / size
    if weights is not None:
        # Zero-sum weights vector (all-zero weight column or
        # entirely filtered-out fold) divides by 0 in the njit kernel and aborts.
        if sum_weights == 0.0:
            weighted_arithmetic_mean = np.nan
        else:
            weighted_arithmetic_mean = weighted_arithmetic_mean / sum_weights

    if return_exotic_means:
        # qubic_mean is the mean of cubes and is negative for net-negative columns (common for returns/residuals), hence the signed cube root.
        quadratic_mean, qubic_mean = _finish_power_means(arr, quadratic_mean, qubic_mean, minimum, maximum, size)
        geometric_mean = _finish_geometric_mean(geometric_mean, geomean_log_mode, size) if npositive else np.nan
        harmonic_mean = _finish_harmonic_mean(harmonic_mean, size)

        if weights is not None:
            weighted_quadratic_mean, weighted_qubic_mean = _finish_weighted_power_means(
                arr, weights, weighted_quadratic_mean, weighted_qubic_mean, minimum, maximum, sum_weights, size
            )
            weighted_geometric_mean = _finish_weighted_geometric_mean(weighted_geometric_mean, geomean_log_mode, sum_weights, npositive)
            weighted_harmonic_mean = _finish_harmonic_mean(weighted_harmonic_mean, sum_weights)

        if whiten_means:
            quadratic_mean, qubic_mean, geometric_mean, harmonic_mean = _whiten(quadratic_mean, qubic_mean, geometric_mean, harmonic_mean, arithmetic_mean)
            if weights is not None:
                weighted_quadratic_mean, weighted_qubic_mean, weighted_geometric_mean, weighted_harmonic_mean = _whiten(
                    weighted_quadratic_mean, weighted_qubic_mean, weighted_geometric_mean, weighted_harmonic_mean, weighted_arithmetic_mean
                )

    return (
        arithmetic_mean, weighted_arithmetic_mean, quadratic_mean, qubic_mean, geometric_mean, harmonic_mean,
        weighted_quadratic_mean, weighted_qubic_mean, weighted_geometric_mean, weighted_harmonic_mean,
    )


# cache=False overrides NUMBA_NJIT_PARAMS for this kernel only: numba's AOT
# cache for functions with many bool kwargs corrupts on Windows (Python 3.11
# + numba 0.59) -- a fresh process that calls this with all kwargs explicit
# loads a stale .nbc compilation and segfaults with an access violation.
# Other kernels in this module are unaffected by the bug and keep cache=True.
@numba.njit(**{**NUMBA_NJIT_PARAMS, "cache": False})
def compute_numerical_aggregates_numba(
    arr: np.ndarray,
    weights: Optional[np.ndarray] = None,
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
    # A hardcoded [0.0] here broke the documented fixed-width output contract get_basic_feature_names()
    # relies on for column-name alignment: the returned vector's length must depend only on which
    # return_* flags are set, not on whether arr happened to be empty. Recurse on a single degenerate
    # zero element instead (size=1, so this branch cannot recurse again) -- it flows through the exact
    # same flag-driven branches as any real call, producing a correctly-sized (if meaningless) result.
    if size == 0:
        # `cast()` is NOT numba-nopython-compatible (this function body is @njit-compiled) --
        # `# type: ignore` is a comment, invisible to numba's AST pass, so it is the only mypy
        # satisfier usable inside this function.
        return compute_numerical_aggregates_numba(  # type: ignore[no-any-return]
            np.zeros(1, dtype=arr.dtype),
            weights if weights is None else np.ones(1, dtype=weights.dtype),
            geomean_log_mode,
            directional_only,
            whiten_means,
            return_drawdown_stats,
            return_profit_factor,
            return_n_zer_pos_int,
            return_exotic_means,
            return_unsorted_stats,
        )

    first = arr[0]
    last = arr[-1]
    if first != 0.0:
        last_to_first = last / first
    else:
        last_to_first = LARGE_CONST * np.sign(last)

    if directional_only:
        arithmetic_mean = np.mean(arr)
        return [arithmetic_mean, last_to_first]

    (
        arithmetic_mean, weighted_arithmetic_mean, quadratic_mean, qubic_mean, geometric_mean, harmonic_mean,
        weighted_quadratic_mean, weighted_qubic_mean, weighted_geometric_mean, weighted_harmonic_mean, sum_weights,
        minimum, maximum, min_index, max_index, nmaxupdates, nminupdates, n_last_crossings, n_last_touches,
        cnt_nonzero, npositive, ninteger, sum_positive, sum_negative, geomean_log_mode, pos_dds, pos_dd_durs, neg_dds, neg_dd_durs,
    ) = _aggregate_scan(arr, weights, first, last, geomean_log_mode, return_drawdown_stats, return_unsorted_stats, return_exotic_means, return_n_zer_pos_int)

    (
        arithmetic_mean, weighted_arithmetic_mean, quadratic_mean, qubic_mean, geometric_mean, harmonic_mean,
        weighted_quadratic_mean, weighted_qubic_mean, weighted_geometric_mean, weighted_harmonic_mean,
    ) = _finish_aggregate_means(
        arr, weights, size, whiten_means, return_exotic_means, geomean_log_mode, arithmetic_mean, weighted_arithmetic_mean, sum_weights, quadratic_mean,
        qubic_mean, geometric_mean, harmonic_mean, weighted_quadratic_mean, weighted_qubic_mean, weighted_geometric_mean, weighted_harmonic_mean,
        minimum, maximum, npositive,
    )

    res = _assemble_aggregates(
        weights is not None, arithmetic_mean, weighted_arithmetic_mean, first, minimum, maximum, last_to_first, min_index, max_index, size,
        nmaxupdates, nminupdates, n_last_crossings, n_last_touches, quadratic_mean, qubic_mean, geometric_mean, harmonic_mean,
        cnt_nonzero, npositive, ninteger, weighted_quadratic_mean, weighted_qubic_mean, weighted_geometric_mean, weighted_harmonic_mean,
        sum_positive, sum_negative, return_unsorted_stats, return_exotic_means, return_n_zer_pos_int, return_profit_factor,
    )

    if return_drawdown_stats:
        _weights_tail = weights if weights is None else weights[1:]
        res.extend(
            compute_numerical_aggregates_numba(
                pos_dds[1:], _weights_tail, geomean_log_mode, directional_only, whiten_means, False, False,
                return_n_zer_pos_int, return_exotic_means, return_unsorted_stats,
            )
        )
        res.extend(
            compute_numerical_aggregates_numba(
                pos_dd_durs[1:] / (size - 1), _weights_tail, geomean_log_mode, directional_only, whiten_means, False, False,
                return_n_zer_pos_int, return_exotic_means, return_unsorted_stats,
            )
        )
        res.extend(
            compute_numerical_aggregates_numba(
                neg_dds[1:], _weights_tail, geomean_log_mode, directional_only, whiten_means, False, False,
                return_n_zer_pos_int, return_exotic_means, return_unsorted_stats,
            )
        )
        res.extend(
            compute_numerical_aggregates_numba(
                neg_dd_durs[1:] / (size - 1), _weights_tail, geomean_log_mode, directional_only, whiten_means, False, False,
                return_n_zer_pos_int, return_exotic_means, return_unsorted_stats,
            )
        )

    return list(res)


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
        weights: Optional[np.ndarray] = None,
        weighted_mean_value: Optional[float] = None,
        xvals: Optional[np.ndarray] = None,
        directional_only: bool = False,
        return_lintrend_approx_stats: bool = True,
    ) -> tuple:  # pragma: no cover
        """Compiled kernel body: computes mad/std/skew/kurt moments plus over/under linear-trend slopes (optionally weighted, optionally Kahan-compensated per the enclosing factory's ``KAHAN``/fastmath compile-time constants) in one fused pass."""
        slope_over, slope_under = 0.0, 0.0
        mad, std, skew, kurt = 0.0, 0.0, 0.0, 0.0
        # Kahan compensation counters. When KAHAN=False these stay 0.0 and get DCE'd along with
        # every `if KAHAN: ... else: ...` block in the loop, so the fast variant pays nothing.
        slope_over_c, slope_under_c, r_sum_c, mad_c, std_c, skew_c, kurt_c = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        # Weighted accumulators exist on every path (the weighted branches are guarded by ``weights is not None``).
        sum_weights, weighted_mad, weighted_std, weighted_skew, weighted_kurt = 0.0, 0.0, 0.0, 0.0, 0.0
        sum_weights_c, weighted_mad_c, weighted_std_c, weighted_skew_c, weighted_kurt_c = 0.0, 0.0, 0.0, 0.0, 0.0

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
            slope_over, slope_over_c = _kahan_add(KAHAN, slope_over, slope_over_c, _inc)

            # slope_under += sl_x**2
            _inc = sl_x * sl_x
            slope_under, slope_under_c = _kahan_add(KAHAN, slope_under, slope_under_c, _inc)

            d = next_value - mean_value

            # r_sum += sl_x * d
            _inc = sl_x * d
            r_sum, r_sum_c = _kahan_add(KAHAN, r_sum, r_sum_c, _inc)

            if has_prev_d and d * prev_d < 0:
                n_mean_crossings += 1
            prev_d = d
            has_prev_d = True

            # mad += abs(d)
            _inc = abs(d)
            mad, mad_c = _kahan_add(KAHAN, mad, mad_c, _inc)

            if weights is not None:
                next_weight = weights[i]
                w_d = next_value - weighted_mean_value

                # sum_weights += next_weight
                sum_weights, sum_weights_c = _kahan_add(KAHAN, sum_weights, sum_weights_c, next_weight)

                # weighted_mad += abs(w_d) * next_weight
                _inc = abs(w_d) * next_weight
                weighted_mad, weighted_mad_c = _kahan_add(KAHAN, weighted_mad, weighted_mad_c, _inc)

            summand = d * d
            # std += summand
            std, std_c = _kahan_add(KAHAN, std, std_c, summand)

            if weights is not None:
                w_summand = w_d * w_d
                # weighted_std += w_summand * next_weight
                _inc = w_summand * next_weight
                weighted_std, weighted_std_c = _kahan_add(KAHAN, weighted_std, weighted_std_c, _inc)

            if not directional_only:

                summand = summand * d
                # skew += summand (d^3)
                skew, skew_c = _kahan_add(KAHAN, skew, skew_c, summand)

                if weights is not None:
                    w_summand = w_summand * w_d
                    # weighted_skew += w_summand * next_weight
                    _inc = w_summand * next_weight
                    weighted_skew, weighted_skew_c = _kahan_add(KAHAN, weighted_skew, weighted_skew_c, _inc)

                # kurt += summand * d (d^4)
                _inc = summand * d
                kurt, kurt_c = _kahan_add(KAHAN, kurt, kurt_c, _inc)

                if weights is not None:
                    # Was `weighted_skew +=` here in the original buggy version: double-
                    # accumulating skew while weighted_kurt stayed 0 -> constant -3.0 feature.
                    _inc = w_summand * w_d * next_weight
                    weighted_kurt, weighted_kurt_c = _kahan_add(KAHAN, weighted_kurt, weighted_kurt_c, _inc)

        (
            slope_over, slope_under, r_sum, mad, std, skew, kurt, weighted_mad, weighted_std, weighted_skew, weighted_kurt,
        ) = _finish_moments(
            KAHAN, directional_only, weights is not None, size, slope_over, slope_over_c, slope_under, slope_under_c, r_sum, r_sum_c, mad, mad_c,
            std, std_c, skew, skew_c, kurt, kurt_c, sum_weights, sum_weights_c, weighted_mad, weighted_mad_c, weighted_std, weighted_std_c,
            weighted_skew, weighted_skew_c, weighted_kurt, weighted_kurt_c,
        )

        slope, intercept, r, n_lintrend_crossings, lintrend_data_diffs = _linear_trend_stats(
            arr, xvals, mean_value, xvals_mean, slope_over, slope_under, r_sum, std, size, return_lintrend_approx_stats
        )

        res: list = []
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
    weights: Optional[np.ndarray] = None,
    weighted_mean_value: Optional[float] = None,
    xvals: Optional[np.ndarray] = None,
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
        return cast(
            tuple,
            _compute_moments_slope_mi_compensated(
                arr=arr,
                mean_value=mean_value,
                weights=weights,
                weighted_mean_value=weighted_mean_value,
                xvals=xvals,
                directional_only=directional_only,
                return_lintrend_approx_stats=return_lintrend_approx_stats,
            ),
        )
    # NaN-gate: see compute_simple_stats_numba for rationale.
    if not np.isfinite(arr).all() or (weights is not None and not np.isfinite(weights).all()):
        return cast(
            tuple,
            _compute_moments_slope_mi_compensated(
                arr=arr,
                mean_value=mean_value,
                weights=weights,
                weighted_mean_value=weighted_mean_value,
                xvals=xvals,
                directional_only=directional_only,
                return_lintrend_approx_stats=return_lintrend_approx_stats,
            ),
        )
    return cast(
        tuple,
        _compute_moments_slope_mi_fast(
            arr=arr,
            mean_value=mean_value,
            weights=weights,
            weighted_mean_value=weighted_mean_value,
            xvals=xvals,
            directional_only=directional_only,
            return_lintrend_approx_stats=return_lintrend_approx_stats,
        ),
    )


_EMPTY_FLOAT32 = np.array([], dtype=np.float32)
