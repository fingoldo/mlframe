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

# 2026-05-21: NUMBA_NJIT_PARAMS + the numeric constants used inside the
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
    GEOMEAN_OVERFLOW_HI,
    GEOMEAN_OVERFLOW_LO,
)


import numba
import numpy as np


@numba.njit(**NUMBA_NJIT_PARAMS)
def _signed_cbrt(x):  # pragma: no cover
    """Sign-preserving cube root: a bare ``(-x)**(1/3)`` is NaN, so route through abs()+sign."""
    return np.sign(x) * np.abs(x) ** (1 / 3)


@numba.njit(**NUMBA_NJIT_PARAMS)
def _scaled_power_means(arr, minimum, maximum, size):  # pragma: no cover
    """Overflow-safe (LAPACK dlassq-style) quadratic and cubic means, keyed on max|x|; ``(0, 0)`` for an all-zero column."""
    _scale = max(abs(minimum), abs(maximum))
    if _scale > 0.0:
        _inv = 1.0 / _scale
        _ssq, _scube = 0.0, 0.0
        for _v in arr:
            _xs = _v * _inv
            _xs2 = _xs * _xs
            _ssq += _xs2
            _scube += _xs2 * _xs
        _qm = _scube / size
        return _scale * np.sqrt(_ssq / size), _scale * np.sign(_qm) * np.abs(_qm) ** (1 / 3)
    return 0.0, 0.0


@numba.njit(**NUMBA_NJIT_PARAMS)
def _scaled_weighted_power_means(arr, weights, minimum, maximum, sum_weights, size):  # pragma: no cover
    """Weighted counterpart of ``_scaled_power_means``."""
    _scale = max(abs(minimum), abs(maximum))
    if _scale > 0.0:
        _inv = 1.0 / _scale
        _wssq, _wscube = 0.0, 0.0
        for _i in range(size):
            _xs = arr[_i] * _inv
            _xs2 = _xs * _xs
            _wssq += weights[_i] * _xs2
            _wscube += weights[_i] * _xs2 * _xs
        _wqm = _wscube / sum_weights
        return _scale * np.sqrt(_wssq / sum_weights), _scale * np.sign(_wqm) * np.abs(_wqm) ** (1 / 3)
    return 0.0, 0.0


@numba.njit(**NUMBA_NJIT_PARAMS)
def _finish_power_means(arr, quadratic_sum, qubic_sum, minimum, maximum, size):  # pragma: no cover
    """Quadratic and cubic means from their raw sums; the naive sums of x^2 / x^3 overflow to inf for extreme-scale columns (|x|~1e154 -> x^2,
    |x|~1e103 -> x^3) even though the true power-mean is finite, so that rare case is recomputed via the scaled pass. The common finite path is the
    naive one, with no added cost when nothing overflowed."""
    if not np.isfinite(quadratic_sum) or not np.isfinite(qubic_sum):
        return _scaled_power_means(arr, minimum, maximum, size)
    return np.sqrt(quadratic_sum / size), _signed_cbrt(qubic_sum / size)


@numba.njit(**NUMBA_NJIT_PARAMS)
def _finish_weighted_power_means(arr, weights, quadratic_sum, qubic_sum, minimum, maximum, sum_weights, size):  # pragma: no cover
    """Weighted ``_finish_power_means``; a zero-sum weights vector gives NaN instead of dividing by 0."""
    if sum_weights == 0.0:
        return np.nan, np.nan
    if not np.isfinite(quadratic_sum) or not np.isfinite(qubic_sum):
        return _scaled_weighted_power_means(arr, weights, minimum, maximum, sum_weights, size)
    return np.sqrt(quadratic_sum / sum_weights), _signed_cbrt(qubic_sum / sum_weights)


@numba.njit(**NUMBA_NJIT_PARAMS)
def _geometric_mean_step(geometric_mean, weighted_geometric_mean, next_value, next_weight, has_weights, geomean_log_mode):  # pragma: no cover
    """Fold one positive sample into the (weighted) running geometric mean; flips to log mode once either product leaves the safe range.

    Returns ``(geometric_mean, weighted_geometric_mean, geomean_log_mode)``.
    """
    if geomean_log_mode:
        addend = np.log(next_value)
        geometric_mean += addend
        if has_weights:
            weighted_geometric_mean += next_weight * addend
        return geometric_mean, weighted_geometric_mean, geomean_log_mode
    geometric_mean *= next_value
    if has_weights:
        weighted_geometric_mean *= next_value**next_weight
    # Check BOTH unweighted and weighted products: either can underflow / overflow independently.
    # A weighted product can hit the tail much faster when |next_weight| > 1.
    unweighted_oor = geometric_mean >= GEOMEAN_OVERFLOW_HI or geometric_mean <= GEOMEAN_OVERFLOW_LO
    weighted_oor = has_weights and (weighted_geometric_mean >= GEOMEAN_OVERFLOW_HI or weighted_geometric_mean <= GEOMEAN_OVERFLOW_LO)
    if unweighted_oor or weighted_oor:
        # convert to log mode (geometric_mean strictly positive here; log is finite)
        geomean_log_mode = True
        geometric_mean = np.log(float(geometric_mean)) if geometric_mean > 0 else -np.inf
        if has_weights:
            weighted_geometric_mean = np.log(float(weighted_geometric_mean)) if weighted_geometric_mean > 0 else -np.inf
    return geometric_mean, weighted_geometric_mean, geomean_log_mode


@numba.njit(**NUMBA_NJIT_PARAMS)
def _finish_geometric_mean(geometric_mean, geomean_log_mode, n):  # pragma: no cover
    """Geometric mean from its running product (or log-sum) over ``n`` samples (``size`` or the weights sum)."""
    if not geomean_log_mode:
        return geometric_mean ** (1 / n)
    return np.exp(geometric_mean / n)


@numba.njit(**NUMBA_NJIT_PARAMS)
def _crossing_step(next_value, last, has_prev_d, prev_d, n_last_crossings, n_last_touches):  # pragma: no cover
    """Count one sample's crossing / touching of the ``last`` level: a sign change of ``(x - last)`` between consecutive samples is a crossing, a zero
    product (or the first sample equal to ``last``) a touch. Returns the updated ``(n_last_crossings, n_last_touches)``."""
    if has_prev_d:
        mul = (next_value - last) * prev_d
        if mul < 0:
            n_last_crossings += 1
        elif mul == 0.0:
            n_last_touches += 1
    elif next_value == last:
        n_last_touches += 1
    return n_last_crossings, n_last_touches


@numba.njit(**NUMBA_NJIT_PARAMS)
def _drawdown_step(i, dd, start_idx, dds, durs):  # pragma: no cover
    """Record sample ``i``'s drawdown ``dd`` and its duration since the last zero-drawdown index; returns the (possibly advanced) start index."""
    dds[i] = dd
    if dd == 0.0:
        start_idx = i
    durs[i] = i - start_idx
    return start_idx


@numba.njit(**NUMBA_NJIT_PARAMS)
def _whiten(quadratic_mean, qubic_mean, geometric_mean, harmonic_mean, arithmetic_mean):  # pragma: no cover
    """The four exotic means expressed relative to the arithmetic mean."""
    return quadratic_mean - arithmetic_mean, qubic_mean - arithmetic_mean, geometric_mean - arithmetic_mean, harmonic_mean - arithmetic_mean


@numba.njit(**NUMBA_NJIT_PARAMS)
def _nonzero_step(next_value, next_weight, has_weights, return_exotic_means, return_n_zer_pos_int, harmonic_mean, weighted_harmonic_mean, ninteger):  # pragma: no cover
    """Fold one nonzero sample into the running (weighted) harmonic-mean sum and the integer-valued count; returns ``(harmonic_mean, weighted_harmonic_mean, ninteger)``."""
    if return_exotic_means:
        addend = 1 / next_value
        harmonic_mean += addend
        if has_weights:
            weighted_harmonic_mean += next_weight * addend
    if return_n_zer_pos_int:
        # Was `next_value % 1` -- robust for positive floats but fragile around negative
        # values and denormals. `np.floor(x) == x` is the exact integer check.
        if np.floor(next_value) == next_value:
            ninteger = ninteger + 1
    return harmonic_mean, weighted_harmonic_mean, ninteger


@numba.njit(**NUMBA_NJIT_PARAMS)
def _finish_harmonic_mean(harmonic_sum, n):  # pragma: no cover
    """Harmonic mean from the running sum of reciprocals over ``n`` samples (or the weights sum); NaN when the sum is zero."""
    if harmonic_sum:
        return n / harmonic_sum
    return np.nan


@numba.njit(**NUMBA_NJIT_PARAMS)
def _finish_weighted_geometric_mean(weighted_geometric_mean, geomean_log_mode, sum_weights, npositive):  # pragma: no cover
    """Weighted geometric mean; NaN without positive samples or with a zero weights sum, and an overflow to inf is reported as 0."""
    if npositive and sum_weights != 0.0:
        weighted_geometric_mean = _finish_geometric_mean(weighted_geometric_mean, geomean_log_mode, sum_weights)
        if weighted_geometric_mean == np.inf:
            weighted_geometric_mean = 0.0
        return weighted_geometric_mean
    return np.nan


@numba.njit(**NUMBA_NJIT_PARAMS)
def _ratio_stats(arithmetic_mean, first, minimum, maximum, last_to_first):  # pragma: no cover
    """Mean, first and minimum expressed as ratios (LARGE_CONST-signed when the denominator is zero), plus the already computed last/first."""
    return (
        arithmetic_mean / first if first else LARGE_CONST * np.sign(arithmetic_mean),
        first / maximum if maximum else LARGE_CONST * np.sign(first),
        minimum / first if first else LARGE_CONST * np.sign(minimum),
        last_to_first,
    )


@numba.njit(**NUMBA_NJIT_PARAMS)
def _relative_extrema_positions(min_index, max_index, size):  # pragma: no cover
    """Positions of the minimum and maximum as a fraction of the series length."""
    return (min_index + 1) / size if size else 0, (max_index + 1) / size if size else 0


@numba.njit(**NUMBA_NJIT_PARAMS)
def _profit_factor(sum_positive, sum_negative):  # pragma: no cover
    """Gross profit over gross loss; 0 with no activity, LARGE_CONST with profit and no loss."""
    if sum_negative != 0.0:
        return sum_positive / -sum_negative
    return 0.0 if sum_positive == 0.0 else LARGE_CONST


@numba.njit(**NUMBA_NJIT_PARAMS)
def _assemble_aggregates(  # pragma: no cover
    has_weights, arithmetic_mean, weighted_arithmetic_mean, first, minimum, maximum, last_to_first, min_index, max_index, size,
    nmaxupdates, nminupdates, n_last_crossings, n_last_touches, quadratic_mean, qubic_mean, geometric_mean, harmonic_mean,
    cnt_nonzero, npositive, ninteger, weighted_quadratic_mean, weighted_qubic_mean, weighted_geometric_mean, weighted_harmonic_mean,
    sum_positive, sum_negative, return_unsorted_stats, return_exotic_means, return_n_zer_pos_int, return_profit_factor,
):
    """Lay the computed aggregates out in the fixed order ``get_basic_feature_names`` relies on; which groups appear depends only on the flags."""
    res = [arithmetic_mean]
    if has_weights:
        res.append(weighted_arithmetic_mean)
    res.extend(
        (
            minimum,
            maximum,
        )
    )  # can't combine with the next statement as it's failing on integer inputs due to tuple dtypes mismatch
    res.extend(_ratio_stats(arithmetic_mean, first, minimum, maximum, last_to_first))

    if return_unsorted_stats:  # must be false for arrays known to be sorted
        res.extend(_relative_extrema_positions(min_index, max_index, size))
        res.extend((nmaxupdates, nminupdates, n_last_crossings, n_last_touches - 1))

    if return_exotic_means:
        res.extend((quadratic_mean, qubic_mean, geometric_mean, harmonic_mean))

    if return_n_zer_pos_int:
        res.extend((cnt_nonzero, npositive, ninteger))

    if has_weights:
        if return_exotic_means:
            res.extend((weighted_quadratic_mean, weighted_qubic_mean, weighted_geometric_mean, weighted_harmonic_mean))

    if return_profit_factor:
        res.append(_profit_factor(sum_positive, sum_negative))
    return res


@numba.njit(**NUMBA_NJIT_PARAMS)
def _extrema_step(i, next_value, minimum, min_index, nminupdates, maximum, max_index, nmaxupdates):  # pragma: no cover
    """Update the running minimum / maximum, their indices and refresh counts with sample ``i``."""
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
    return minimum, min_index, nminupdates, maximum, max_index, nmaxupdates


@numba.njit(**NUMBA_NJIT_PARAMS)
def _power_sums_step(next_value, next_weight, has_weights, quadratic_sum, qubic_sum, weighted_quadratic_sum, weighted_qubic_sum):  # pragma: no cover
    """Fold one sample's square and cube into the running (weighted) sums."""
    temp_value = next_value * next_value
    quadratic_sum += temp_value
    if has_weights:
        weighted_quadratic_sum += temp_value * next_weight

    temp_value = temp_value * next_value
    qubic_sum += temp_value
    if has_weights:
        weighted_qubic_sum += temp_value * next_weight
    return quadratic_sum, qubic_sum, weighted_quadratic_sum, weighted_qubic_sum


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

    ninteger, npositive, cnt_nonzero = 0, 0, 0
    sum_positive, sum_negative = 0.0, 0.0

    if not geomean_log_mode:
        geometric_mean = 1.0
    else:
        geometric_mean = 0.0

    arithmetic_mean, quadratic_mean, qubic_mean, harmonic_mean = 0.0, 0.0, 0.0, 0.0
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
                arr=pos_dds[1:],
                weights=_weights_tail,
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
                weights=_weights_tail,
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
                weights=_weights_tail,
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
                weights=_weights_tail,
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

    return list(res)


# The helpers below are inlined (``inline="always"``) into the moments kernel, so they are compiled under the caller's fastmath setting -- a separately
# compiled helper has its own and changes the fast variant's rounding. They are module-level rather than closure-captured because a kernel that closes over
# other dispatchers cannot be reloaded from numba's on-disk cache ("No module named '<dynamic>'").
@numba.njit(inline="always", **NUMBA_NJIT_PARAMS)
def _kahan_add(kahan, total, comp, inc):  # pragma: no cover
    """``total + inc`` with the running compensation ``comp`` updated when ``kahan`` (a compile-time constant in the kernel, so the fast variant is a plain
    add and ``comp`` passes through untouched); returns ``(total, comp)``."""
    if kahan:
        _t = total + inc
        if abs(total) >= abs(inc):
            comp += (total - _t) + inc
        else:
            comp += (inc - _t) + total
        return _t, comp
    return total + inc, comp


@numba.njit(inline="always", **NUMBA_NJIT_PARAMS)
def _standardise_moments(skew, kurt, std, n):  # pragma: no cover
    """Skewness and excess kurtosis from the raw third / fourth central-moment sums over ``n`` samples (or the weights sum); ``(0, 0)`` for a
    constant column, and the sums are returned unchanged when the scale factor is zero."""
    if std == 0:
        return 0.0, 0.0
    factor = n * std**3
    if factor:
        skew = skew / factor

        factor = factor * std
        kurt = kurt / factor - 3.0
    return skew, kurt


@numba.njit(inline="always", **NUMBA_NJIT_PARAMS)
def _linear_trend_stats(arr, xvals, mean_value, xvals_mean, slope_over, slope_under, r_sum, std, size, return_lintrend_approx_stats):  # pragma: no cover
    """OLS slope / intercept / correlation of ``arr`` against ``xvals`` from the accumulated sums, the number of sign changes of the residuals, and
    (optionally) the residual vector. A degenerate x spread gives ``r=0`` and NaN everywhere else.

    Returns ``(slope, intercept, r, n_lintrend_crossings, lintrend_data_diffs)``.
    """
    if np.isclose(slope_under, 0) or np.isnan(slope_under):
        return np.nan, np.nan, 0.0, np.nan, None
    slope = slope_over / slope_under

    # R-value
    if np.isclose(std, 0):
        r = 0.0
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
            lintrend_data_diffs[i] = d  # allocated iff return_lintrend_approx_stats, same guard as here
    return slope, intercept, r, n_lintrend_crossings, lintrend_data_diffs


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
            # sum_weights==0 (all-zero weight column)
            # used to crash the njit kernel here.
            if sum_weights == 0.0:
                weighted_std = np.nan
            else:
                weighted_std = np.sqrt(weighted_std / sum_weights)

        if not directional_only:
            mad = mad / size

            skew, kurt = _standardise_moments(skew, kurt, std, size)

            if weights is not None:
                # Same sum_weights==0 guard.
                if sum_weights == 0.0:
                    weighted_mad = np.nan
                else:
                    weighted_mad = weighted_mad / sum_weights
                if weighted_std == 0:
                    weighted_skew, weighted_kurt = 0.0, 0.0
                else:
                    # ``sum_weights``, not ``size``. The accumulators above sum ``w_i * d_i**k``, so the
                    # weighted moment is that divided by the total WEIGHT -- which is what ``weighted_std`` and
                    # ``weighted_mad`` in this same block already divide by. Dividing by the row count instead
                    # scaled both statistics by ``sum_weights / size``: with weights normalised to sum to 1 at
                    # n=200 that is a factor of 200, and the excess kurtosis then collapsed toward the constant
                    # -3.0 -- the same signature the comment forty lines up records from an earlier bug here.
                    factor = sum_weights * weighted_std**3
                    if factor:
                        weighted_skew = weighted_skew / factor

                        factor = factor * weighted_std
                        weighted_kurt = weighted_kurt / factor - 3.0

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
