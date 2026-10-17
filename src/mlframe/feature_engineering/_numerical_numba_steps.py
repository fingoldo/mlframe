"""Small ``@njit`` step/finish helpers for the numerical-aggregates and moments kernels in ``_numerical_numba``.

Split out of ``_numerical_numba.py`` to keep that module under the file-size budget; every helper is re-imported there by name.
"""

from __future__ import annotations

import numba
import numpy as np

from ._numerical_constants import (
    NUMBA_NJIT_PARAMS,
    LARGE_CONST,
    GEOMEAN_OVERFLOW_HI,
    GEOMEAN_OVERFLOW_LO,
)


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
def _finish_moments(  # pragma: no cover
    kahan, directional_only, has_weights, size, slope_over, slope_over_c, slope_under, slope_under_c, r_sum, r_sum_c, mad, mad_c,
    std, std_c, skew, skew_c, kurt, kurt_c, sum_weights, sum_weights_c, weighted_mad, weighted_mad_c, weighted_std, weighted_std_c,
    weighted_skew, weighted_skew_c, weighted_kurt, weighted_kurt_c,
):
    """Close the moments accumulation: fold in the Kahan compensations, then turn the raw sums into std / mad / skew / kurt (and their weighted twins).

    Returns ``(slope_over, slope_under, r_sum, mad, std, skew, kurt, weighted_mad, weighted_std, weighted_skew, weighted_kurt)``.
    """
    # Apply Kahan corrections once at the end. DCE'd when kahan=False.
    if kahan:
        slope_over += slope_over_c
        slope_under += slope_under_c
        r_sum += r_sum_c
        mad += mad_c
        std += std_c
        skew += skew_c
        kurt += kurt_c

    std = np.sqrt(std / size)
    if has_weights:
        if kahan:
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

        if has_weights:
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
    return slope_over, slope_under, r_sum, mad, std, skew, kurt, weighted_mad, weighted_std, weighted_skew, weighted_kurt


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
