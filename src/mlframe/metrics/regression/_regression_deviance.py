"""Tweedie / Poisson / Gamma GLM deviance metrics.

Carved from ``_regression_extras.py`` (Tier-2 non-Gaussian deviances) so the
parent stays under the LOC ceiling. Re-exported from the parent + ``metrics.core``.
"""
from __future__ import annotations

from math import log
from typing import Tuple

import numpy as np
import numba

from .._numba_params import NUMBA_NJIT_PARAMS

# ============================================================================
# Tweedie / Poisson / Gamma deviances (Tier 2)
# ============================================================================
#
# GLM deviances for non-Gaussian targets:
#   power=0: Normal (= MSE)
#   power=1: Poisson (count data, insurance claim counts, click counts)
#   power=2: Gamma (positive continuous, claim severity, lifetime)
#   power=3: Inverse Gaussian
#   1 < power < 2: compound Poisson-Gamma (insurance pure-premium)
# Formula follows sklearn's mean_tweedie_deviance exactly (Pregibon 1984):
#   D(y, p) = 2 * mean of:
#     power=0:   (y - p)^2
#     power=1:   y * log(y/p) - (y - p)              (with 0*log0 = 0)
#     power=2:   log(p/y) + (y - p)/p
#     general:   max(y,0)^(2-p)/((1-p)(2-p)) - y*p^(1-p)/(1-p) + p^(2-p)/(2-p)
# All require strictly positive predictions for p>=1, and non-negative y;
# kernels return NaN with a count of skipped rows when constraints break.
#
# Denominator convention: out-of-support rows (y_pred<=0 or y outside the distribution's support) are excluded
# from the SUM (their per-row term is mathematically undefined) but the divisor is still the FULL row count,
# not the surviving count. Dividing by the surviving count alone would make deviance numbers incomparable
# across two models with different invalid-row counts on the same eval set (a model that emits more
# out-of-support predictions would get its deviance averaged over a smaller, easier subset, biasing model
# selection toward whichever model excludes its hardest rows). Dividing by the full count keeps two runs on
# the same denominator; invalid rows implicitly contribute 0 to the sum rather than being dropped from both
# sides of the ratio.


@numba.njit(**NUMBA_NJIT_PARAMS)
def _tweedie_deviance_poisson_kernel(
    y_true: np.ndarray, y_pred: np.ndarray,
) -> Tuple[float, int]:
    """Returns (deviance, count_invalid). Skips rows where y_pred<=0 OR y<0.
    y=0 is fine: y*log(y) is taken as 0 by convention."""
    s = 0.0
    used = 0
    invalid = 0
    for i in range(y_true.shape[0]):
        yt = y_true[i]
        yp = y_pred[i]
        if yp <= 0.0 or yt < 0.0:
            invalid += 1
            continue
        if yt == 0.0:
            term = -(yt - yp)  # = yp; y*log(y/p) is 0
        else:
            term = yt * log(yt / yp) - (yt - yp)
        s += 2.0 * term
        used += 1
    total = y_true.shape[0]
    return (s / total) if used > 0 else np.nan, invalid


@numba.njit(**NUMBA_NJIT_PARAMS)
def _tweedie_deviance_gamma_kernel(
    y_true: np.ndarray, y_pred: np.ndarray,
) -> Tuple[float, int]:
    """Gamma deviance: power=2. Requires y > 0 AND p > 0 (the log
    is undefined at 0 for both)."""
    s = 0.0
    used = 0
    invalid = 0
    for i in range(y_true.shape[0]):
        yt = y_true[i]
        yp = y_pred[i]
        if yp <= 0.0 or yt <= 0.0:
            invalid += 1
            continue
        s += 2.0 * (log(yp / yt) + (yt - yp) / yp)
        used += 1
    total = y_true.shape[0]
    return (s / total) if used > 0 else np.nan, invalid


@numba.njit(**NUMBA_NJIT_PARAMS)
def _tweedie_deviance_general_kernel(
    y_true: np.ndarray, y_pred: np.ndarray, power: float,
) -> Tuple[float, int]:
    """General 1 < power < 2 OR power > 2 case. Sklearn convention
    requires y >= 0 (negative not in support of distribution) AND p > 0
    (predictions must be strictly positive for log/power terms)."""
    s = 0.0
    used = 0
    invalid = 0
    p = power
    # yp**(2-p) == yp**(1-p) * yp, so one variable-exponent pow (an exp(log) under the hood) is dropped per row; the per-call constant factors are hoisted out of the loop too.
    e1 = 1.0 - p
    e2 = 2.0 - p
    c_y = 1.0 / (e1 * e2)
    c_yp = 1.0 / e1
    c_p = 1.0 / e2
    for i in range(y_true.shape[0]):
        yt = y_true[i]
        yp = y_pred[i]
        if yp <= 0.0 or yt < 0.0:
            invalid += 1
            continue
        # max(y, 0)^(2-p): yt >= 0 here, so use yt directly.
        if yt == 0.0:
            term_y = 0.0  # 0^(2-p) for 1<p<2: power > 0 -> 0; for p>2: power <0 -> inf, but
            # sklearn special-cases this to 0 (sup-zero discontinuity).
        else:
            term_y = (yt**e2) * c_y
        yp_pow1 = yp**e1
        term_yp = yt * yp_pow1 * c_yp
        term_p = (yp_pow1 * yp) * c_p
        s += 2.0 * (term_y - term_yp + term_p)
        used += 1
    total = y_true.shape[0]
    return (s / total) if used > 0 else np.nan, invalid


_TWEEDIE_WARN_COUNTS: dict = {}


def _maybe_warn_tweedie(name: str, invalid: int, total: int) -> None:
    """Emit a RuntimeWarning when a Tweedie/Poisson/Gamma deviance skipped out-of-support rows.

    Tracks an occurrence COUNT per (name, invalid, total) rather than a seen/unseen set: the first
    occurrence always warns, and every subsequent occurrence with the exact same counts (e.g. the same
    fold shape recurring across CV) also warns, with the running occurrence count in the message, so
    recurrence across folds stays visible instead of being silently swallowed after the first hit.
    """
    if invalid <= 0:
        return
    key = (name, int(invalid), int(total))
    count = _TWEEDIE_WARN_COUNTS.get(key, 0) + 1
    _TWEEDIE_WARN_COUNTS[key] = count
    import warnings
    suffix = "" if count == 1 else f" (recurrence #{count} with the same counts)"
    warnings.warn(
        f"{name}: {invalid} of {total} rows skipped (y_pred<=0 or y_true out of support); "
        f"check that the model emits strictly positive predictions matching the target's "
        f"distributional support.{suffix}",
        RuntimeWarning, stacklevel=3,
    )


def fast_poisson_deviance(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Tweedie deviance at power=1 (Poisson).

    Use for count targets where the variance scales with the mean
    (claim counts, calls per hour, click counts). Lower is better.
    Equivalent to sklearn's ``mean_poisson_deviance``.

    Rows with y_pred <= 0 or y_true < 0 are skipped with a rate-limited
    warning - silently dropping them masks model misspecification.
    """
    yt = np.ascontiguousarray(y_true, dtype=np.float64)
    yp = np.ascontiguousarray(y_pred, dtype=np.float64)
    if yt.shape[0] == 0:
        return np.nan
    val, invalid = _tweedie_deviance_poisson_kernel(yt, yp)
    _maybe_warn_tweedie("fast_poisson_deviance", invalid, yt.shape[0])
    return float(val)


def fast_gamma_deviance(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Tweedie deviance at power=2 (Gamma).

    Use for positive continuous targets where the variance scales with
    the mean SQUARED (claim severity, lifetimes, financial losses).
    Lower is better. Equivalent to sklearn's ``mean_gamma_deviance``.

    Rows with y_pred <= 0 or y_true <= 0 are skipped (log is undefined
    at 0 for either) with a rate-limited warning.
    """
    yt = np.ascontiguousarray(y_true, dtype=np.float64)
    yp = np.ascontiguousarray(y_pred, dtype=np.float64)
    if yt.shape[0] == 0:
        return np.nan
    val, invalid = _tweedie_deviance_gamma_kernel(yt, yp)
    _maybe_warn_tweedie("fast_gamma_deviance", invalid, yt.shape[0])
    return float(val)


def fast_tweedie_deviance(
    y_true: np.ndarray, y_pred: np.ndarray, *, power: float = 0.0,
) -> float:
    """General Tweedie deviance at arbitrary power.

    Common values:
      power=0  -> Normal (= MSE) - falls through to the simple kernel
      power=1  -> Poisson (use ``fast_poisson_deviance`` for the dedicated path)
      power=2  -> Gamma   (use ``fast_gamma_deviance`` for the dedicated path)
      1<p<2    -> compound Poisson-Gamma (insurance pure-premium)

    Rows with y_pred <= 0 or y_true outside the distributional support
    are skipped with a rate-limited warning. Equivalent to sklearn's
    ``mean_tweedie_deviance(y_true, y_pred, power=power)``.
    """
    yt = np.ascontiguousarray(y_true, dtype=np.float64)
    yp = np.ascontiguousarray(y_pred, dtype=np.float64)
    if yt.shape[0] == 0:
        return np.nan
    if power == 0.0:
        d = yt - yp
        return float(np.mean(d * d))
    if power == 1.0:
        return fast_poisson_deviance(y_true, y_pred)
    if power == 2.0:
        return fast_gamma_deviance(y_true, y_pred)
    if power < 1.0:
        raise ValueError(f"Tweedie power must be 0 or >= 1 (Pregibon 1984); got power={power}. " "Use power=0 for Normal (MSE) or power in [1, inf) for GLM.")
    val, invalid = _tweedie_deviance_general_kernel(yt, yp, float(power))
    _maybe_warn_tweedie(f"fast_tweedie_deviance(power={power})", invalid, yt.shape[0])
    return float(val)
