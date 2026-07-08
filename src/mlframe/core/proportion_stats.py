"""Proportion (probability) confidence intervals and required sample size.

From PZAD «Оценки среднего, вероятности и плотности» (Дьяконов 2020, slides 45-47):
estimating a probability p = m/n is a mean of Bernoulli outcomes, and its estimate has
a precision that shrinks like 1/sqrt(n). The lecture's practical question — "n=10000 is
enough to estimate p to +/-0.01 at 99% confidence" — and the zodiac-scoring cautionary
tale (are the between-group differences even significant at the given n?) both need a
proportion confidence interval and a required-sample-size formula.

Provided:
- ``wilson_interval``: the Wilson score interval (better small-n / near-0/1 coverage than Wald).
- ``required_n_for_proportion``: n needed for a target half-width at a confidence level (worst-case p=0.5 or a supplied p).
- ``proportions_significantly_different``: do two observed proportions differ beyond their CIs (the zodiac check).
"""

from __future__ import annotations

import logging
import math

logger = logging.getLogger(__name__)

__all__ = ["wilson_interval", "required_n_for_proportion", "proportions_significantly_different", "z_for_confidence"]


def z_for_confidence(confidence: float) -> float:
    """Two-sided normal quantile z for a confidence level (e.g. 0.95 -> 1.96, 0.99 -> 2.576)."""
    if not (0.0 < confidence < 1.0):
        raise ValueError(f"z_for_confidence: confidence must be in (0, 1), got {confidence}.")
    # Inverse standard-normal CDF at (1 + confidence)/2 via the Acklam rational approximation (abs err < 1.15e-9).
    p = (1.0 + confidence) / 2.0
    a = [-3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02, 1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00]
    b = [-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02, 6.680131188771972e01, -1.328068155288572e01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00, -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00, 3.754408661907416e00]
    plow = 0.02425
    if p < plow:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    if p <= 1.0 - plow:
        q = p - 0.5
        r = q * q
        return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
    q = math.sqrt(-2.0 * math.log(1.0 - p))
    return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)


def wilson_interval(m: int, n: int, *, confidence: float = 0.95) -> tuple[float, float]:
    """Wilson score confidence interval for a proportion ``p = m/n`` (m successes of n trials).

    Better coverage than the Wald interval near 0/1 and at small n. Returns ``(low, high)`` clipped to [0, 1].
    """
    if n <= 0:
        raise ValueError("wilson_interval: n must be > 0.")
    if not (0 <= m <= n):
        raise ValueError("wilson_interval: require 0 <= m <= n.")
    z = z_for_confidence(confidence)
    phat = m / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (phat + z2 / (2 * n)) / denom
    half = (z / denom) * math.sqrt(phat * (1.0 - phat) / n + z2 / (4 * n * n))
    return max(0.0, center - half), min(1.0, center + half)


def required_n_for_proportion(half_width: float, *, confidence: float = 0.95, p: float = 0.5) -> int:
    """Sample size n so a proportion estimate has the given (Wald) half-width at the confidence level.

    ``n = ceil(z^2 * p*(1-p) / half_width^2)``. ``p=0.5`` (default) is the worst case (widest interval);
    supply a known/expected p for a tighter requirement. Reproduces the lecture's "n=10000 for +/-0.01 at 99%".
    """
    if not (0.0 < half_width < 1.0):
        raise ValueError("required_n_for_proportion: half_width must be in (0, 1).")
    if not (0.0 <= p <= 1.0):
        raise ValueError("required_n_for_proportion: p must be in [0, 1].")
    z = z_for_confidence(confidence)
    return math.ceil(z * z * p * (1.0 - p) / (half_width * half_width))


def proportions_significantly_different(m1: int, n1: int, m2: int, n2: int, *, confidence: float = 0.95) -> bool:
    """Two-proportion z-test: do observed proportions m1/n1 and m2/n2 differ significantly at the level?

    The zodiac-scoring check: with a huge n, tiny percentage gaps can be significant; with a small n, large
    gaps may not be. Uses the pooled-variance two-sided normal test.
    """
    if n1 <= 0 or n2 <= 0:
        raise ValueError("proportions_significantly_different: n1, n2 must be > 0.")
    p1 = m1 / n1
    p2 = m2 / n2
    pooled = (m1 + m2) / (n1 + n2)
    se = math.sqrt(pooled * (1.0 - pooled) * (1.0 / n1 + 1.0 / n2))
    if se <= 0.0:
        return p1 != p2
    z_stat = abs(p1 - p2) / se
    return z_stat > z_for_confidence(confidence)
