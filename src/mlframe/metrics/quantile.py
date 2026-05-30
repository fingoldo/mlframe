"""Quantile-regression metrics with numba-JITed inner kernels.

Public API (sklearn-compatible signatures):
- ``pinball_loss(y, q, alpha)`` - per-(y, q-vector, alpha) score
- ``pinball_loss_per_alpha(y, preds_NK, alphas)`` - dict {alpha: loss}
- ``coverage(y, q_lo, q_hi)`` - empirical coverage in [0, 1]
- ``mean_interval_width(q_lo, q_hi)`` - sharpness
- ``winkler_score(y, q_lo, q_hi, alpha_target)`` - combined score
- ``pit_values(y, preds_NK, alphas)`` - probability-integral-transform
  values per row (uniform if calibrated)

Follows the existing mlframe numba pattern
(``NUMBA_NJIT_PARAMS = dict(fastmath=False, cache=True, nogil=True)``,
parallel variant for N > 100k where it pays).

Pinball formula (sklearn ``mean_pinball_loss``):
    L(y, q, alpha) = mean(max(alpha * (y - q), (alpha - 1) * (y - q)))

Coverage: ``mean((y >= q_lo) & (y <= q_hi))``.

Winkler interval score (Winkler 1972; Gneiting & Raftery 2007):
    S = (q_hi - q_lo)
        + (2 / alpha_target) * (q_lo - y) * (y < q_lo)
        + (2 / alpha_target) * (y - q_hi) * (y > q_hi)
where alpha_target is the nominal MISCOVERAGE (e.g. 0.2 for an 80%
interval, i.e. (alpha_high - alpha_low) when (alpha_low, alpha_high)
are quantile-LEVELS).
"""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np

try:
    import numba
    _NUMBA_AVAILABLE = True
except ImportError:
    _NUMBA_AVAILABLE = False
    numba = None  # type: ignore

# Mirror metrics.py NUMBA_NJIT_PARAMS pattern.
_NJIT_KW = dict(fastmath=False, cache=True, nogil=True)


# ----------------------------------------------------------------------------
# Numba kernels (private)
# ----------------------------------------------------------------------------


if _NUMBA_AVAILABLE:

    @numba.njit(**_NJIT_KW)
    def _fast_pinball(y: np.ndarray, q: np.ndarray, alpha: float) -> float:
        """Mean pinball loss for one alpha. y, q: (N,) float64."""
        n = y.shape[0]
        if n == 0:
            return 0.0
        s = 0.0
        for i in range(n):
            e = y[i] - q[i]
            if e > 0:
                s += alpha * e
            else:
                s += (alpha - 1.0) * e
        return s / n

    @numba.njit(**_NJIT_KW)
    def _fast_coverage(y: np.ndarray, q_lo: np.ndarray, q_hi: np.ndarray) -> float:
        n = y.shape[0]
        if n == 0:
            return 0.0
        c = 0
        for i in range(n):
            if q_lo[i] <= y[i] <= q_hi[i]:
                c += 1
        return c / n

    @numba.njit(**_NJIT_KW)
    def _fast_winkler(
        y: np.ndarray, q_lo: np.ndarray, q_hi: np.ndarray, alpha_miscov: float,
    ) -> float:
        """Mean Winkler interval score. ``alpha_miscov`` is the NOMINAL
        miscoverage (e.g. 0.2 for an 80% interval). Lower is better."""
        n = y.shape[0]
        if n == 0:
            return 0.0
        s = 0.0
        two_over_a = 2.0 / max(alpha_miscov, 1e-12)
        for i in range(n):
            width = q_hi[i] - q_lo[i]
            penalty = 0.0
            if y[i] < q_lo[i]:
                penalty = two_over_a * (q_lo[i] - y[i])
            elif y[i] > q_hi[i]:
                penalty = two_over_a * (y[i] - q_hi[i])
            s += width + penalty
        return s / n

else:
    # Numpy fallbacks (slow path; identical contract).
    def _fast_pinball(y, q, alpha):
        e = y - q
        return float(np.mean(np.maximum(alpha * e, (alpha - 1.0) * e)))

    def _fast_coverage(y, q_lo, q_hi):
        return float(np.mean((y >= q_lo) & (y <= q_hi)))

    def _fast_winkler(y, q_lo, q_hi, alpha_miscov):
        width = q_hi - q_lo
        below = (y < q_lo) * (2.0 / max(alpha_miscov, 1e-12)) * (q_lo - y)
        above = (y > q_hi) * (2.0 / max(alpha_miscov, 1e-12)) * (y - q_hi)
        return float(np.mean(width + below + above))


# ----------------------------------------------------------------------------
# Public wrappers
# ----------------------------------------------------------------------------


def pinball_loss(y_true, y_pred, alpha: float) -> float:
    """Mean pinball loss at level ``alpha``.

    Equivalent to ``sklearn.metrics.mean_pinball_loss(y_true, y_pred, alpha=alpha)``.
    """
    y = np.ascontiguousarray(np.asarray(y_true, dtype=np.float64).ravel())
    q = np.ascontiguousarray(np.asarray(y_pred, dtype=np.float64).ravel())
    if y.shape != q.shape:
        raise ValueError(
            f"pinball_loss: y_true.shape={y.shape} != y_pred.shape={q.shape}"
        )
    return float(_fast_pinball(y, q, float(alpha)))


def pinball_loss_per_alpha(
    y_true, preds_NK, alphas: Sequence[float],
) -> Dict[float, float]:
    """Per-alpha pinball loss; key = alpha (float), value = mean loss."""
    y = np.ascontiguousarray(np.asarray(y_true, dtype=np.float64).ravel())
    P = np.ascontiguousarray(np.asarray(preds_NK, dtype=np.float64))
    if P.ndim != 2:
        raise ValueError(f"preds_NK must be 2-D; got shape {P.shape}")
    if P.shape[0] != y.shape[0]:
        raise ValueError(
            f"preds_NK.shape[0]={P.shape[0]} != len(y_true)={y.shape[0]}"
        )
    if P.shape[1] != len(alphas):
        raise ValueError(
            f"preds_NK.shape[1]={P.shape[1]} != len(alphas)={len(alphas)}"
        )
    return {
        float(a): float(_fast_pinball(y, np.ascontiguousarray(P[:, j]), float(a)))
        for j, a in enumerate(alphas)
    }


def coverage(y_true, q_lo, q_hi) -> float:
    """Empirical coverage: fraction of y in [q_lo, q_hi]. In [0, 1].

    iter610: dropped the unconditional ``dtype=np.float64`` cast on
    each input (same pattern as iter595-608). Kernel ``_fast_coverage``
    is two comparisons + a counter per element -- inside the iter597
    safe band. Bench n=100k: (int, f64, f64) 2.17x, (f64, f64, f64)
    1.20x. Bit-equivalent. bench-attempt-rejected for
    ``pinball_loss`` (4 ops/element with branched accumulator -- 0.92x
    regression on float64+float64 @100k); pinball keeps its cast."""
    y = np.ascontiguousarray(y_true).ravel()
    lo = np.ascontiguousarray(q_lo).ravel()
    hi = np.ascontiguousarray(q_hi).ravel()
    if not (y.shape == lo.shape == hi.shape):
        raise ValueError(
            f"coverage: shape mismatch y={y.shape}, q_lo={lo.shape}, q_hi={hi.shape}"
        )
    return float(_fast_coverage(y, lo, hi))


def mean_interval_width(q_lo, q_hi) -> float:
    """Mean width ``q_hi - q_lo`` (sharpness; lower is sharper)."""
    lo = np.asarray(q_lo, dtype=np.float64).ravel()
    hi = np.asarray(q_hi, dtype=np.float64).ravel()
    if lo.shape != hi.shape:
        raise ValueError(
            f"mean_interval_width: q_lo.shape={lo.shape} != q_hi.shape={hi.shape}"
        )
    return float(np.mean(hi - lo))


def winkler_score(y_true, q_lo, q_hi, alpha_miscov: float) -> float:
    """Mean Winkler interval score at nominal miscoverage ``alpha_miscov``.

    For an 80% nominal interval (alphas = (0.1, 0.9)),
    ``alpha_miscov = 0.2``. Lower is better.

    Formula (Winkler 1972):
        S = (q_hi - q_lo)
            + (2/alpha_miscov) * (q_lo - y) * I(y < q_lo)
            + (2/alpha_miscov) * (y - q_hi) * I(y > q_hi)
    """
    y = np.ascontiguousarray(np.asarray(y_true, dtype=np.float64).ravel())
    lo = np.ascontiguousarray(np.asarray(q_lo, dtype=np.float64).ravel())
    hi = np.ascontiguousarray(np.asarray(q_hi, dtype=np.float64).ravel())
    if not (y.shape == lo.shape == hi.shape):
        raise ValueError(
            f"winkler_score: shape mismatch y={y.shape}, q_lo={lo.shape}, q_hi={hi.shape}"
        )
    if not (0.0 < alpha_miscov < 1.0):
        raise ValueError(f"alpha_miscov must be in (0, 1); got {alpha_miscov}")
    return float(_fast_winkler(y, lo, hi, float(alpha_miscov)))


def pit_values(y_true, preds_NK, alphas: Sequence[float]) -> np.ndarray:
    """Probability-integral-transform values per row.

    For each row i, find the alpha-grid position where ``y_true[i]``
    crosses the predicted quantile curve (linearly interpolated on the
    sorted alphas). Returns an (N,) array in [0, 1] (clipped).

    A well-calibrated quantile predictor produces a UNIFORM PIT
    distribution; deviations indicate miscalibration:
    - U-shape -> intervals too narrow (over-confident)
    - inverted-U -> intervals too wide
    - skew -> systematic under/over prediction
    """
    y = np.asarray(y_true, dtype=np.float64).ravel()
    P = np.asarray(preds_NK, dtype=np.float64)
    a_arr = np.asarray(alphas, dtype=np.float64)
    if P.ndim != 2 or P.shape[0] != y.shape[0] or P.shape[1] != a_arr.shape[0]:
        raise ValueError(
            f"pit_values: shape mismatch y={y.shape}, preds={P.shape}, "
            f"alphas={a_arr.shape}"
        )
    out = np.empty(y.shape[0], dtype=np.float64)
    for i in range(y.shape[0]):
        row = P[i]
        # Sort the row so np.interp's monotonicity contract holds even
        # if a tiny crossing slipped through the post-fix.
        order = np.argsort(row)
        sorted_q = row[order]
        sorted_a = a_arr[order]
        if y[i] <= sorted_q[0]:
            out[i] = float(sorted_a[0])
        elif y[i] >= sorted_q[-1]:
            out[i] = float(sorted_a[-1])
        else:
            out[i] = float(np.interp(y[i], sorted_q, sorted_a))
    return np.clip(out, 0.0, 1.0)


def quantile_summary(
    y_true, preds_NK, alphas: Sequence[float],
    coverage_pairs: Sequence[Sequence[float]] = ((0.1, 0.9),),
) -> Dict[str, Any]:
    """Convenience: returns dict with pinball-per-alpha + per-pair
    {coverage, mean_width, winkler}.

    Used by ``report_quantile_model_perf`` and the dispatcher.
    """
    pinball = pinball_loss_per_alpha(y_true, preds_NK, alphas)
    out: Dict[str, Any] = {"pinball_per_alpha": pinball}
    a_arr = list(alphas)
    for lo_a, hi_a in coverage_pairs:
        if lo_a not in a_arr or hi_a not in a_arr:
            continue
        col_lo = a_arr.index(lo_a)
        col_hi = a_arr.index(hi_a)
        q_lo = preds_NK[:, col_lo]
        q_hi = preds_NK[:, col_hi]
        # Nominal miscoverage = 1 - nominal-coverage = 1 - (hi_a - lo_a).
        # E.g. (0.1, 0.9) -> 80% nominal coverage -> miscov = 0.2.
        nominal_miscov_winkler = max(1e-12, 1.0 - (hi_a - lo_a))
        out[f"coverage_{lo_a}_{hi_a}"] = coverage(y_true, q_lo, q_hi)
        out[f"mean_width_{lo_a}_{hi_a}"] = mean_interval_width(q_lo, q_hi)
        out[f"winkler_{lo_a}_{hi_a}"] = winkler_score(
            y_true, q_lo, q_hi, nominal_miscov_winkler,
        )
    return out


def crps_from_quantiles(
    y_true: np.ndarray,
    preds_NK: np.ndarray,
    alphas: Sequence[float],
) -> float:
    """CRPS estimated from a discrete set of predicted quantiles.

    Theoretical identity (Gneiting & Raftery 2007, eq. 4.13):
        CRPS(F, y) = 2 * integral_0^1 PinballLoss(alpha, y, q(alpha)) d alpha
    where q(alpha) is the predicted alpha-quantile and PinballLoss is the
    per-row quantile loss (NOT averaged across rows).

    Numerical evaluation: trapezoidal rule on the supplied ``alphas``.
    Accuracy improves with denser alpha grids; common deployments use
    9 quantiles (0.1, 0.2, ..., 0.9), 19 (0.05 step), or 99 (0.01 step).

    Lower is better. Reduces to the Brier score for a Bernoulli target
    with a single-quantile prediction; for regression with quantile
    predictions it's the proper-scoring-rule generalisation of pinball.

    Parameters
    ----------
    y_true : (N,) array of observed targets.
    preds_NK : (N, K) array of predicted quantiles, columns aligned with ``alphas``.
    alphas : K-vector of quantile levels in (0, 1), sorted ascending.

    Returns
    -------
    Scalar CRPS averaged across rows.
    """
    y = np.asarray(y_true, dtype=np.float64)
    p = np.asarray(preds_NK, dtype=np.float64)
    a = np.asarray(alphas, dtype=np.float64)
    if y.shape[0] == 0 or p.size == 0:
        return float("nan")
    if p.ndim != 2:
        raise ValueError(f"preds_NK must be 2-D (N, K), got shape {p.shape}")
    if p.shape[0] != y.shape[0]:
        raise ValueError(
            f"row count mismatch: y_true={y.shape[0]}, preds_NK={p.shape[0]}",
        )
    if a.shape[0] != p.shape[1]:
        raise ValueError(
            f"alpha-count mismatch: K={p.shape[1]} cols, len(alphas)={a.shape[0]}",
        )
    if not np.all(np.diff(a) > 0):
        raise ValueError("alphas must be strictly increasing")
    # Per-alpha mean pinball loss vector (length K).
    per_alpha = np.empty(a.shape[0], dtype=np.float64)
    for k in range(a.shape[0]):
        per_alpha[k] = pinball_loss(y, p[:, k], float(a[k]))
    # Trapezoidal integration over alpha in [a[0], a[-1]], then scale by 2.
    # Manual formula (np.trapz removed in NumPy 2; np.trapezoid only exists
    # from 2.0) keeps the kernel numpy-version-agnostic.
    integral = float(np.sum((a[1:] - a[:-1]) * (per_alpha[1:] + per_alpha[:-1]) * 0.5))
    return 2.0 * integral


__all__ = [
    "pinball_loss",
    "pinball_loss_per_alpha",
    "coverage",
    "mean_interval_width",
    "winkler_score",
    "pit_values",
    "quantile_summary",
    "crps_from_quantiles",
]


# Late-bound type alias to avoid touching the typing import line.
from typing import Any  # noqa: E402  (kept at bottom for reduced churn)
