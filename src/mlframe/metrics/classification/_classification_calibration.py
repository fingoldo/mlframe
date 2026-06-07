"""Calibration / goodness-of-fit classification metrics (Tier 2).

Hosmer-Lemeshow test + Accuracy Ratio (CAP), carved from
``_classification_extras.py`` so the parent stays under the LOC ceiling.
Re-exported from the parent + ``metrics.core``.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import numba

from .._numba_params import NUMBA_NJIT_PARAMS


# ============================================================================
# Hosmer-Lemeshow + Accuracy Ratio (Tier 2)
# ============================================================================


@numba.njit(**NUMBA_NJIT_PARAMS)
def _hosmer_lemeshow_kernel(
    y_true_sorted: np.ndarray, y_score_sorted: np.ndarray, n_groups: int,
) -> Tuple[float, int]:
    """Returns (chi-square statistic, degrees-of-freedom-used).

    Pre-condition: ``y_score_sorted`` is ascending and ``y_true_sorted``
    is reordered to match. The kernel splits the score-sorted array into
    ``n_groups`` equal-count buckets and accumulates the per-group
    expected/observed positive counts.
    """
    n = y_true_sorted.shape[0]
    if n < n_groups:
        return np.nan, 0
    chi2 = 0.0
    counted = 0
    base = n // n_groups
    rem = n - base * n_groups  # spread remainder across first ``rem`` groups
    start = 0
    for g in range(n_groups):
        size = base + (1 if g < rem else 0)
        end = start + size
        O = 0
        E = 0.0
        for j in range(start, end):
            if y_true_sorted[j] != 0:
                O += 1
            E += y_score_sorted[j]
        # Per-Hosmer-Lemeshow denominator. When E or (N-E) collapse to 0
        # the term blows up; skip those groups (rare for n_groups<=10).
        denom_inner = E * (size - E) / size if size > 0 else 0.0
        if denom_inner > 0.0:
            diff = O - E
            chi2 += diff * diff / denom_inner
            counted += 1
        start = end
    return chi2, counted


def hosmer_lemeshow_test(
    y_true: np.ndarray, y_score: np.ndarray, n_groups: int = 10,
) -> Tuple[float, float, int]:
    """Hosmer-Lemeshow chi-square goodness-of-fit test.

    Sorts rows by predicted probability, splits into ``n_groups`` equal-
    count buckets, and accumulates the per-group chi-square statistic:

        H = sum_g (O_g - E_g)^2 / (E_g * (N_g - E_g) / N_g)

    where O_g = observed positives, E_g = sum of predicted probabilities,
    N_g = group size. Under perfect calibration H ~ chi^2(n_groups - 2);
    a large H (p < 0.05) signals miscalibration.

    Returns (H, p_value, dof). ``dof = n_groups - 2`` is canonical; we
    return the actual used dof since degenerate groups (E=0 or E=N) are
    skipped from the sum.

    Spiegelhalter Z is a stronger test (no binning), but HL is the
    conventional report in medical/credit-risk literature, where
    decile-binned diagnostics are expected.
    """
    yt = np.asarray(y_true).astype(np.int64, copy=False)
    ys = np.asarray(y_score, dtype=np.float64)
    n = yt.shape[0]
    if n == 0 or n_groups < 2:
        return np.nan, np.nan, 0
    order = np.argsort(ys, kind="quicksort")
    chi2, counted = _hosmer_lemeshow_kernel(yt[order], ys[order], int(n_groups))
    if counted < 2:
        return float(chi2), np.nan, counted
    # chi^2 sf via scipy. Pure-numpy alternative would be incomplete-gamma;
    # scipy is already a hard dep elsewhere in mlframe and the call cost
    # is negligible (<1 us) compared to the sort.
    from scipy.stats import chi2 as _chi2_dist
    dof = max(1, counted - 2)
    p_value = float(_chi2_dist.sf(chi2, dof))
    return float(chi2), p_value, dof


def accuracy_ratio(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Accuracy Ratio from the Cumulative Accuracy Profile (CAP).

    Mathematically equivalent to Gini = 2*AUC - 1; exposed under this
    name because credit-risk / banking literature reports the
    Accuracy Ratio (a.k.a. CAP-AR or Powerstat) rather than AUC.

    We compute via the explicit cumulative-positive-fraction integral
    (trapezoidal rule on score-sorted rows) rather than the AUC-derived
    closed form. Result agrees with ``2*roc_auc - 1`` to within fp
    tolerance and serves as a sanity check on the AUC computation.
    """
    yt = np.asarray(y_true).astype(np.int64, copy=False)
    ys = np.asarray(y_score, dtype=np.float64)
    n = yt.shape[0]
    if n == 0:
        return np.nan
    n_pos = int(yt.sum())
    if n_pos == 0 or n_pos == n:
        return np.nan  # CAP undefined when only one class is present
    # Sort by descending score - ties broken arbitrarily; equal-score rows
    # contribute the same amount to the area regardless of intra-tie
    # ordering by symmetry of the cumulative process.
    order = np.argsort(-ys, kind="quicksort")
    yt_s = yt[order]
    # CAP curve y-axis: cumulative true positive ratio (TP_k / n_pos)
    # x-axis: cumulative population (k/n). Trapezoidal area in (x, y) space.
    cum_tp = np.cumsum(yt_s).astype(np.float64) / n_pos
    cum_pop = (np.arange(1, n + 1, dtype=np.float64)) / n
    # Prepend (0, 0) so the trapezoid starts at the origin.
    cum_pop = np.concatenate(([0.0], cum_pop))
    cum_tp = np.concatenate(([0.0], cum_tp))
    # Manual trapezoid formula (np.trapz removed in NumPy 2; np.trapezoid
    # exists only from 2.0). Keeps the kernel numpy-version-agnostic.
    area_model = float(np.sum((cum_pop[1:] - cum_pop[:-1]) * (cum_tp[1:] + cum_tp[:-1]) * 0.5))
    # Random baseline: area = 0.5. Perfect: area = 1 - (n_pos / 2 / n).
    area_perfect = 1.0 - (n_pos / (2.0 * n))
    denom = area_perfect - 0.5
    if denom <= 0.0:
        return np.nan
    return (area_model - 0.5) / denom
