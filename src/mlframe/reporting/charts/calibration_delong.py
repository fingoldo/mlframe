"""DeLong AUC variance and logit-transformed confidence interval (re-exported from ``calibration``)."""

from __future__ import annotations

import logging
from typing import Tuple

import numpy as np

logger = logging.getLogger(__name__)


def _midrank(x: np.ndarray) -> np.ndarray:
    """Midranks of ``x`` (tied values share the mean of the ranks they would occupy). O(n log n) via one argsort.

    Midranks are the core primitive of DeLong's fast AUC-variance algorithm: AUC equals a function of the midranks of
    the pooled, the positive-only, and the negative-only score samples, so the variance follows in closed form.

    Fully vectorised: after sorting, each tie block spans ``[start, end)`` of equal values and gets the 1-indexed mean
    rank ``0.5*(start + end - 1) + 1``. ``searchsorted`` on the sorted values gives every value's left/right tie bound
    in two O(n log n) passes (no python per-element loop -- the loop form was ~93% of DeLong's wall at n=1e6).
    """
    n = x.shape[0]
    if n == 0:
        return np.empty(0, dtype=np.float64)
    order = np.argsort(x, kind="quicksort")
    sorted_x = x[order]
    # For each sorted position, the inclusive tie block is [left, right): searchsorted finds the first equal and the
    # first strictly-greater value. The mean 0-based position is 0.5*(left + right - 1); +1 makes it a 1-indexed rank.
    left = np.searchsorted(sorted_x, sorted_x, side="left")
    right = np.searchsorted(sorted_x, sorted_x, side="right")
    sorted_ranks = 0.5 * (left + right - 1) + 1.0
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = sorted_ranks
    return ranks


def delong_auc_variance(y_true: np.ndarray, y_score: np.ndarray) -> Tuple[float, float]:
    """ROC-AUC and its DeLong variance (Sun & Xu 2014 fast O(n log n) form), single binary problem.

    Returns ``(auc, var)``. ``y_true`` is binary {0,1}; ``y_score`` higher = more positive. The DeLong estimator is
    the standard closed-form AUC-variance: no bootstrap needed. ``var`` is NaN when either class is empty (AUC and
    its variance are undefined). Ties are handled via midranks, so tied scores give the correct (smaller) variance.
    """
    yt = np.asarray(y_true).ravel()
    ys = np.asarray(y_score, dtype=np.float64).ravel()
    # NaN never compares equal to itself, so a non-finite score's tie block is empty and its midrank is meaningless
    # -- and the AUC built from it is silently wrong rather than undefined. Drop those rows and say how many.
    finite = np.isfinite(ys)
    if not np.all(finite):
        logger.warning(
            "delong_auc_variance: dropping %d of %d rows with a non-finite score; a NaN has no rank, so leaving it "
            "in produces a midrank from an empty tie block and an AUC that is wrong rather than undefined.",
            int((~finite).sum()), int(ys.size),
        )
        yt, ys = yt[finite], ys[finite]
    labels = np.unique(yt)
    off_domain = [v for v in labels.tolist() if v not in (0, 1, True, False)]
    if off_domain:
        logger.warning(
            "delong_auc_variance: y_true carries %d label(s) outside {0, 1} (%s); those rows are treated as the "
            "NEGATIVE class, which is a one-vs-rest reading of a target that may not be binary.",
            len(off_domain), off_domain[:5],
        )
    pos = ys[yt == 1]
    neg = ys[yt == 0]
    m = pos.size
    n = neg.size
    if m == 0 or n == 0:
        return float("nan"), float("nan")
    # Sun-Xu midrank decomposition: structural components V10 (per positive) and V01 (per negative).
    tz = _midrank(np.concatenate([pos, neg]))
    tx = _midrank(pos)
    ty = _midrank(neg)
    # AUC = (sum of positive midranks in the pooled sample - m(m+1)/2) / (m n) -- the Mann-Whitney form via midranks.
    auc = (tz[:m].sum() - m * (m + 1.0) / 2.0) / (m * n)
    v10 = (tz[:m] - tx) / n
    v01 = 1.0 - (tz[m:] - ty) / m
    s10 = float(np.var(v10, ddof=1)) if m > 1 else 0.0
    s01 = float(np.var(v01, ddof=1)) if n > 1 else 0.0
    var = s10 / m + s01 / n
    return float(auc), float(var)


def delong_auc_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    alpha: float = 0.05,
) -> Tuple[float, float, float]:
    """ROC-AUC with a two-sided ``1-alpha`` DeLong confidence interval, on the LOGIT scale.

    Returns ``(auc, lower, upper)``. The interval is built on the logit-transformed AUC -- ``logit(auc) +- z *
    sqrt(var)/(auc*(1-auc))`` (delta-method SE of the transform) -- then mapped back through the sigmoid. This is
    the bounded-transform DeLong CI (Cortes & Mohri 2005; pROC's ``method="logit"``). The naive normal-theory
    interval ``auc +- z*sqrt(var)`` clipped to [0,1] UNDER-COVERS near AUC≈1: the symmetric bar pokes past 1 and the
    clip truncates the upper tail, so the realised coverage drops below ``1-alpha`` exactly where AUCs cluster in
    practice. The logit interval is asymmetric, stays strictly inside (0,1) by construction, and never hits the
    clip, restoring coverage near the boundary. It still narrows as ``~1/sqrt(n)``. ``(nan, nan, nan)`` when a class
    is empty; the degenerate ``auc in {0,1}`` (zero var or undefined logit) falls back to the point.
    """
    auc, var = delong_auc_variance(y_true, y_score)
    if not np.isfinite(auc) or not np.isfinite(var):
        return auc, float("nan"), float("nan")
    from scipy.stats import norm

    z = float(norm.ppf(1.0 - alpha / 2.0))
    sd = float(np.sqrt(max(var, 0.0)))
    # logit transform is undefined / has infinite SE at auc in {0,1}; with zero spread there is no interval to form.
    if not (0.0 < auc < 1.0) or sd == 0.0:
        return auc, auc, auc
    logit = float(np.log(auc / (1.0 - auc)))
    se_logit = sd / (auc * (1.0 - auc))  # delta-method: d/dx logit(x) = 1/(x(1-x))
    half = z * se_logit
    lo = 1.0 / (1.0 + np.exp(-(logit - half)))
    hi = 1.0 / (1.0 + np.exp(-(logit + half)))
    return auc, float(lo), float(hi)
