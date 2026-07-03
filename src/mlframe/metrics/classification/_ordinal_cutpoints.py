"""Optimal ordinal cutpoints: bin a continuous prediction into ordinal classes to maximize a target metric.

From PZAD «Минимизация ошибок» (Дьяконов 2020, slide 3, the CrowdFlower «выбор порогов» case): the modeling
idiom for ordinal targets scored by Quadratic Weighted Kappa is to fit a plain regression on the ordinal grade,
then DIRECTLY optimize the cut thresholds ``c_1 < ... < c_{l-1}`` that digitize the continuous prediction into
grades ``0..l-1`` so the target metric is maximized. This is the «решающее правило» / direct-functional-tuning
pattern (F(B·C_c) -> min): a base model plus a parametric post-transform tuned against the real metric.

``optimal_ordinal_cutpoints`` searches the thresholds (Powell / Nelder-Mead from prevalence-quantile inits, with
a coordinate-scan fallback if SciPy is unavailable) to maximize QWK (or accuracy). ``apply_cutpoints`` digitizes.
"""

from __future__ import annotations

import logging

import numpy as np

from ._weighted_kappa import quadratic_weighted_kappa

logger = logging.getLogger(__name__)

__all__ = ["optimal_ordinal_cutpoints", "apply_cutpoints", "CUTPOINT_METRICS"]

CUTPOINT_METRICS = ("qwk", "accuracy")


def apply_cutpoints(y_pred: np.ndarray, thresholds: np.ndarray, n_classes: int) -> np.ndarray:
    """Digitize continuous ``y_pred`` into integer grades ``0..n_classes-1`` at the sorted ``thresholds`` (length n_classes-1)."""
    yp = np.ascontiguousarray(y_pred, dtype=np.float64)
    thr = np.sort(np.ascontiguousarray(thresholds, dtype=np.float64))
    return np.clip(np.digitize(yp, thr), 0, n_classes - 1).astype(np.int64)


def _score(y_true, labels, n_classes, metric):
    if metric == "qwk":
        return quadratic_weighted_kappa(y_true, labels, n_classes=n_classes)
    return float(np.mean(labels == y_true))


def _coordinate_scan(y_true, y_pred, thr, n_classes, metric):
    """SciPy-free fallback: sweep each threshold over prediction midpoints, keep improvements, a few passes."""
    cand = np.unique(y_pred)
    mids = (cand[:-1] + cand[1:]) / 2.0 if cand.shape[0] > 1 else cand
    best = _score(y_true, apply_cutpoints(y_pred, thr, n_classes), n_classes, metric)
    for _ in range(4):
        improved = False
        for k in range(len(thr)):
            for m in mids:
                trial = thr.copy()
                trial[k] = m
                sc = _score(y_true, apply_cutpoints(y_pred, trial, n_classes), n_classes, metric)
                if sc > best:
                    best, thr, improved = sc, np.sort(trial), True
        if not improved:
            break
    return thr, best


def optimal_ordinal_cutpoints(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    n_classes: int,
    metric: str = "qwk",
    init: np.ndarray | None = None,
):
    """Find the ``n_classes-1`` cut thresholds on ``y_pred`` that maximize ``metric`` against ordinal ``y_true``.

    Parameters
    ----------
    y_true : np.ndarray
        Integer ordinal grades ``0..n_classes-1``.
    y_pred : np.ndarray
        Continuous predictions (e.g. a regression fit on the grade).
    n_classes : int
        Number of ordinal grades.
    metric : {'qwk', 'accuracy'}
        Objective to maximize (QWK is the standard ordinal metric).
    init : np.ndarray, optional
        Initial thresholds (length ``n_classes-1``). Defaults to the prediction quantiles at the true-label prevalence.

    Returns
    -------
    (np.ndarray, float)
        ``(thresholds, best_score)``. Digitize new predictions with :func:`apply_cutpoints`.
    """
    if metric not in CUTPOINT_METRICS:
        raise ValueError(f"optimal_ordinal_cutpoints: metric must be one of {CUTPOINT_METRICS}, got {metric!r}.")
    if n_classes < 2:
        raise ValueError("optimal_ordinal_cutpoints: n_classes must be >= 2.")
    yt = np.ascontiguousarray(y_true).astype(np.int64)
    yp = np.ascontiguousarray(y_pred, dtype=np.float64)
    if yt.shape[0] != yp.shape[0]:
        raise ValueError("optimal_ordinal_cutpoints: y_true and y_pred length mismatch.")

    if init is not None:
        thr = np.sort(np.ascontiguousarray(init, dtype=np.float64))
    else:
        # cumulative label prevalence -> matching prediction quantiles as a strong warm start
        counts = np.bincount(yt, minlength=n_classes).astype(np.float64)
        cum = np.cumsum(counts)[:-1] / max(yt.shape[0], 1)
        thr = np.quantile(yp, cum)
    thr = np.asarray(thr, dtype=np.float64)

    try:
        from scipy.optimize import minimize

        def neg(t):
            return -_score(yt, apply_cutpoints(yp, t, n_classes), n_classes, metric)

        res = minimize(neg, thr, method="Powell")
        cand_thr = np.sort(np.asarray(res.x, dtype=np.float64).ravel())
        cand_score = -float(res.fun)
        # guard: Powell can wander; keep it only if it beats the warm start
        base_score = _score(yt, apply_cutpoints(yp, thr, n_classes), n_classes, metric)
        if cand_score >= base_score:
            return cand_thr, cand_score
        return thr, base_score
    except Exception:
        # scipy.optimize failed -> fall back to the coordinate scan. Log at DEBUG so a genuine bug (vs a
        # legitimately hard objective) is diagnosable rather than a silent optimizer swap.
        logger.debug("ordinal cutpoints: scipy optimize failed; falling back to coordinate scan", exc_info=True)
        return _coordinate_scan(yt, yp, thr, n_classes, metric)
