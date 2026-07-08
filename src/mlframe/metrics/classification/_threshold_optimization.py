"""Optimal decision-threshold search for binary scoring functions (PZAD err_classification).

The crisp-classification lecture (Дьяконов 2020, slides 20-30) shows that when you binarize a
score with a threshold, the threshold that MAXIMIZES each quality functional is different, and
moves with the class balance: F1's optimum drifts with prevalence, balanced-accuracy's optimum
sits near the class-separating point, MCC/kappa peak elsewhere again. mlframe already has every
functional (`_classification_extras`) and an F1-only sweep buried in the reporting layer, but no
general "give me the threshold that maximizes THIS functional on THESE scores" primitive.

``optimal_threshold`` does one descending sort and sweeps every distinct cut, maintaining the
confusion counts (tp, fp, tn, fn) incrementally (the ROC-sweep trick, O(n log n)), evaluating the
requested functional at each cut and returning the arg-max. Supported functionals:
``f1``, ``balanced_accuracy``, ``mcc``, ``youden`` (Youden's J = TPR + TNR - 1), ``accuracy``.
"""

from __future__ import annotations

import numpy as np
from numba import njit

__all__ = ["optimal_threshold", "THRESHOLD_METRICS"]

THRESHOLD_METRICS = ("f1", "balanced_accuracy", "mcc", "youden", "accuracy")
_METRIC_CODE = {m: i for i, m in enumerate(THRESHOLD_METRICS)}


@njit(fastmath=False, cache=True, nogil=True)
def _score_from_counts(tp: float, fp: float, tn: float, fn: float, code: int) -> float:
    """Compute one of ``THRESHOLD_METRICS`` (selected by its integer ``code``, matching ``_METRIC_CODE``) from confusion-matrix counts at a single threshold; degenerate zero-denominator cases return 0.0 rather than raising."""
    if code == 0:  # f1
        denom = 2.0 * tp + fp + fn
        return 2.0 * tp / denom if denom > 0.0 else 0.0
    if code == 1:  # balanced_accuracy = (TPR + TNR) / 2
        tpr = tp / (tp + fn) if (tp + fn) > 0.0 else 0.0
        tnr = tn / (tn + fp) if (tn + fp) > 0.0 else 0.0
        return 0.5 * (tpr + tnr)
    if code == 2:  # mcc
        num = tp * tn - fp * fn
        den = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
        return num / np.sqrt(den) if den > 0.0 else 0.0
    if code == 3:  # youden J = TPR + TNR - 1
        tpr = tp / (tp + fn) if (tp + fn) > 0.0 else 0.0
        tnr = tn / (tn + fp) if (tn + fp) > 0.0 else 0.0
        return tpr + tnr - 1.0
    # accuracy
    total = tp + fp + tn + fn
    return (tp + tn) / total if total > 0.0 else 0.0


@njit(fastmath=False, cache=True, nogil=True)
def _optimal_threshold_kernel(y_sorted: np.ndarray, s_sorted: np.ndarray, code: int):
    """y_sorted/s_sorted are ordered by DESCENDING score. Sweep every distinct cut; return (best_thr, best_score).

    At a cut after the first k points, predictions for those k (highest-score) points are positive.
    ``best_thr`` is the score value such that ``score >= best_thr`` reproduces the winning prediction;
    +inf means "predict all negative" won (empty positive set)."""
    n = y_sorted.shape[0]
    P = 0.0
    for i in range(n):
        if y_sorted[i] > 0.5:
            P += 1.0
    N = float(n) - P

    best_score = _score_from_counts(0.0, 0.0, N, P, code)  # k=0: predict all negative
    best_thr = np.inf

    tp = 0.0
    fp = 0.0
    i = 0
    while i < n:
        # advance through all points sharing this score (a threshold can only cut between distinct scores)
        cur = s_sorted[i]
        while i < n and s_sorted[i] == cur:
            if y_sorted[i] > 0.5:
                tp += 1.0
            else:
                fp += 1.0
            i += 1
        tn = N - fp
        fn = P - tp
        sc = _score_from_counts(tp, fp, tn, fn, code)
        if sc > best_score:
            best_score = sc
            best_thr = cur  # predict positive iff score >= cur
    return best_thr, best_score


def optimal_threshold(y_true: np.ndarray, y_score: np.ndarray, *, metric: str = "f1"):
    """Find the decision threshold on ``y_score`` that maximizes ``metric`` against binary ``y_true``.

    Parameters
    ----------
    y_true : np.ndarray
        Binary labels (0/1). Any nonzero is treated as the positive class.
    y_score : np.ndarray
        Real-valued scores (higher = more positive). A point is predicted positive iff ``score >= threshold``.
    metric : {'f1', 'balanced_accuracy', 'mcc', 'youden', 'accuracy'}
        Functional to maximize (see module docstring).

    Returns
    -------
    (float, float)
        ``(best_threshold, best_metric_value)``. ``best_threshold`` is ``+inf`` when predicting all-negative wins.
    """
    if metric not in _METRIC_CODE:
        raise ValueError(f"optimal_threshold: metric must be one of {THRESHOLD_METRICS}, got {metric!r}.")
    yt = np.ascontiguousarray(y_true, dtype=np.float64)
    ys = np.ascontiguousarray(y_score, dtype=np.float64)
    if yt.shape[0] != ys.shape[0]:
        raise ValueError("optimal_threshold: y_true and y_score length mismatch.")
    if yt.shape[0] == 0:
        return np.inf, np.nan
    order = np.argsort(-ys, kind="stable")
    thr, score = _optimal_threshold_kernel(yt[order], ys[order], _METRIC_CODE[metric])
    return float(thr), float(score)
