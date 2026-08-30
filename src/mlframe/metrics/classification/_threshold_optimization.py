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

import numba
import numpy as np
from numba import njit

__all__ = ["optimal_threshold", "optimal_threshold_bootstrap_ci", "THRESHOLD_METRICS"]

# ``cost`` is last so the existing integer codes are unchanged. It is the only functional that needs the caller to
# say what a mistake is worth; every other one silently assumes a false positive and a false negative cost the same,
# which at a 2.6% base rate is almost never true.
THRESHOLD_METRICS = ("f1", "balanced_accuracy", "mcc", "youden", "accuracy", "cost")
_METRIC_CODE = {m: i for i, m in enumerate(THRESHOLD_METRICS)}


@njit(fastmath=False, cache=True, nogil=True)
def _score_from_counts(tp: float, fp: float, tn: float, fn: float, code: int, fp_cost: float = 1.0, fn_cost: float = 1.0) -> float:
    """Compute one of ``THRESHOLD_METRICS`` (selected by its integer ``code``, matching ``_METRIC_CODE``) from confusion-matrix counts at a single threshold; degenerate zero-denominator cases return 0.0 rather than raising.

    Every functional here is HIGHER-is-better so one sweep can arg-max them all, which is why ``cost`` returns the
    NEGATED average cost per row rather than the cost itself."""
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
    if code == 4:  # accuracy
        total = tp + fp + tn + fn
        return (tp + tn) / total if total > 0.0 else 0.0
    # cost: negated average cost per row, so the sweep's arg-max is the cost MINIMUM
    total = tp + fp + tn + fn
    return -(fp * fp_cost + fn * fn_cost) / total if total > 0.0 else 0.0


@njit(fastmath=False, cache=True, nogil=True)
def _optimal_threshold_kernel(y_sorted: np.ndarray, s_sorted: np.ndarray, code: int, fp_cost: float, fn_cost: float):
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

    best_score = _score_from_counts(0.0, 0.0, N, P, code, fp_cost, fn_cost)  # k=0: predict all negative
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
        sc = _score_from_counts(tp, fp, tn, fn, code, fp_cost, fn_cost)
        if sc > best_score:
            best_score = sc
            best_thr = cur  # predict positive iff score >= cur
    return best_thr, best_score


def optimal_threshold(y_true: np.ndarray, y_score: np.ndarray, *, metric: str = "f1", fp_cost: float = 1.0, fn_cost: float = 1.0):
    """Find the decision threshold on ``y_score`` that maximizes ``metric`` against binary ``y_true``.

    Parameters
    ----------
    y_true : np.ndarray
        Binary labels (0/1). Any nonzero is treated as the positive class.
    y_score : np.ndarray
        Real-valued scores (higher = more positive). A point is predicted positive iff ``score >= threshold``.
    metric : {'f1', 'balanced_accuracy', 'mcc', 'youden', 'accuracy', 'cost'}
        Functional to maximize (see module docstring).
    fp_cost, fn_cost : float
        Cost of one false positive / one false negative. Read only by ``metric='cost'``, which minimizes the
        average cost per row. Their RATIO is what moves the threshold; the absolute scale only rescales the
        returned value.

    Returns
    -------
    (float, float)
        ``(best_threshold, best_metric_value)``. ``best_threshold`` is ``+inf`` when predicting all-negative wins.

    HOLDOUT CONTRACT: this FITS a parameter (the threshold) by maximizing ``metric`` on the exact
    ``(y_true, y_score)`` supplied -- an in-sample call is optimistically biased (the threshold is chosen
    to look good on these exact labels) and ``best_metric_value`` overstates deployed performance. The
    function cannot detect which rows are training vs. holdout -- the CALLER must pass a holdout/OOF split
    to get an honest threshold and an honest score, the same discipline this package's own
    ``quantile.coverage``/``quantile.pit_values`` already document for their (lower-risk, read-only)
    calibration checks.
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
    thr, score = _optimal_threshold_kernel(yt[order], ys[order], _METRIC_CODE[metric], float(fp_cost), float(fn_cost))
    return float(thr), float(score)


@njit(parallel=True, fastmath=False, cache=True, nogil=True)
def _bootstrap_threshold_kernel(y_sorted: np.ndarray, s_sorted: np.ndarray, idx: np.ndarray, code: int, fp_cost: float, fn_cost: float):
    """One sweep per bootstrap resample, all resamples in parallel; returns the per-resample winning thresholds.

    ``idx`` is ``(n_boot, n)`` of positions into the ALREADY DESCENDING-sorted arrays, drawn once by the caller so
    the RNG order is reproducible and the draw itself stays out of the GIL-bound Python loop. Sorting once and
    resampling positions keeps every resample in sorted order for free, which is the whole cost of the sweep.
    """
    n_boot = idx.shape[0]
    n = idx.shape[1]
    out = np.empty(n_boot, dtype=np.float64)
    for b in numba.prange(n_boot):
        P = 0.0
        for i in range(n):
            if y_sorted[idx[b, i]] > 0.5:
                P += 1.0
        N = float(n) - P
        best_score = _score_from_counts(0.0, 0.0, N, P, code, fp_cost, fn_cost)
        best_thr = np.inf
        tp = 0.0
        fp = 0.0
        i = 0
        while i < n:
            cur = s_sorted[idx[b, i]]
            while i < n and s_sorted[idx[b, i]] == cur:
                if y_sorted[idx[b, i]] > 0.5:
                    tp += 1.0
                else:
                    fp += 1.0
                i += 1
            sc = _score_from_counts(tp, fp, N - fp, P - tp, code, fp_cost, fn_cost)
            if sc > best_score:
                best_score = sc
                best_thr = cur
        out[b] = best_thr
    return out


def optimal_threshold_bootstrap_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    metric: str = "f1",
    fp_cost: float = 1.0,
    fn_cost: float = 1.0,
    n_boot: int = 200,
    alpha: float = 0.05,
    random_state: int = 0,
):
    """Percentile interval for the tuned threshold itself, so a reader can see how much of it is noise.

    A threshold is a FITTED parameter, and on a rare-positive target it is fitted from very few positives -- a
    point estimate alone invites treating a coin flip as a decision rule. When the interval spans most of the
    score range the honest reading is that tuning bought nothing here and 0.5 is as defensible.

    Returns ``(lo, hi)``; both are ``nan`` for an empty input. Resamples that land on the all-negative solution
    contribute ``+inf`` and are kept, since "predict nothing" being inside the interval is itself the answer.
    """
    yt = np.ascontiguousarray(y_true, dtype=np.float64)
    ys = np.ascontiguousarray(y_score, dtype=np.float64)
    if yt.shape[0] != ys.shape[0]:
        raise ValueError("optimal_threshold_bootstrap_ci: y_true and y_score length mismatch.")
    n = yt.shape[0]
    if n == 0 or n_boot <= 0:
        return float("nan"), float("nan")
    if metric not in _METRIC_CODE:
        raise ValueError(f"optimal_threshold_bootstrap_ci: metric must be one of {THRESHOLD_METRICS}, got {metric!r}.")
    order = np.argsort(-ys, kind="stable")
    # All resample indices in ONE draw: the per-resample RNG call is what makes a Python-level bootstrap loop
    # GIL-bound, and drawing up front is also what makes the run reproducible from the seed alone.
    idx = np.random.default_rng(random_state).integers(0, n, size=(int(n_boot), n)).astype(np.int64)
    thrs = _bootstrap_threshold_kernel(yt[order], ys[order], idx, _METRIC_CODE[metric], float(fp_cost), float(fn_cost))
    # ``method="nearest"`` returns an OBSERVED threshold rather than interpolating between two: interpolating
    # across the +inf ("predict all negative") solution yields inf-inf = nan and silently erases the interval,
    # and a threshold halfway between two observed cuts is not itself a cut anyway.
    lo = float(np.quantile(thrs, alpha / 2.0, method="nearest"))
    hi = float(np.quantile(thrs, 1.0 - alpha / 2.0, method="nearest"))
    return lo, hi
