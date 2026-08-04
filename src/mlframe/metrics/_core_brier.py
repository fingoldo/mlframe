"""Brier score kernels + scorers, carved out of ``_core_auc_brier.py`` to keep it under the 1k LOC
ceiling. Public symbols are re-exported from ``_core_auc_brier.py`` (and transitively from
``mlframe.metrics``) so existing importers are unaffected.
"""

from __future__ import annotations

from typing import Callable, cast

import numba
import numpy as np

from ._numba_params import NUMBA_NJIT_PARAMS, _PARALLEL_REDUCTION_THRESHOLD, _check_equal_length


@numba.njit(**NUMBA_NJIT_PARAMS)
def _fast_brier_score_loss_seq(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Sequential Brier score (mean squared error between labels and predicted probabilities); the small-N counterpart of ``_fast_brier_score_loss_par``."""
    return float(np.mean((y_true - y_prob) ** 2))


@numba.njit(**NUMBA_NJIT_PARAMS, parallel=True)
def _fast_brier_score_loss_par(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Parallel variant. ~7.7x faster than seq at N=10M (verified
    on 8-thread numba runtime). Loses to seq below N~50k due to
    thread-spawn overhead -- the public ``fast_brier_score_loss``
    wrapper auto-dispatches based on N."""
    n = len(y_true)
    s = 0.0
    for i in numba.prange(n):
        d = y_true[i] - y_prob[i]
        s += d * d
    return s / n


@numba.njit(**NUMBA_NJIT_PARAMS)
def _fast_brier_checked_seq(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Fused validation + Brier in ONE pass. Returns nan if any ``y_prob`` is
    non-finite or outside [0, 1] (``not (0.0 <= p <= 1.0)`` catches nan/inf AND
    out-of-range in a single comparison), else the mean squared error. Replaces the
    3 separate numpy reduction passes (isfinite, min, max) the Python wrapper did
    before the kernel -- those dominated the cost (~80% at n=1M). Brier value is
    bit-identical to ``_fast_brier_score_loss_seq`` on valid input."""
    n = len(y_true)
    s = 0.0
    bad = 0
    for i in range(n):
        pi = y_prob[i]
        if not (0.0 <= pi <= 1.0):
            bad += 1
        d = y_true[i] - pi
        s += d * d
    if bad > 0:
        return np.nan
    return s / n


@numba.njit(**NUMBA_NJIT_PARAMS, parallel=True)
def _fast_brier_checked_par(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Parallel fused validation + Brier. ``bad`` and ``s`` are both prange
    reductions. See ``_fast_brier_checked_seq``."""
    n = len(y_true)
    s = 0.0
    bad = 0
    for i in numba.prange(n):
        pi = y_prob[i]
        if not (0.0 <= pi <= 1.0):
            bad += 1
        d = y_true[i] - pi
        s += d * d
    if bad > 0:
        return np.nan
    return s / n


# Crossover threshold for parallel kernels. See ``_numba_params.py`` (SSOT).
# Sum-reduction kernels (brier, log loss, prf1 counts) parallel-win from
# N~50-100k upwards. Multilabel row-loop kernels (subset accuracy, jaccard)
# win from N~10-50k. Conservative thresholds chosen to avoid the lose-band
# at low N.


def fast_brier_score_loss(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Brier score (mean squared error of probabilities), auto seq/par.

    Sequential numba kernel below ~100k rows (cold-start cost
    of the parallel runtime exceeds the per-element gain). Parallel
    kernel above the threshold -- 7.7x faster at N=10M on an 8-thread
    runtime. Tunable via ``_PARALLEL_REDUCTION_THRESHOLD``.

    Returns ``np.nan`` on out-of-[0,1] or NaN probabilities, mirroring
    ``fast_log_loss_binary``: the kernels would otherwise square whatever
    garbage was passed and report a plausible-looking but meaningless score.
    """
    _check_equal_length(y_true, y_prob)
    # Fused validation + Brier: the finite/[0,1] guard now happens INSIDE the njit
    # sweep (single pass) instead of 3 separate numpy reductions (isfinite/min/max)
    # that dominated wall time (~80% at n=1M). nan on invalid probs, unchanged.
    if len(y_true) == 0:
        return float(np.nan)
    if len(y_true) >= _PARALLEL_REDUCTION_THRESHOLD:
        return float(_fast_brier_checked_par(y_true, y_prob))
    return float(_fast_brier_checked_seq(y_true, y_prob))


# Backward-compat alias - older code and tests import `brier_score_loss` from this module.
# Keep the name visible but route it to the renamed fast_brier_score_loss so the intent is clear.
brier_score_loss = fast_brier_score_loss


def brier_and_precision_score(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    precision_threshold: float = 0.5,
    brier_threshold: float = 0.25,
) -> float:
    """precision_score - fast_brier_score_loss when both thresholds pass, else 0.

    Brier must be <= brier_threshold and precision must be >= precision_threshold
    (at a 0.5 decision boundary) for a non-zero result. Useful as a conservative
    optimisation objective that rewards only models that are simultaneously
    calibrated AND precise at the top.
    """
    from ._core_precision_mape import fast_precision

    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba, dtype=float)

    brier = fast_brier_score_loss(y_true=y_true, y_prob=y_proba)
    # fast_brier_score_loss returns NaN on out-of-[0,1] / NaN probabilities. NaN > threshold is False, so without
    # this guard an invalid-probability model would slip through the gate and make this model-selection scorer
    # return NaN (or precision - NaN), silently changing which model wins. Treat a non-finite Brier as a hard fail.
    if not np.isfinite(brier) or brier > brier_threshold:
        return 0.0
    y_pred = (y_proba > 0.5).astype(int)
    # This is a strictly binary scorer. A multiclass y_true (labels outside {0, 1}) is a real input-contract
    # violation that must surface, not be silently ignored -- fast_precision(nclasses=2) would drop the extra
    # classes and report a corrupted precision, so validate before the numba kernel (mirrors what sklearn's
    # precision_score raised on this input). Returning 0.0 here would make the model-selection scorer report
    # the worst value and silently change which model wins.
    labels = np.unique(y_true)
    if labels.size and (labels.min() < 0 or labels.max() > 1):
        raise ValueError(f"brier_and_precision_score requires binary y_true in {{0, 1}}; got labels {labels.tolist()}")
    # fast_precision returns the positive-class (class-1) precision, matching binary precision_score
    # with zero_division=0 (empty positive predictions -> 0.0), and is the project's numba equivalent.
    precision = fast_precision(y_true.astype(np.int64), y_pred.astype(np.int64), nclasses=2, zero_division=0)
    if precision < precision_threshold:
        return 0.0
    return float(precision - brier)


def make_brier_precision_scorer(precision_threshold: float = 0.5, brier_threshold: float = 0.25) -> Callable:
    """Return an sklearn scorer wrapping brier_and_precision_score (needs_proba=True)."""
    from sklearn.metrics import make_scorer

    # New sklearn (>=1.4) replaces `needs_proba` with `response_method`; fall back
    # to the legacy kwarg for older versions.
    try:
        return cast(Callable, make_scorer(
            brier_and_precision_score,
            response_method="predict_proba",
            greater_is_better=True,
            precision_threshold=precision_threshold,
            brier_threshold=brier_threshold,
        ))
    except TypeError:
        return cast(Callable, make_scorer(
            brier_and_precision_score,
            needs_proba=True,
            greater_is_better=True,
            precision_threshold=precision_threshold,
            brier_threshold=brier_threshold,
        ))
