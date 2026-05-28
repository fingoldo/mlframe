"""Binary log-loss + probability-separation score for ``mlframe.metrics.core``.

Split out from ``core.py`` to keep that file below the 1k-line monolith
threshold. Behaviour preserved bit-for-bit; every moved symbol is
re-exported from ``core`` so existing
``from mlframe.metrics.core import fast_log_loss`` (and the other moved
names) imports continue to work.

What lives here:
  - Binary log loss kernels (``_fast_log_loss_binary_seq`` / ``_par``) plus
    the auto-dispatching wrapper ``fast_log_loss_binary``.
  - sklearn-compat front door ``fast_log_loss`` (handles pandas/polars
    Series, dtype coercion, eps='auto' semantics).
  - Probability-separation kernels (``_probability_separation_score_seq`` /
    ``_par``) plus the public ``probability_separation_score`` dispatcher.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl
import numba

from ._numba_params import (
    NUMBA_NJIT_PARAMS,
    _PARALLEL_REDUCTION_THRESHOLD,
    _PARALLEL_MULTILABEL_THRESHOLD,
)


@numba.njit(**NUMBA_NJIT_PARAMS)
def _fast_log_loss_binary_seq(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-15) -> float:
    """Sequential numba binary log loss. See ``fast_log_loss_binary``
    public wrapper for auto seq/par dispatch.

    bench-attempt-rejected (2026-05-28, c0070): fusing the out-of-range scan
    into the accumulation loop (as the ``_par`` variant below does to trim a
    prange launch) does NOT reliably help the SEQ variant -- measured 1.31x at
    n=5k but 0.88x (REGRESSION) at n=50k, 1.03x at n=200k. The seq pass-1 is a
    branch-only bounds scan that leaves ``y_pred`` cache-warm for pass-2, so
    there is no launch overhead to amortise and the fused ``bad``-counter form
    just adds a branch. The par fusion win does not carry over; keep two passes."""
    n = len(y_true)
    if n == 0:
        return 0.0

    loss_sum = 0.0
    has_class_0 = False
    has_class_1 = False

    # Explicit out-of-range probability check: return NaN rather than silently
    # clipping whatever garbage the caller passed (previously a negative / >1
    # prob was just clipped to eps / 1-eps and the result looked valid).
    for i in range(n):
        if y_pred[i] < 0.0 or y_pred[i] > 1.0:
            return np.nan

    for i in range(n):
        p = y_pred[i]
        # Clip to prevent log(0)
        p = max(eps, min(1 - eps, p))
        if y_true[i] == 1:
            loss_sum -= np.log(p)
            has_class_1 = True
        else:
            loss_sum -= np.log(1 - p)
            has_class_0 = True

    # Return nan if only one class present (mimics sklearn behavior)
    if not (has_class_0 and has_class_1):
        return np.nan

    return loss_sum / n


@numba.njit(**NUMBA_NJIT_PARAMS, parallel=True)
def _fast_log_loss_binary_par(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-15) -> float:
    """Parallel binary log loss. ~4.8x faster than seq at N=10M.
    Auto-selected by ``fast_log_loss_binary`` above N=100k.

    Single-pass: out-of-range detection + clip + log + sum-reduce all
    fold into one ``prange`` walk over y_pred. Bad rows skip the loss
    accumulation; a non-zero ``bad`` counter post-walk yields ``np.nan``.
    Previously two separate ``prange`` walks (bounds scan, then loss scan);
    the fused form trims one full N-read (~6-9% kernel speedup at N=1M-5M;
    bench: ``profiling/bench_fast_log_loss_binary_fused.py``).
    """
    n = len(y_true)
    if n == 0:
        return 0.0

    bad = 0
    loss_sum = 0.0
    n_pos = 0
    for i in numba.prange(n):
        p = y_pred[i]
        if p < 0.0 or p > 1.0:
            bad += 1
            continue
        if p < eps:
            p = eps
        elif p > 1 - eps:
            p = 1 - eps
        if y_true[i] == 1:
            loss_sum -= np.log(p)
            n_pos += 1
        else:
            loss_sum -= np.log(1 - p)

    if bad > 0:
        return np.nan
    # Need at least one of each class.
    if n_pos == 0 or n_pos == n:
        return np.nan
    return loss_sum / n


def fast_log_loss_binary(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-15) -> float:
    """Numba-accelerated binary log loss (cross-entropy), auto seq/par.

    Equivalent to sklearn.metrics.log_loss for binary classification.
    Faster than sklearn due to no input validation overhead. Above
    N=100k rows the parallel kernel is selected (~4.8x faster than
    seq at N=10M, 8-thread runtime).

    Returns np.nan if only one class is present in y_true.
    """
    if len(y_true) >= _PARALLEL_REDUCTION_THRESHOLD:
        return _fast_log_loss_binary_par(y_true, y_pred, eps)
    return _fast_log_loss_binary_seq(y_true, y_pred, eps)


def fast_log_loss(y_true: np.ndarray, y_pred: np.ndarray, eps: float = None) -> float:
    """Fast log loss using numba. Drop-in replacement for sklearn.metrics.log_loss.

    Args:
        y_true: Ground truth labels (0 or 1).
        y_pred: Predicted probabilities for class 1.
        eps: Small value for clipping to prevent log(0). Default uses dtype's eps (sklearn-compatible).

    Returns:
        Binary cross-entropy loss.
    """
    if isinstance(y_true, (pd.Series, pl.Series)):
        y_true = y_true.to_numpy()
    if isinstance(y_pred, (pd.Series, pl.Series)):
        y_pred = y_pred.to_numpy()

    # Ensure float type for numba
    if y_true.dtype not in (np.float32, np.float64):
        y_true = y_true.astype(np.float64)
    if y_pred.dtype not in (np.float32, np.float64):
        y_pred = y_pred.astype(np.float64)

    # Use dtype's epsilon for sklearn compatibility (eps="auto" behavior)
    if eps is None:
        eps = np.finfo(y_pred.dtype).eps

    return fast_log_loss_binary(y_true, y_pred, eps)


@numba.njit(**NUMBA_NJIT_PARAMS)
def _probability_separation_score_seq(y_true: np.ndarray, y_prob: np.ndarray, class_label: int = 1, std_weight: float = 0.5) -> float:
    """Sequential variant. Public wrapper auto-dispatches at N>=50k."""
    idx = y_true == class_label
    if idx.sum() == 0:
        return np.nan
    res = np.mean(y_prob[idx])
    if std_weight != 0.0:
        addend = np.std(y_prob[idx]) * std_weight
        if class_label == 1:
            res = res - addend
        else:
            res = res + addend
    return res


@numba.njit(**NUMBA_NJIT_PARAMS, parallel=True)
def _probability_separation_score_par(y_true: np.ndarray, y_prob: np.ndarray, class_label: int = 1, std_weight: float = 0.5) -> float:
    """Parallel variant. ~5x faster than seq at N=1M.

    Two prange passes: first computes count + sum (mean), second
    computes variance using the mean. Both passes use ``+=``
    reductions which numba auto-recognises."""
    n = len(y_true)
    if n == 0:
        return np.nan
    n_in = 0
    s = 0.0
    for i in numba.prange(n):
        if y_true[i] == class_label:
            n_in += 1
            s += y_prob[i]
    if n_in == 0:
        return np.nan
    mean = s / n_in
    if std_weight == 0.0:
        return mean

    sse = 0.0
    for i in numba.prange(n):
        if y_true[i] == class_label:
            d = y_prob[i] - mean
            sse += d * d
    std = np.sqrt(sse / n_in)
    addend = std * std_weight
    if class_label == 1:
        return mean - addend
    else:
        return mean + addend


def probability_separation_score(y_true: np.ndarray, y_prob: np.ndarray, class_label: int = 1, std_weight: float = 0.5) -> float:
    """Mean predicted probability for the in-class population, optionally
    discounted by std (separation = mean - std * weight). Auto seq/par
    dispatch above N=50k (~5x faster at N=1M)."""
    if len(y_true) >= _PARALLEL_MULTILABEL_THRESHOLD:
        return _probability_separation_score_par(y_true, y_prob, class_label, std_weight)
    return _probability_separation_score_seq(y_true, y_prob, class_label, std_weight)
