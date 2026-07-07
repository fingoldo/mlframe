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

from typing import Optional

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
        return float(_fast_log_loss_binary_par(y_true, y_pred, eps))
    return float(_fast_log_loss_binary_seq(y_true, y_pred, eps))


def fast_log_loss(y_true: np.ndarray, y_pred: np.ndarray, eps: Optional[float] = None) -> float:
    """Fast log loss using numba. Drop-in replacement for sklearn.metrics.log_loss.

    Args:
        y_true: Ground truth labels (0 or 1).
        y_pred: Predicted probabilities for class 1.
        eps: Small value for clipping to prevent log(0). Default uses the input dtype's machine eps (``np.finfo(y_pred.dtype).eps``), matching sklearn's
            legacy ``eps="auto"``. This is dtype-DEPENDENT on purpose: float32 cannot represent probabilities in ``(1 - 1.19e-7, 1)`` -- a confident
            ``1 - 1e-8`` collapses to exactly ``1.0`` on cast -- so clipping a float32 input at the smaller float64 eps (2.22e-16) would penalise that
            unrepresentable-near-1 region with ``-log(2.22e-16) ~= 36`` instead of the intended ``-log(1e-8) ~= 18``, OVERSHOOTING the honest value. The
            float32 eps is the correct precision-matched floor for float32 inputs; pass an explicit ``eps`` to override.
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
        eps = float(np.finfo(y_pred.dtype).eps)

    return fast_log_loss_binary(y_true, y_pred, eps)


@numba.njit(**NUMBA_NJIT_PARAMS)
def _probability_separation_score_seq(y_true: np.ndarray, y_prob: np.ndarray, class_label: int = 1, std_weight: float = 0.5) -> float:
    """Sequential variant. Public wrapper auto-dispatches at N>=50k.

    iter-fused: two scalar-accumulating passes (count+sum, then centred SSE)
    replace the previous ``idx = y_true == class_label`` boolean-mask alloc +
    ``y_prob[idx]`` fancy-index copy (built TWICE, once for mean once for std)
    + np.mean + np.std. The old form allocated ~3 arrays and walked the data
    ~4 times; the fused form is zero-alloc, two passes. Mirrors the SEQ
    structure of ``_probability_separation_score_par`` below (without prange).
    Bench (``_benchmarks/bench_prob_separation_seq_fused.py``, py3.14/numba):
    2.86x@n=2k, 1.18-1.46x@n=10k, 1.16-1.21x@n=49k; max|diff|=0.0 (bit-
    identical: plain running sum vs np pairwise summation happened to agree to
    the last ULP on the tested shapes, the contract allows ~1e-15)."""
    n = y_true.shape[0]
    n_in = 0
    s = 0.0
    for i in range(n):
        if y_true[i] == class_label:
            n_in += 1
            s += y_prob[i]
    if n_in == 0:
        return np.nan
    mean = s / n_in
    if std_weight == 0.0:
        return mean
    sse = 0.0
    for i in range(n):
        if y_true[i] == class_label:
            d = y_prob[i] - mean
            sse += d * d
    std = np.sqrt(sse / n_in)
    addend = std * std_weight
    if class_label == 1:
        return float(mean - addend)
    return float(mean + addend)


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
        return float(mean - addend)
    else:
        return float(mean + addend)


def probability_separation_score(y_true: np.ndarray, y_prob: np.ndarray, class_label: int = 1, std_weight: float = 0.5) -> float:
    """Mean predicted probability for the in-class population, optionally
    discounted by std (separation = mean - std * weight). Auto seq/par
    dispatch above N=50k (~5x faster at N=1M)."""
    if len(y_true) >= _PARALLEL_MULTILABEL_THRESHOLD:
        return float(_probability_separation_score_par(y_true, y_prob, class_label, std_weight))
    return float(_probability_separation_score_seq(y_true, y_prob, class_label, std_weight))
