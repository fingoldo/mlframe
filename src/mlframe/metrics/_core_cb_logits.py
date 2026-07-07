"""CatBoost logits->probability kernels for ``mlframe.metrics.core``.

Carved from ``core.py``. Public symbols are re-exported from the parent.
"""

from __future__ import annotations

import numba
import numpy as np

from ._numba_params import NUMBA_NJIT_PARAMS, _PARALLEL_REDUCTION_THRESHOLD


@numba.njit(**NUMBA_NJIT_PARAMS)
def _cb_logits_to_probs_binary_seq(logits: np.ndarray) -> np.ndarray:
    """Sequential variant. Public wrapper auto-dispatches at N>=100k."""
    n = len(logits)
    probs = np.empty((n, 2), dtype=np.float64)
    for i in range(n):
        p1 = 1.0 / (1.0 + np.exp(-logits[i]))
        probs[i, 0] = 1.0 - p1
        probs[i, 1] = p1
    return probs


@numba.njit(**NUMBA_NJIT_PARAMS, parallel=True)
def _cb_logits_to_probs_binary_par(logits: np.ndarray) -> np.ndarray:
    """Parallel sigmoid."""
    n = len(logits)
    probs = np.empty((n, 2), dtype=np.float64)
    for i in numba.prange(n):
        p1 = 1.0 / (1.0 + np.exp(-logits[i]))
        probs[i, 0] = 1.0 - p1
        probs[i, 1] = p1
    return probs


def cb_logits_to_probs_binary(logits: np.ndarray) -> np.ndarray:
    """Convert CatBoost binary logits to probabilities, auto seq/par.

    Args:
        logits: 1D array of logits from CatBoost (single class output)

    Returns:
        2D array of shape (n_samples, 2) with probabilities for [class_0, class_1]
    """
    if len(logits) >= _PARALLEL_REDUCTION_THRESHOLD:
        return np.asarray(_cb_logits_to_probs_binary_par(logits))
    return np.asarray(_cb_logits_to_probs_binary_seq(logits))


@numba.njit(**NUMBA_NJIT_PARAMS)
def _cb_logits_to_probs_multiclass_seq(logits_list: np.ndarray) -> np.ndarray:
    """Sequential variant. Public wrapper auto-dispatches at N>=100k.

    bench-attempt-rejected (2026-05-22, c0108 / iter165): transposing
    the (K, N) input to (N, K) first to make inner-loop reads
    contiguous is 11-21% SLOWER at every size (K=3, 8; N=50k..1M).
    Modern CPU prefetchers handle the stride-N reads well for small K;
    the upfront 24-100 MB transpose memcpy never earns itself back.
    Bit-equivalent output. Bench:
    profiling/bench_cb_logits_softmax_layout.py.
    """
    n_classes, n_samples = logits_list.shape
    probs = np.empty((n_samples, n_classes), dtype=np.float64)
    for i in range(n_samples):
        max_logit = logits_list[0, i]
        for c in range(1, n_classes):
            if logits_list[c, i] > max_logit:
                max_logit = logits_list[c, i]
        exp_sum = 0.0
        for c in range(n_classes):
            probs[i, c] = np.exp(logits_list[c, i] - max_logit)
            exp_sum += probs[i, c]
        for c in range(n_classes):
            probs[i, c] /= exp_sum
    return probs


@numba.njit(**NUMBA_NJIT_PARAMS, parallel=True)
def _cb_logits_to_probs_multiclass_par(logits_list: np.ndarray) -> np.ndarray:
    """Parallel softmax."""
    n_classes, n_samples = logits_list.shape
    probs = np.empty((n_samples, n_classes), dtype=np.float64)
    for i in numba.prange(n_samples):
        max_logit = logits_list[0, i]
        for c in range(1, n_classes):
            if logits_list[c, i] > max_logit:
                max_logit = logits_list[c, i]
        exp_sum = 0.0
        for c in range(n_classes):
            probs[i, c] = np.exp(logits_list[c, i] - max_logit)
            exp_sum += probs[i, c]
        for c in range(n_classes):
            probs[i, c] /= exp_sum
    return probs


def cb_logits_to_probs_multiclass(logits_list: np.ndarray) -> np.ndarray:
    """Convert CatBoost multiclass logits to probabilities (softmax),
    auto seq/par.

    Args:
        logits_list: 2D array of shape (n_classes, n_samples) with logits

    Returns:
        2D array of shape (n_samples, n_classes) with probabilities
    """
    if logits_list.shape[1] >= _PARALLEL_REDUCTION_THRESHOLD:
        return np.asarray(_cb_logits_to_probs_multiclass_par(logits_list))
    return np.asarray(_cb_logits_to_probs_multiclass_seq(logits_list))
