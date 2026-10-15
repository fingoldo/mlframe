"""Confusion-count numba kernels for the classification extras, kept in a leaf module.

Leaf module: the classification facade and its block sibling both use these kernels, and this module imports nothing from the package.
"""

from __future__ import annotations

from typing import Tuple
import numba
import numpy as np
from .._numba_params import NUMBA_NJIT_PARAMS


@numba.njit(**NUMBA_NJIT_PARAMS)
def _confusion_counts_binary(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[int, int, int, int]:
    """Return (TP, FP, TN, FN) for binary 0/1 labels.

    Robust to any int/bool dtype: cast happens at call site. The kernel
    treats anything != 0 as positive.
    """
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for i in range(y_true.shape[0]):
        if y_true[i] != 0:
            if y_pred[i] != 0:
                tp += 1
            else:
                fn += 1
        else:
            if y_pred[i] != 0:
                fp += 1
            else:
                tn += 1
    return tp, fp, tn, fn


@numba.njit(**NUMBA_NJIT_PARAMS, parallel=True)
def _confusion_counts_binary_par(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    nthr: int,
) -> Tuple[int, int, int, int]:
    """Per-thread accumulators avoid the `+= scalar` reduction-detection
    edge case that fails on tuples (numba only auto-reduces scalars in
    prange). Final reduction outside the prange."""
    tp_a = np.zeros(nthr, dtype=np.int64)
    fp_a = np.zeros(nthr, dtype=np.int64)
    tn_a = np.zeros(nthr, dtype=np.int64)
    fn_a = np.zeros(nthr, dtype=np.int64)
    n = y_true.shape[0]
    for i in numba.prange(n):
        t = numba.get_thread_id()
        if y_true[i] != 0:
            if y_pred[i] != 0:
                tp_a[t] += 1
            else:
                fn_a[t] += 1
        else:
            if y_pred[i] != 0:
                fp_a[t] += 1
            else:
                tn_a[t] += 1
    return int(tp_a.sum()), int(fp_a.sum()), int(tn_a.sum()), int(fn_a.sum())


@numba.njit(**NUMBA_NJIT_PARAMS)
def _multiclass_confusion_kernel(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    K: int,
) -> np.ndarray:
    """Returns K x K confusion matrix C[true, pred]."""
    C = np.zeros((K, K), dtype=np.int64)
    for i in range(y_true.shape[0]):
        t = y_true[i]
        p = y_pred[i]
        if 0 <= t < K and 0 <= p < K:
            C[t, p] += 1
    return C
