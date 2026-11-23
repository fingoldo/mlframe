"""FUSED dense-code write + observed MI of one column (carved out of ``_batch_kernels``, which hit the 1k-LOC budget).

Re-exported by ``_batch_kernels``; the function body is unchanged.
"""
from __future__ import annotations

import math

import numpy as np
from numba import njit


@njit(nogil=True, cache=True)
def _densify_and_relevance_fused(
    use_su: bool,
    col: np.ndarray,
    lookup: np.ndarray,
    classes_dense: np.ndarray,
    k: int,
    freqs_dense: np.ndarray,
    K_x: int,
    classes_y: np.ndarray,
    freqs_y: np.ndarray,
    dtype,
) -> float:
    """FUSED dense-code WRITE + observed-MI of column ``k`` in ONE pass over the n rows (F2, 2026-06-22,
    cProfile-driven). The legacy path made TWO strided full-column passes per candidate: one to write
    ``classes_dense[r,k] = lookup[col[r]]`` and a SECOND inside ``_relevance_from_dense`` to re-read
    ``classes_dense[r,k]`` + ``classes_y[r]`` into the joint histogram. At the canonical 30k-subsample
    K~3888 chunk that observed-MI sweep is the dominant CPU FE cost (~5.8s/chunk even with npermutations=0,
    i.e. the permutation shuffles are NOT the cost - this densify/relevance double-pass is). Fusing them
    accumulates the (dense_x, y) joint WHILE writing the dense code, so the n-row column is touched once
    here instead of twice. BIT-IDENTICAL to ``classes_dense[r,k]=lookup[col[r]]`` followed by
    ``_relevance_from_dense(...)``: the dense codes written are the same values, the joint counts are the
    same increments, and the ``jf*log(jf/(px*py))`` reduction order over the (K_x, K_y) histogram is
    unchanged. ``classes_dense`` is still fully written (the permutation path re-reads it per shuffle)."""
    n = col.shape[0]
    K_y = len(freqs_y)
    # int64 counter: a joint cell can reach ``n``; int32 (default ``dtype``) wraps negative above ~2.1e9 rows -> NaN MI.
    joint_counts = np.zeros((K_x, K_y), dtype=np.int64)
    for r in range(n):
        dc = lookup[col[r]]
        classes_dense[r, k] = dc
        joint_counts[dc, classes_y[r]] += 1
    inv_n = 1.0 / n

    mi_xy = 0.0
    for i in range(K_x):
        prob_x = freqs_dense[k, i]
        for j in range(K_y):
            jc = joint_counts[i, j]
            if jc != 0:
                prob_y = freqs_y[j]
                jf = jc * inv_n
                mi_xy += jf * math.log(jf / (prob_x * prob_y))

    if not use_su:
        return mi_xy

    h_x = 0.0
    for i in range(K_x):
        p = freqs_dense[k, i]
        if p > 0:
            h_x -= p * math.log(p)
    h_y = 0.0
    for j in range(K_y):
        p = freqs_y[j]
        if p > 0:
            h_y -= p * math.log(p)
    denom = h_x + h_y
    if denom <= 1e-12:
        return 0.0
    # Floor the plug-in numerator at 0: float round-off on near-deterministic columns can leave ``mi_xy`` slightly negative -> tiny negative SU treated as valid low relevance instead of 0.
    if mi_xy < 0.0:
        mi_xy = 0.0
    return 2.0 * mi_xy / denom
