"""Densified-column relevance (MI / SU) kernel shared by the batched FE-candidate noise-gate kernels."""
from __future__ import annotations

import math

import numpy as np
from numba import njit


@njit(nogil=True, cache=True)
def _relevance_from_dense(
    use_su: bool,
    classes_dense: np.ndarray,
    k: int,
    freqs_dense: np.ndarray,
    K_x: int,
    classes_y: np.ndarray,
    freqs_y: np.ndarray,
    dtype,
) -> float:
    """MI (or SU) of densified column ``k`` against ``y`` - inlined twin of
    ``compute_relevance_score`` reading the column from the padded ``classes_dense`` /
    ``freqs_dense`` buffers. Numerically identical to calling
    ``compute_relevance_score(use_su, classes_dense[:, k], freqs_dense[k, :K_x], ...)``:
    same indexed joint-count pass, same ``jf * log(jf / (px * py))`` accumulation order.
    """
    n = classes_dense.shape[0]
    K_y = len(freqs_y)
    # bench-attempt-rejected (2026-06-07): REUSE a thread-local joint_counts scratch
    # (indexed by numba.get_thread_id()) instead of this per-call ``np.zeros`` to remove
    # the K*(nperm+1) allocations (Q5). BYTE-IDENTICAL but a NET LOSS: 0.58x (K=300) ..
    # 0.77x (K=4000) on the scene-like CPU gate. numba pool/stack-allocates this tiny
    # (K_x x K_y ~ 10x2) array essentially for free; a shared (nthreads, max_Kx*K_y)
    # scratch adds get_thread_id() + a manual zeroing loop + flat-index arithmetic +
    # cross-thread false sharing that all cost MORE than the elided alloc. Keep the
    # per-call np.zeros. (proto D:/Temp/q5_scratch_proto.py)
    # int64 counter: a joint cell can reach ``n``; int32 (default ``dtype``) wraps negative above ~2.1e9 rows -> NaN MI.
    joint_counts = np.zeros((K_x, K_y), dtype=np.int64)
    for r in range(n):
        joint_counts[classes_dense[r, k], classes_y[r]] += 1
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
