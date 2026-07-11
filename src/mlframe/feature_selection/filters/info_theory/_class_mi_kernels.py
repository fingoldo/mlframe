"""MI / SU computed directly from pre-computed class vectors + marginals (the permutation-loop scoring path where ``classes_y`` is shuffled in place)."""
from __future__ import annotations

import math

import numpy as np
from numba import njit

from ._state_and_dispatch import use_mi_miller_madow, use_su_normalization


@njit(nogil=True, cache=True)
def _mm_bias(freqs_x: np.ndarray, freqs_y: np.ndarray, n: int) -> float:
    """Closed-form Miller-Madow MI bias ``(k_x-1)(k_y-1)/(2n)`` from occupied-bin counts of the pre-computed marginals (freqs are probabilities, ``>0`` = occupied)."""
    k_x = 0
    for i in range(len(freqs_x)):
        if freqs_x[i] > 0:
            k_x += 1
    k_y = 0
    for j in range(len(freqs_y)):
        if freqs_y[j] > 0:
            k_y += 1
    if k_x <= 1 or k_y <= 1 or n <= 0:
        return 0.0
    return (k_x - 1) * (k_y - 1) / (2.0 * n)


@njit(nogil=True, cache=True)
def compute_mi_mm_from_classes(
    classes_x: np.ndarray,
    freqs_x: np.ndarray,
    classes_y: np.ndarray,
    freqs_y: np.ndarray,
    dtype=np.int32,
) -> float:
    """Miller-Madow bias-corrected MI from pre-computed class arrays + marginals: ``I_plugin - (k_x-1)(k_y-1)/(2n)``, floored at 0.

    Subtracts the closed-form small-sample bias so a high-cardinality NOISE column no longer out-ranks a low-cardinality TRUE-relevant column by sheer entropy at
    small n. The bias term is identical across permutations of y (k_x, k_y, n fixed), so under the significance-gated null-debias screen it cancels in
    ``observed - null_mean``; it bites on the RAW relevance ranking (no-null / non-gated paths) and on the absolute relevance floor.
    """
    mi_xy = compute_mi_from_classes(classes_x, freqs_x, classes_y, freqs_y, dtype=dtype)
    corrected = mi_xy - _mm_bias(freqs_x, freqs_y, len(classes_x))
    return corrected if corrected > 0.0 else 0.0


@njit(nogil=True, cache=True)
def compute_mi_from_classes(
    classes_x: np.ndarray,
    freqs_x: np.ndarray,
    classes_y: np.ndarray,
    freqs_y: np.ndarray,
    dtype=np.int32,
) -> float:
    """Mutual information from two pre-computed class arrays + their marginals. Used by the permutation loop where ``classes_y`` is shuffled in place and we don't want to re-bin from scratch each time.

    Indexed range-loop over zip-iter: ~25% faster at n=50k..1M (numba's
    zip pair unboxing adds per-iteration overhead vs the indexed form).
    On-the-fly freq computation via inv_n also avoids the (K_x, K_y)
    float64 joint_freqs intermediate allocation. Bench:
    profiling/bench_compute_mi_from_classes_no_zip.py.
    """
    n = len(classes_x)
    K_x = len(freqs_x)
    K_y = len(freqs_y)
    # A single joint cell count can reach ``n``; the caller-supplied ``dtype`` (int32 by default) wraps negative above ~2.1e9 rows, turning ``log(jf/...)`` into NaN. The counter is bounded by ``n`` regardless of class encoding, so int64 is the safe width at negligible cost for the small (K_x, K_y) array (mirrors the ``merge_vars`` freqs/lookup-table int64 fix).
    joint_counts = np.zeros((K_x, K_y), dtype=np.int64)
    for k in range(n):
        joint_counts[classes_x[k], classes_y[k]] += 1
    inv_n = 1.0 / n

    total = 0.0
    for i in range(K_x):
        prob_x = freqs_x[i]
        for j in range(K_y):
            jc = joint_counts[i, j]
            if jc != 0:
                prob_y = freqs_y[j]
                jf = jc * inv_n
                total += jf * math.log(jf / (prob_x * prob_y))
    return total


@njit(nogil=True, cache=True)
def compute_su_from_classes(
    classes_x: np.ndarray,
    freqs_x: np.ndarray,
    classes_y: np.ndarray,
    freqs_y: np.ndarray,
    dtype=np.int32,
) -> float:
    """Symmetric Uncertainty from pre-computed class arrays + marginals.

    SU(X, Y) = 2 * I(X; Y) / (H(X) + H(Y)). Built atop the same joint-counts
    pass as ``compute_mi_from_classes`` so the permutation loop in
    ``permutation.py`` can swap to this scorer when ``mi_normalization='su'``
    without recomputing classes/freqs. Reuses the freqs_x / freqs_y arrays
    to compute H(X), H(Y) -- one log-pass per marginal, O(K_x + K_y).
    """
    n = len(classes_x)
    K_x = len(freqs_x)
    K_y = len(freqs_y)
    # int64 counter: a joint cell can reach ``n``; int32 (default ``dtype``) wraps negative above ~2.1e9 rows -> NaN SU. See ``compute_mi_from_classes``.
    joint_counts = np.zeros((K_x, K_y), dtype=np.int64)
    for k in range(n):
        joint_counts[classes_x[k], classes_y[k]] += 1
    inv_n = 1.0 / n
    mi_xy = 0.0
    for i in range(K_x):
        prob_x = freqs_x[i]
        for j in range(K_y):
            jc = joint_counts[i, j]
            if jc != 0:
                prob_y = freqs_y[j]
                jf = jc * inv_n
                mi_xy += jf * math.log(jf / (prob_x * prob_y))
    h_x = 0.0
    for i in range(K_x):
        p = freqs_x[i]
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
    # Floor the plug-in numerator at 0 (matching the CMI clamp): on near-deterministic columns float round-off in ``H(X)+H(Y)-H(XY)`` can leave ``mi_xy`` slightly negative, yielding a tiny negative SU treated as a valid low relevance instead of 0.
    if mi_xy < 0.0:
        mi_xy = 0.0
    return 2.0 * mi_xy / denom


@njit(nogil=True, cache=True)
def compute_relevance_score(
    use_su: bool,
    classes_x: np.ndarray,
    freqs_x: np.ndarray,
    classes_y: np.ndarray,
    freqs_y: np.ndarray,
    dtype=np.int32,
    use_mm: bool = False,
) -> float:
    """njit-callable dispatcher between raw MI and Symmetric Uncertainty.

    ``permutation.py``'s njit kernels cannot read the Python-level thread-local
    SU toggle directly; this branch-on-flag helper lets the joblib entry point
    (``mi_direct``) thread the SU mode down once per call, and the njit kernel
    selects the scorer at runtime with a single bool check.

    Both branches share the same dtype + array contracts as
    ``compute_mi_from_classes`` so the existing permutation loops stay
    byte-for-byte stable in the SU-off code path.
    """
    if use_su:
        return float(compute_su_from_classes(classes_x, freqs_x, classes_y, freqs_y, dtype=dtype))
    if use_mm:
        return float(compute_mi_mm_from_classes(classes_x, freqs_x, classes_y, freqs_y, dtype=dtype))
    return float(compute_mi_from_classes(classes_x, freqs_x, classes_y, freqs_y, dtype=dtype))


def mi_or_su_from_classes(classes_x, freqs_x, classes_y, freqs_y, dtype=np.int32) -> float:
    """Dispatch raw MI or SU from pre-computed classes based on the thread-local toggle.

    Cheap when SU is off: one Python-call delegation to the njit ``compute_mi_from_classes``.
    Used by ``permutation.py`` so the relevance gate in MRMR's simple-mode path picks up
    the cardinality-bias-corrected scorer when ``MRMR(mi_normalization='su')``.
    """
    if use_su_normalization():
        return float(compute_su_from_classes(classes_x, freqs_x, classes_y, freqs_y, dtype=dtype))
    if use_mi_miller_madow():
        return float(compute_mi_mm_from_classes(classes_x, freqs_x, classes_y, freqs_y, dtype=dtype))
    return float(compute_mi_from_classes(classes_x, freqs_x, classes_y, freqs_y, dtype=dtype))
