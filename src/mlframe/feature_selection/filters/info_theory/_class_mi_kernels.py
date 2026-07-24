"""MI / SU computed directly from pre-computed class vectors + marginals (the permutation-loop scoring path where ``classes_y`` is shuffled in place)."""
from __future__ import annotations

import math

import numpy as np
from numba import njit

from ._state_and_dispatch import use_mi_chao_shen, use_mi_miller_madow, use_su_normalization


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
def _chao_shen_entropy_from_counts(counts: np.ndarray, n: int) -> float:
    """Chao & Shen (2003) coverage-adjusted entropy estimator from a category count vector.

    Corrects the plug-in entropy estimator's downward bias from UNSEEN categories by rescaling each
    observed frequency by the estimated sample coverage ``C_hat = 1 - f1/n`` (``f1`` = count of
    singleton categories) and applying the Horvitz-Thompson-style ``1 - (1-p)^n`` inclusion-probability
    correction per category. Falls back to the plug-in estimator (mathematically equivalent to
    ``C_hat=1``) when every observed category is a singleton (``f1==n``, coverage estimate undefined)
    or when the per-category correction denominator underflows -- both edge cases where the plug-in
    term is already a reasonable finite-sample estimate and the correction would otherwise blow up.
    """
    f1 = 0
    for i in range(len(counts)):
        if counts[i] == 1:
            f1 += 1
    if f1 >= n:
        c_hat = (n - 1.0) / n if n > 1 else 1.0
    else:
        c_hat = 1.0 - f1 / n
    h = 0.0
    for i in range(len(counts)):
        ni = counts[i]
        if ni <= 0:
            continue
        p_tilde = c_hat * ni / n
        if p_tilde <= 0.0:
            continue
        lam = 1.0 - (1.0 - p_tilde) ** n
        term = -p_tilde * math.log(p_tilde)
        if lam > 1e-12:
            term = term / lam
        h += term
    return h


@njit(nogil=True, cache=True)
def compute_mi_cs_from_classes(
    classes_x: np.ndarray,
    freqs_x: np.ndarray,
    classes_y: np.ndarray,
    freqs_y: np.ndarray,
    dtype: type = np.int32,
) -> float:
    """Chao-Shen (2003) coverage-adjusted MI from pre-computed class arrays: ``H_CS(X) + H_CS(Y) -
    H_CS(X,Y)``, floored at 0. Unlike Miller-Madow's closed-form additive bias term, Chao-Shen
    directly re-estimates each marginal/joint entropy from its own observed-category coverage, which
    better tracks bias on SPARSE high-cardinality joints (many singleton cells) where Miller-Madow's
    ``(k_x-1)(k_y-1)/(2n)`` term systematically under-corrects (findings #7, 05_concurrency_and_statistics.md).
    """
    n = len(classes_x)
    K_x = len(freqs_x)
    K_y = len(freqs_y)
    counts_x = np.zeros(K_x, dtype=np.int64)
    counts_y = np.zeros(K_y, dtype=np.int64)
    joint_counts = np.zeros((K_x, K_y), dtype=np.int64)
    for k in range(n):
        cx = classes_x[k]
        cy = classes_y[k]
        counts_x[cx] += 1
        counts_y[cy] += 1
        joint_counts[cx, cy] += 1
    h_x = _chao_shen_entropy_from_counts(counts_x, n)
    h_y = _chao_shen_entropy_from_counts(counts_y, n)
    joint_flat = joint_counts.reshape(K_x * K_y)
    h_xy = _chao_shen_entropy_from_counts(joint_flat, n)
    mi = h_x + h_y - h_xy
    return mi if mi > 0.0 else 0.0


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
def weighted_class_freqs(classes: np.ndarray, weights: np.ndarray, n_classes: int) -> np.ndarray:
    """Weighted marginal frequencies ``P(class)`` from a pre-computed class array + per-row weights.

    Companion to ``merge_vars``'s unweighted ``freqs`` output: binning is weight-independent (a row's
    bin assignment doesn't depend on its weight), so any ``merge_vars`` ``classes`` array can be
    re-weighted here without re-running the njit merge itself.
    """
    out = np.zeros(n_classes, dtype=np.float64)
    total = 0.0
    for k in range(len(classes)):
        w = weights[k]
        out[classes[k]] += w
        total += w
    if total > 0.0:
        for i in range(n_classes):
            out[i] /= total
    return out


@njit(nogil=True, cache=True)
def compute_mi_from_classes_weighted(
    classes_x: np.ndarray,
    classes_y: np.ndarray,
    weights: np.ndarray,
    dtype=np.int32,
) -> float:
    """Weighted mutual information from two pre-computed class arrays + per-row sample weights.

    cat-FE's downstream confirmation/rerank steps (MM re-rank,
    permutation test, bandit-UCB1, Westfall-Young, bootstrap CI, K-fold stability,
    anti-redundancy rerank, k-way greedy expansion) previously recomputed every statistic
    UNWEIGHTED even when the search-phase point estimate used sample weights -- a weighted
    ``II_obs`` was tested against an unweighted null/refinement. This weighted joint-MI kernel
    (weighted joint counts + weighted marginals, mirroring ``compute_mi_from_classes`` exactly
    when ``weights`` is uniform) is the shared primitive threaded through every downstream step.
    """
    n = len(classes_x)
    k_x = int(classes_x.max()) + 1 if n > 0 else 0
    k_y = int(classes_y.max()) + 1 if n > 0 else 0
    joint_w = np.zeros((k_x, k_y), dtype=np.float64)
    marg_x = np.zeros(k_x, dtype=np.float64)
    marg_y = np.zeros(k_y, dtype=np.float64)
    total_w = 0.0
    for k in range(n):
        w = weights[k]
        cx = classes_x[k]
        cy = classes_y[k]
        joint_w[cx, cy] += w
        marg_x[cx] += w
        marg_y[cy] += w
        total_w += w
    if total_w <= 0.0:
        return 0.0
    inv_w = 1.0 / total_w
    total = 0.0
    for i in range(k_x):
        prob_x = marg_x[i] * inv_w
        if prob_x <= 0.0:
            continue
        for j in range(k_y):
            jc = joint_w[i, j]
            if jc != 0.0:
                prob_y = marg_y[j] * inv_w
                jf = jc * inv_w
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
    use_cs: bool = False,
) -> float:
    """njit-callable dispatcher between raw MI, Symmetric Uncertainty, Miller-Madow- and
    Chao-Shen-corrected MI.

    ``permutation.py``'s njit kernels cannot read the Python-level thread-local
    SU/MM/CS toggles directly; this branch-on-flag helper lets the joblib entry point
    (``mi_direct``) thread the mode down once per call, and the njit kernel
    selects the scorer at runtime with a single bool check.

    All branches share the same dtype + array contracts as
    ``compute_mi_from_classes`` so the existing permutation loops stay
    byte-for-byte stable in the SU/MM/CS-off code path. ``use_mm``/``use_cs`` are mutually
    exclusive in practice (MRMR's ``mi_correction`` is a single string knob); ``use_cs`` is
    checked first if both are somehow set.
    """
    if use_su:
        return float(compute_su_from_classes(classes_x, freqs_x, classes_y, freqs_y, dtype=dtype))
    if use_cs:
        return float(compute_mi_cs_from_classes(classes_x, freqs_x, classes_y, freqs_y, dtype=dtype))
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
    if use_mi_chao_shen():
        return float(compute_mi_cs_from_classes(classes_x, freqs_x, classes_y, freqs_y, dtype=dtype))
    if use_mi_miller_madow():
        return float(compute_mi_mm_from_classes(classes_x, freqs_x, classes_y, freqs_y, dtype=dtype))
    return float(compute_mi_from_classes(classes_x, freqs_x, classes_y, freqs_y, dtype=dtype))
