"""CPU permutation testing for the mRMR confidence step.

Public functions
----------------
* ``mi_direct(factors_data, x, y, ...)`` -- compute the original MI of ``(x, y)``, then permute ``y`` ``npermutations`` times and count how often the permuted
  MI exceeds the original. Optionally distributes the permutations across joblib workers.
* ``parallel_mi`` -- the inner ``@njit`` worker that ``mi_direct`` ships to joblib pool members.
* ``shuffle_arr`` -- ``@njit`` shim around ``np.random.shuffle`` so it's callable from ``parallel_mi``.
* ``distribute_permutations`` -- partition the permutation budget across worker chunks.

Both ``parallel_mi`` and the caller-side ``confidence = 1 - nfailed / (i + 1)`` guard against ``npermutations == 0`` (returns ``(0, 0)`` from the worker,
confidence ``0.0`` from the caller) to avoid ``UnboundLocalError`` on ``i`` when the for-loop never executes.
"""
from __future__ import annotations

import logging
import math
from typing import Optional, cast

import numpy as np
from joblib import Parallel
from numba import njit, prange

from ._internals import NMAX_NONPARALLEL_ITERS

# Typed uint64 zero used as the default ``base_seed`` for the njit permutation kernels. Built once as a
# module-level singleton so the argument default keeps its exact ``np.uint64`` dtype (the kernels do
# ``state >> np.uint64(..)`` arithmetic) without a function call in the signature default.
_DEFAULT_BASE_SEED = np.uint64(0)

# HISTORICAL (2026-07-19): a first pass at this fix gated the "outer" joblib.Parallel pool behind a measured
# floor, ``_OUTER_PARALLEL_MIN_PERMUTATIONS = 64`` (npermutations=10 -> 0.36-0.40x SLOWER than serial at
# n=99401, =100 -> 1.45-1.79x, crossover ~50-100). A follow-up head-to-head bench
# (``_benchmarks/bench_permutation_njit_prange_vs_joblib.py``) found the existing ``parallel_mi_prange`` njit
# kernel dominates the joblib pool at EVERY scale tested (including above the floor), so the pool branch was
# retired outright (see the comment block at its former call site) rather than merely gated -- the constant
# itself is no longer needed and was removed as dead code.
from .info_theory import (
    merge_vars, compute_relevance_score, use_su_normalization, use_mi_miller_madow, use_mi_chao_shen,
)

logger = logging.getLogger(__name__)

# GPU CIRCUIT BREAKER (2026-07-09 fix, mirrors info_theory._cmi_cuda's ``_CMI_GPU_FAILED``). Before this,
# ``mi_direct``'s GPU fastpath had no persistent failure memory: a CUDA context poisoned by ONE launch fault
# (see _cmi_cuda.py's own circuit-breaker docstring for the mechanism -- a launch fault poisons the context, so
# every subsequent launch on it faults identically) meant every SUBSEQUENT ``mi_direct`` call would re-attempt
# the GPU, fail again, and pay the same failed-launch overhead -- a retry-storm identical in kind to the
# "1515 futile GPU retries" incident that motivated the CMI-path breaker, just not yet mirrored here. Trips on
# the first GPU-fastpath exception; every subsequent call in this process routes straight to CPU without
# re-attempting. Reset only via ``reset_mi_direct_gpu_circuit_breaker()`` (tests / a fresh CUDA context).
_MI_DIRECT_GPU_FAILED = False


def reset_mi_direct_gpu_circuit_breaker() -> None:
    """Re-arm ``mi_direct``'s GPU fastpath (tests / after a fresh CUDA context)."""
    global _MI_DIRECT_GPU_FAILED
    _MI_DIRECT_GPU_FAILED = False


# Number of y-permutations used by ``mi_direct(return_null_mean=True)`` to estimate BOTH the empirical relevance null mean AND the permutation p-value that gates the
# significance-aware debiasing in ``evaluate_candidate``. The MRMR screen's exceedance budget (``baseline_npermutations``, default 2) is far too few shuffles for either
# purpose -- 2 samples give a null-mean estimate with ~70% relative noise and a p-value that resolves only to {0, 0.5, 1.0}. 32 serves two needs: (a) it brings the null-mean
# standard error down ~4x, and (b) it makes the p-value resolve to 1/32 ~ 0.031, FINE ENOUGH for a textbook alpha=0.05 significance cut to cleanly separate weak-but-real
# signal (sits above its null, p ~ 0) from spurious noise (sits within its null, p large) -- at 16 perms the p-resolution (1/16 ~ 0.0625) is coarser than alpha and the gate
# would have to demand ZERO exceedances, which is too strict for a genuinely-weak leg whose null occasionally ties it. 32 of these per candidate is still microseconds on the
# screening hot path (each is one ``compute_relevance_score`` call). Tunable via the ``MLFRAME_MRMR_NULL_PERMS`` env var for users who want a tighter/looser null estimate.
import os as _os
# VARIANCE CAVEAT (mrmr_critique N-F5, DOC): the null MEAN estimated from ``_NULL_MEAN_MIN_PERMS`` (default 32)
# shuffles has a sampling SE of ~sigma_null/sqrt(32) (~18% of the per-shuffle spread). That SE is subtracted directly
# from the observed relevance in the significance-gated debiasing, so two candidates whose debiased relevance is
# within the null-mean SE of each other can swap order seed-to-seed in the irreversible greedy path. Raise
# ``MLFRAME_MRMR_NULL_PERMS`` (finer null, proportional cost) when near-tie stability matters; a shrunk/analytic
# null mean at small budgets is a FUTURE option.
_NULL_MEAN_MIN_PERMS = max(2, int(_os.environ.get("MLFRAME_MRMR_NULL_PERMS", "32")))


@njit(nogil=True, cache=True)
def _relevance_mi_1var_fused(
    factors_data: np.ndarray,
    ix: int,
    nb_x: int,
    classes_y: np.ndarray,
    freqs_y: np.ndarray,
) -> tuple:
    """Fused single-variable raw MI I(X_ix; Y) + occupied-x-bin count in ONE O(n) pass.

    Returns ``(mi, bx)`` where ``mi`` is BIT-IDENTICAL to
    ``compute_mi_from_classes(*merge_vars(factors_data, [ix], ...)[:2], classes_y, freqs_y)``
    and ``bx`` equals the pruned ``len(freqs_x)`` (occupied x-bins) that
    ``merge_vars`` returns -- exactly the two quantities the ANALYTIC-null branch of
    ``mi_direct`` consumes (``_ax_freqs.shape[0]`` and the MI). The legacy path built the
    length-n ``classes_x`` relabel array via ``merge_vars`` (one O(n) accumulate pass plus a
    second O(n) lookup-remap pass when any x-bin is empty) and then walked all n samples AGAIN
    inside ``compute_mi_from_classes`` to build the joint histogram -- and the analytic branch
    DISCARDS ``classes_x`` entirely (it needs only ``len(freqs_x)`` and the MI scalar). This
    kernel builds the ``(nb_x, K_y)`` joint histogram directly from ``factors_data[:, ix]`` and
    ``classes_y`` in a SINGLE row pass, then derives ``freqs_x`` from the row sums -- eliminating
    the separate ``merge_vars(x)`` pass + its ``final_classes``/lookup allocations. Mirrors the
    ``joint_freqs_2var`` / ``joint_entropy_2var`` pruned-fast-path wins on the DCD pairwise path.

    BIT-IDENTITY to ``compute_mi_from_classes(merge_vars(...))`` (raw MI, no SU/MM):
      * ``merge_vars`` renumbers OCCUPIED x-bins densely in ASCENDING original-bin order, so
        ``compute_mi_from_classes`` visits x-classes ``i in range(len(freqs_x))`` in that same
        ascending order. Iterating ``i in range(nb_x)`` here and SKIPPING empty rows
        (``rowcount == 0``) visits the identical occupied bins in the identical order.
      * ``prob_x = rowcount / n`` reproduces ``merge_vars``'s ``freqs / n_rows`` (int64 count,
        float64 true division); ``prob_y = freqs_y[j]`` is passed through unchanged; and
        ``jf = jc * inv_n`` with ``inv_n = 1.0 / n`` matches ``compute_mi_from_classes`` exactly
        -- same operands, same order, so the ``jf*log(jf/(prob_x*prob_y))`` terms accumulate
        bit-for-bit.
      * Empty x-bins (and empty joint cells) contribute nothing to MI in EITHER path, so pruning
        vs skipping is numerically inert.
    """
    n = factors_data.shape[0]
    K_y = len(freqs_y)
    if n == 0:
        return 0.0, 0
    joint = np.zeros((nb_x, K_y), dtype=np.int64)
    for k in range(n):
        joint[factors_data[k, ix], classes_y[k]] += 1
    inv_n = 1.0 / n
    total = 0.0
    bx = 0
    for i in range(nb_x):
        rowcount = 0
        for j in range(K_y):
            rowcount += joint[i, j]
        if rowcount == 0:
            continue
        bx += 1
        prob_x = rowcount / n
        for j in range(K_y):
            jc = joint[i, j]
            if jc != 0:
                prob_y = freqs_y[j]
                jf = jc * inv_n
                total += jf * math.log(jf / (prob_x * prob_y))
    return total, bx


def _addone_pvalue_enabled() -> bool:
    """Whether the permutation p-value uses the add-one (Monte-Carlo) estimator ``p = (1 + nfailed) / (nchecked + 1)`` instead of the plain ``nfailed / nchecked``.

    The plain rate is a biased-low estimator of the true tail probability: with a finite permutation budget it can read p=0 (confidence=1.0) for a feature whose true p is
    merely small, and on discrete / low-cardinality data the exceedance comparator ``mi_perm >= original_mi`` counts ties as failures, so the plain rate is ALSO upward-biased
    on the noise side. The add-one form ``(1 + nfailed)/(nchecked + 1)`` is the standard, unbiased Monte-Carlo p-value estimator (Davison & Hinkley 1997; Phipson & Smyth 2010):
    it can never assert p exactly 0 from a finite sample and corrects the tie bias toward a calibrated value. SELECTION-ALTERING (it shifts the confidence each candidate
    receives), so it is gated. Default ON per the corrective-mechanism convention after the multi-seed selection bench (``_benchmarks/bench_perm_pvalue_addone.py``) confirmed it
    does not worsen selection on discrete data. Set ``MLFRAME_MRMR_ADDONE_PVALUE=0`` for the legacy plain-rate estimator.
    """
    return _os.environ.get("MLFRAME_MRMR_ADDONE_PVALUE", "1") != "0"


def _perm_pvalue(nfailed: int, nchecked: int, full_budget: Optional[int] = None) -> float:
    """Permutation p-value from the exceedance count.

    ``full_budget`` (P2): when an early-break path stopped at ``nchecked < full_budget`` because ``nfailed`` blew past ``max_failed``, the plain ``nfailed/nchecked`` OVERSTATES
    the failure rate relative to a full-budget run (the run stopped precisely because failures were piling up early). Passing the full budget makes the denominator budget-
    consistent so the returned confidence does not depend on WHERE the early break happened to fire. The add-one numerator/denominator both then use the full budget.
    """
    if nchecked <= 0:
        return 1.0
    denom = int(full_budget) if (full_budget is not None and int(full_budget) > nchecked) else nchecked
    if _addone_pvalue_enabled():
        return (1.0 + nfailed) / (denom + 1.0)
    return nfailed / float(denom)


@njit(cache=True)
def distribute_permutations(npermutations: int, n_workers: int) -> list:
    """Split ``npermutations`` across ``n_workers``; the remainder lands on the last worker."""
    avg_perms_per_worker = npermutations // n_workers
    diff = npermutations - avg_perms_per_worker * n_workers
    workload = [avg_perms_per_worker] * n_workers
    if diff > 0:
        workload[-1] = workload[-1] + diff
    return workload


@njit(nogil=True, cache=True)
def shuffle_arr(arr: np.ndarray) -> None:
    """In-place Fisher-Yates shuffle via numba's ``np.random.shuffle`` (the simple permutation source; ``shuffle_arr_lcg`` is the faster inline-LCG variant for hot loops)."""
    np.random.shuffle(arr)


@njit(nogil=True, cache=True)
def _shuffle_arr_lcg_kernel(arr: np.ndarray, state: np.uint64) -> np.uint64:
    """In-place Fisher-Yates shuffle of ``arr`` driven by an inline LCG; returns the advanced LCG state."""
    n = len(arr)
    for j in range(n - 1, 0, -1):
        state = state * np.uint64(6364136223846793005) + np.uint64(1442695040888963407)
        k = int(state >> np.uint64(33)) % (j + 1)
        tmp = arr[j]
        arr[j] = arr[k]
        arr[k] = tmp
    return state


def shuffle_arr_lcg(arr: np.ndarray, state: np.uint64) -> np.uint64:
    """Inline LCG Fisher-Yates -- same RNG pattern as
    ``parallel_mi_besag_clifford`` (LCG state-machine + bit-shifted high
    bits modulo j+1). Returns the post-shuffle state so the caller threads
    it across iterations.

    At n=200_000 the inline variant runs ~6x faster than the @njit
    ``np.random.shuffle`` wrapper above (0.6 ms vs 3.7 ms / call on the
    c0055 fuzz combo: 99 calls * 3.7 ms = 370 ms in cProfile, dropped to
    ~60 ms after the swap). Same statistical guarantee for the permutation
    test: nfailed/npermutations is an unbiased estimator of the p-value
    regardless of whether the shuffle source is numba's np.random or an
    inline LCG.

    Thin Python wrapper around ``_shuffle_arr_lcg_kernel`` (2026-07-10 fix): numba boxes a
    ``uint64`` return value as a plain Python ``int``, not an ``np.uint64`` scalar. Once the LCG
    state's high bit sets (roughly even odds per iteration once the stream looks random --
    typically within the first few calls), re-passing that bare ``int`` into another njit call made
    numba infer ``int64`` (a raw Python int carries no dtype), which fails to unbox with
    ``OverflowError: int too big to convert`` for any value >= 2**63. Confirmed via a minimal
    repro: the failure is at the numba dispatcher's argument-unboxing boundary (not inside the
    kernel body), single-threaded, with no other cause -- pre-existing, unrelated to this session's
    ``nogil=True`` addition (reproduces identically without it). Explicitly wrapping both the input
    and the return in ``np.uint64(...)`` at the PYTHON level (not inside numba) reconstructs a real
    ``np.uint64`` scalar object each time, which numba always unboxes correctly regardless of
    magnitude -- verified over 5000 chained calls with zero failures (pre-fix: fails within 1-5
    calls almost every seed).
    """
    return cast(np.uint64, np.uint64(_shuffle_arr_lcg_kernel(arr, np.uint64(state))))


@njit(cache=True)
def parallel_mi_besag_clifford(
    classes_x: np.ndarray,
    freqs_x: np.ndarray,
    classes_y: np.ndarray,
    freqs_y: np.ndarray,
    npermutations: int,
    original_mi: float,
    base_seed: np.uint64,
    p_low: float = 0.01,
    p_high: float = 0.05,
    min_perms: int = 30,
    dtype: type = np.int32,
    use_su: bool = False,  # 2026-05-28: SU normalization toggle threaded from mi_direct.
) -> tuple:
    """Besag-Clifford sequential permutation test with early stopping.

    Standard fixed-budget permutation test always runs ``npermutations`` shuffles. Many candidates are obviously significant (no permuted MI ever exceeds
    ``original_mi``) or obviously non-significant (``nfailed`` blows past ``max_failed`` immediately). Besag-Clifford (1991) computes a running confidence
    interval on the p-value after each permutation; once the CI falls entirely below ``p_low`` (clearly significant) or entirely above ``p_high`` (clearly
    null) we can stop without running the rest of the budget. Saves 5-10x permutations on average for typical mRMR workloads.

    Returns ``(nfailed, nchecked)`` -- same contract as ``parallel_mi``.

    References: Besag & Clifford (1991) "Sequential Monte Carlo p-values." Biometrika 78(2): 301-304.
    """
    if npermutations == 0:
        return 0, 0

    n = len(classes_y)
    nfailed = 0
    i = 0

    # Single LCG state -- this is sequential by nature (running CI test).
    state = np.uint64(base_seed) * np.uint64(2654435761) + np.uint64(1)
    local = classes_y.copy()

    for i in range(npermutations):
        # Fisher-Yates shuffle.
        for j in range(n - 1, 0, -1):
            state = state * np.uint64(6364136223846793005) + np.uint64(1442695040888963407)
            k = int(state >> np.uint64(33)) % (j + 1)
            tmp = local[j]
            local[j] = local[k]
            local[k] = tmp

        mi_perm = compute_relevance_score(
            use_su, classes_x, freqs_x, local, freqs_y, dtype=dtype,
        )
        if mi_perm >= original_mi:
            nfailed += 1

        # Early-stop check after at least min_perms permutations.
        n_done = i + 1
        if n_done >= min_perms:
            # 95% Wilson confidence interval on the failure rate (= p-value estimate). Stop if CI is entirely below p_low or entirely above p_high.
            phat = nfailed / n_done
            z = 1.96  # 95% normal quantile
            denom = 1.0 + z * z / n_done
            center = (phat + z * z / (2 * n_done)) / denom
            half = (z / denom) * np.sqrt(phat * (1 - phat) / n_done + z * z / (4 * n_done * n_done))
            ci_low = center - half
            ci_high = center + half
            if ci_high < p_low or ci_low > p_high:
                break

    return nfailed, i + 1


@njit(cache=True)
def parallel_mi_besag_clifford_with_null(
    classes_x: np.ndarray,
    freqs_x: np.ndarray,
    classes_y: np.ndarray,
    freqs_y: np.ndarray,
    npermutations: int,
    original_mi: float,
    base_seed: np.uint64,
    p_low: float = 0.01,
    p_high: float = 0.05,
    min_perms: int = 30,
    dtype: type = np.int32,
    use_su: bool = False,
) -> tuple:
    """Null-mean-accumulating twin of :func:`parallel_mi_besag_clifford`.

    Bit-identical ``(nfailed, nchecked)`` to the legacy kernel (same LCG stream, same early-stop logic) but ALSO accumulates the sum of the per-permutation MIs it
    already computes and returns ``(nfailed, nchecked, sum_perm_mi)``. The empirical permutation-null mean is ``sum_perm_mi / max(1, nchecked)`` -- it costs nothing
    extra because ``mi_perm`` is computed regardless; the legacy kernel simply discarded it. Kept as a sibling (not an in-place edit) so the 2-tuple contract every
    existing caller and test relies on stays untouched.
    """
    if npermutations == 0:
        return 0, 0, 0.0

    n = len(classes_y)
    nfailed = 0
    i = 0
    sum_perm_mi = 0.0

    state = np.uint64(base_seed) * np.uint64(2654435761) + np.uint64(1)
    local = classes_y.copy()

    for i in range(npermutations):
        for j in range(n - 1, 0, -1):
            state = state * np.uint64(6364136223846793005) + np.uint64(1442695040888963407)
            k = int(state >> np.uint64(33)) % (j + 1)
            tmp = local[j]
            local[j] = local[k]
            local[k] = tmp

        mi_perm = compute_relevance_score(
            use_su, classes_x, freqs_x, local, freqs_y, dtype=dtype,
        )
        sum_perm_mi += mi_perm
        if mi_perm >= original_mi:
            nfailed += 1

        n_done = i + 1
        if n_done >= min_perms:
            phat = nfailed / n_done
            z = 1.96
            denom = 1.0 + z * z / n_done
            center = (phat + z * z / (2 * n_done)) / denom
            half = (z / denom) * np.sqrt(phat * (1 - phat) / n_done + z * z / (4 * n_done * n_done))
            ci_low = center - half
            ci_high = center + half
            if ci_high < p_low or ci_low > p_high:
                break

    return nfailed, i + 1, sum_perm_mi


@njit(parallel=True, nogil=True, cache=True)
def parallel_mi_prange(
    classes_x: np.ndarray,
    freqs_x: np.ndarray,
    classes_y: np.ndarray,
    freqs_y: np.ndarray,
    npermutations: int,
    original_mi: float,
    base_seed: np.uint64,
    dtype: type = np.int32,
    use_su: bool = False,  # 2026-05-28: SU normalization toggle threaded from mi_direct.
) -> tuple:
    """Inner-loop parallel permutation test.

    Runs ``npermutations`` shuffles in a numba ``prange``. Each iteration owns a private ``classes_y`` copy and a private LCG seeded with
    ``base_seed * 2654435761 + i`` (Knuth's multiplicative hash). The seeding scheme is **independent of n_workers**, so the ``(nfailed, nchecked)`` output is
    bit-exact across ``n_workers in {1, 2, 4, 8}`` for the same ``base_seed`` (verified by ``test_phase1_reproducibility``).

    Differences from ``parallel_mi`` (joblib-process worker):
    * No early termination on ``nfailed >= max_failed`` -- every permutation in the budget runs because ``prange`` iterations are independent. For short budgets
      (npermutations < 30) the early-exit win was negligible anyway.
    * No global ``np.random.shuffle``; manual Fisher-Yates with a per-iteration LCG so the parallel race that legacy code hit under multi-thread numba is gone
      by construction.
    """
    if npermutations == 0:
        return 0, 0

    n = len(classes_y)
    nfailed_arr = np.zeros(npermutations, dtype=np.int64)

    for i in prange(npermutations):
        # Per-iteration LCG state. Knuth multiplicative hash + fold of i gives a deterministic, n_workers-independent stream.
        state = np.uint64(base_seed) * np.uint64(2654435761) + np.uint64(i + 1)

        local = classes_y.copy()
        # Fisher-Yates shuffle with the per-iter LCG.
        for j in range(n - 1, 0, -1):
            # PCG-like step.
            state = state * np.uint64(6364136223846793005) + np.uint64(1442695040888963407)
            k = int(state >> np.uint64(33)) % (j + 1)
            tmp = local[j]
            local[j] = local[k]
            local[k] = tmp

        mi_perm = compute_relevance_score(
            use_su, classes_x, freqs_x, local, freqs_y, dtype=dtype,
        )
        if mi_perm >= original_mi:
            nfailed_arr[i] = 1

    return int(nfailed_arr.sum()), npermutations


@njit(parallel=True, nogil=True, cache=True)
def parallel_mi_prange_with_null(
    classes_x: np.ndarray,
    freqs_x: np.ndarray,
    classes_y: np.ndarray,
    freqs_y: np.ndarray,
    npermutations: int,
    original_mi: float,
    base_seed: np.uint64,
    dtype: type = np.int32,
    use_su: bool = False,
    use_mm: bool = False,
    use_cs: bool = False,
) -> tuple:
    """Null-mean-accumulating twin of :func:`parallel_mi_prange`.

    ``use_mm``/``use_cs`` (critique N-F1, extended to Chao-Shen at ): the permutation null
    MUST use the SAME estimator as the observed relevance it is tested against. The observed MI is
    computed with Miller-Madow or Chao-Shen when the corresponding ``mi_correction`` is active, so each
    shuffle's MI is computed with the same correction too -- otherwise the exceedance test compares
    plug-in shuffles against a corrected observed value (over-rejection) AND
    ``observed_corrected - null_mean_plugin`` subtracts a mismatched bias (double correction). No-op
    when both are False (the default), so the plug-in path is unchanged.

    Bit-identical ``(nfailed, npermutations)`` to the legacy prange kernel (same per-iter LCG seeding, same exceedance count) but ALSO returns the sum of the
    per-permutation MIs as a third element ``(nfailed, npermutations, sum_perm_mi)``. The mean permutation-null MI is ``sum_perm_mi / npermutations``; subtracting it
    from the observed MI debiases the in-sample plug-in inflation that out-ranks genuine low-cardinality signal on a wide engineered pool. The per-iter ``mi_perm``
    is already computed by the legacy kernel and discarded; here it is accumulated into a private reduction array so the parallel reduction stays deterministic.
    """
    if npermutations == 0:
        return 0, 0, 0.0

    n = len(classes_y)
    nfailed_arr = np.zeros(npermutations, dtype=np.int64)
    mi_perm_arr = np.zeros(npermutations, dtype=np.float64)

    for i in prange(npermutations):
        state = np.uint64(base_seed) * np.uint64(2654435761) + np.uint64(i + 1)

        local = classes_y.copy()
        for j in range(n - 1, 0, -1):
            state = state * np.uint64(6364136223846793005) + np.uint64(1442695040888963407)
            k = int(state >> np.uint64(33)) % (j + 1)
            tmp = local[j]
            local[j] = local[k]
            local[k] = tmp

        mi_perm = compute_relevance_score(
            use_su, classes_x, freqs_x, local, freqs_y, dtype=dtype, use_mm=use_mm, use_cs=use_cs,
        )
        mi_perm_arr[i] = mi_perm
        if mi_perm >= original_mi:
            nfailed_arr[i] = 1

    return int(nfailed_arr.sum()), npermutations, float(mi_perm_arr.sum())


@njit(nogil=True, cache=True)
def parallel_mi(
    classes_x: np.ndarray,
    freqs_x: np.ndarray,
    classes_y: np.ndarray,
    freqs_y: np.ndarray,
    npermutations: int,
    original_mi: float,
    max_failed: int,
    dtype: type = np.int32,
    base_seed: np.uint64 = _DEFAULT_BASE_SEED,
    use_su: bool = False,  # 2026-05-28: SU normalization toggle threaded from mi_direct.
    perm_offset: int = 0,  # 2026-05-30 Wave 9.1 iter 18: cumulative permutation index offset for n_workers-independent seeding.
) -> tuple[int, int]:
    """Worker for the joblib pool used by ``mi_direct``. Returns ``(n_failed, n_checked)`` so the caller can aggregate across pool members. ``npermutations=0`` returns ``(0, 0)`` cleanly.

    ``base_seed`` threads a per-worker seed through the inline LCG Fisher-Yates so two parallel suite calls with the same seed produce identical ``(nfailed, _i+1)`` output. ``base_seed=0`` keeps the legacy stream
    deterministic across calls in the same process; parent callers (``mi_direct`` joblib branch) should derive per-worker seeds via Knuth multiplicative hash so the worker streams stay independent.

    2026-05-30 Wave 9.1 fix (loop iter 18): switched to per-permutation
    LCG seeding (``base_seed * 2654435761 + (perm_offset + i + 1)``)
    matching ``parallel_mi_prange``. Pre-fix the function advanced a
    single LCG across all iterations from ``base_seed``, so the random
    stream content was a function of the worker's perm count -
    aggregating across workers with different counts gave different
    ``(nfailed, n_checked)``. ``mi_direct(parallelism='outer',
    n_workers=1..8)`` was NOT bit-exact for the same ``base_seed``
    (confirmed 4 distinct confidence values across n_workers in {1,2,4,8}
    for the same data and seed). The new scheme is n_workers-independent
    by construction: worker w running perms [offset, offset+count)
    consumes the SAME seeds that a single-worker run would use at those
    same perm indices.
    """
    if npermutations == 0:
        return 0, 0

    nfailed = 0
    classes_y_safe = np.asarray(classes_y).copy()
    n = classes_y_safe.shape[0]
    _i = 0
    for _i in range(npermutations):
        # Per-iteration LCG state, index-keyed (matches parallel_mi_prange:176).
        # Bit-exact across n_workers because the seed for perm index k is
        # purely a function of base_seed and k - never of the partition.
        state = np.uint64(base_seed) * np.uint64(2654435761) + np.uint64(perm_offset + _i + 1)
        # Inline Fisher-Yates with the per-iter LCG.
        for j in range(n - 1, 0, -1):
            state = state * np.uint64(6364136223846793005) + np.uint64(1442695040888963407)
            k = int(state >> np.uint64(33)) % (j + 1)
            tmp = classes_y_safe[j]
            classes_y_safe[j] = classes_y_safe[k]
            classes_y_safe[k] = tmp
        mi = compute_relevance_score(
            use_su, classes_x, freqs_x, classes_y_safe, freqs_y, dtype=dtype,
        )
        if mi >= original_mi:
            nfailed += 1
            if nfailed >= max_failed:
                break
        # Reset classes_y_safe for the next iteration to mirror prange's
        # ``local = classes_y.copy()``. Mutating in place between perms
        # would compound permutations into the seed sequence and break
        # the per-iter independence.
        classes_y_safe = np.asarray(classes_y).copy()
    return nfailed, _i + 1


@njit(nogil=True, cache=True)
def parallel_mi_with_null(
    classes_x: np.ndarray,
    freqs_x: np.ndarray,
    classes_y: np.ndarray,
    freqs_y: np.ndarray,
    npermutations: int,
    original_mi: float,
    max_failed: int,
    dtype: type = np.int32,
    base_seed: np.uint64 = _DEFAULT_BASE_SEED,
    use_su: bool = False,
    perm_offset: int = 0,
) -> tuple:
    """Null-mean-accumulating twin of :func:`parallel_mi` (joblib worker).

    Returns ``(nfailed, nchecked, sum_perm_mi)``; the first two are bit-identical to the legacy worker for the same ``(base_seed, perm_offset, npermutations)``. Note the
    ``max_failed`` early break is preserved, so on a clearly-non-significant candidate the worker stops early and ``nchecked < npermutations`` -- the null mean is then the
    average over the perms actually run (``sum_perm_mi / nchecked``), which is the correct empirical estimate for the perms drawn. Used only by the ``return_null_mean`` path,
    which always runs the full budget (``max_failed`` set to a no-trip sentinel) so the null mean is over all ``npermutations`` shuffles.
    """
    if npermutations == 0:
        return 0, 0, 0.0

    nfailed = 0
    sum_perm_mi = 0.0
    classes_y_safe = np.asarray(classes_y).copy()
    n = classes_y_safe.shape[0]
    _i = 0
    for _i in range(npermutations):
        state = np.uint64(base_seed) * np.uint64(2654435761) + np.uint64(perm_offset + _i + 1)
        for j in range(n - 1, 0, -1):
            state = state * np.uint64(6364136223846793005) + np.uint64(1442695040888963407)
            k = int(state >> np.uint64(33)) % (j + 1)
            tmp = classes_y_safe[j]
            classes_y_safe[j] = classes_y_safe[k]
            classes_y_safe[k] = tmp
        mi = compute_relevance_score(
            use_su, classes_x, freqs_x, classes_y_safe, freqs_y, dtype=dtype,
        )
        sum_perm_mi += mi
        if mi >= original_mi:
            nfailed += 1
            if nfailed >= max_failed:
                break
        classes_y_safe = np.asarray(classes_y).copy()
    return nfailed, _i + 1, sum_perm_mi


def mi_direct(
    factors_data: np.ndarray,
    x: tuple,
    y: tuple,
    factors_nbins: np.ndarray,
    min_occupancy: Optional[int] = None,
    dtype: type = np.int32,
    npermutations: int = 10,
    max_failed: Optional[int] = None,
    min_nonzero_confidence: float = 0.95,
    classes_y: Optional[np.ndarray] = None,
    classes_y_safe: Optional[np.ndarray] = None,
    freqs_y: Optional[np.ndarray] = None,
    n_workers: int = 1,
    workers_pool: Optional[Parallel] = None,
    parallel_kwargs: Optional[dict] = None,
    parallelism: str = "outer",
    base_seed: int = 0,
    prefer_gpu: bool = True,
    return_null_mean: bool = False,
) -> tuple:
    """CPU mutual-information + permutation-test wrapper.

    ``return_null_mean`` (default ``False``, fully backward-compatible): when ``True`` the call returns a 4-tuple ``(original_mi, confidence, null_mean, p_value)`` instead of
    the 2-tuple. ``null_mean`` is the EMPIRICAL permutation-null mean MI -- the average of the ``mi_perm`` values the permutation kernels already compute -- and ``p_value`` is
    the permutation p-value ``nfailed / nchecked`` (the fraction of shuffles whose MI tied or beat the observed MI). The MRMR screen uses BOTH for SIGNIFICANCE-GATED debiasing:
    a feature that is permutation-SIGNIFICANT (``p_value < alpha``) keeps its full observed MI (protecting weak-but-real signal that sits above its own null), while a feature
    that is NOT significant (``p_value >= alpha``) has the null mean subtracted (``relevance = max(0, observed - null_mean)``), demoting spurious high-cardinality / heavy-tailed
    / monotone columns toward zero. The p-value is the correct discriminator the null mean alone cannot provide: weak genuine signal and pure noise can BOTH carry a high null
    mean (coarse binning inflates the plug-in null), but only noise sits WITHIN its null distribution. Default ``False`` so every existing caller is unaffected.

    ``parallelism`` modes:
    * ``"outer"`` (default): RETIRED as a joblib pool (2026-07-19) -- always dispatches to the same ``parallel_mi_prange`` njit kernel as ``"inner"``,
      regardless of ``n_workers``. The joblib.Parallel(backend="threading") pool this mode used to build is strictly dominated by ``parallel_mi_prange``
      at every scale measured (see ``_benchmarks/bench_permutation_njit_prange_vs_joblib.py``: n=100000/npermutations=500 -> prange 4.57x over serial vs
      joblib 1.6-1.7x; n=20000/npermutations=500 -> prange 7.26x vs joblib 3.3-4.1x), so the pool code is kept only as an inline comment for reference and
      is never executed. ``n_workers``/``workers_pool``/``parallel_kwargs`` are accepted for backward compatibility but no longer change dispatch.
    * ``"inner"``: numba ``prange`` over permutations inside a single thread pool, per-iteration LCG seed. Same ``(base_seed, npermutations)`` plus any
      ``n_workers`` value yields identical ``(nfailed, nchecked)``. As of 2026-07-19 this is identical in effect to ``"outer"``; the two names are kept
      distinct for caller-site clarity, not because they dispatch differently.
    * ``"bc"``: Besag-Clifford sequential permutation test with adaptive early stopping. Sequential but typically 5-10x fewer permutations than fixed budget.
    * ``"none"``: sequential, no parallelism (used by golden tests)."""
    global _MI_DIRECT_GPU_FAILED
    if parallel_kwargs is None:
        parallel_kwargs = {}

    # ---- Analytic large-n null (2026-06-16) -------------------------------------------------------
    # The permutation null -- the CPU prange shuffles AND the GPU cupy-argsort branch just below --
    # is the dominant large-n cost (bench_scaling: the GPU argsort permutation generator was 72% of a
    # 400k fit). At large n the plug-in MI null is analytic, so the shuffles are unnecessary:
    #   null_mean = (Bx-1)(By-1)/(2N)  [nats, Miller-Madow]    p = chi2.sf(2N*MI, (Bx-1)(By-1)) [G-test].
    # Validated against the permutation kernel (npermutations=64) across n in {5k..200k}: the null mean
    # matches to 3+ digits even at 5k and the p reproduces the significance decision. Engaged ONLY when
    # MI is raw (NOT SU-normalised -- the 2N*MI~chi2 identity requires it) AND n >= a threshold; below
    # it the legacy permutation path runs byte-for-byte unchanged. Covers BOTH the 2-tuple confidence
    # and 4-tuple null-mean contracts, so it also bypasses the GPU permutation branch below.
    try:
        from ._analytic_mi_null import (
            analytic_mi_null, analytic_null_enabled, analytic_null_min_n, analytic_null_applicable,
        )

        # SCREEN_CONFIRM_B-2 fix: the analytic-null fast path returned RAW plug-in
        # MI unconditionally (compute_relevance_score(False, ...), no use_mm/use_cs) while the sub-threshold
        # permutation path below always applies the active mi_correction -- a silent formula discontinuity
        # in the reported relevance value exactly at the analytic_null_min_n() threshold whenever
        # mi_correction='miller_madow'/'chao_shen' is active. Fall through to the exact permutation path
        # (unchanged) when either correction is active, matching the existing SU/sparsity fall-through pattern.
        _analytic_ok = (
            analytic_null_enabled()
            and int(factors_data.shape[0]) >= analytic_null_min_n()
            and not use_su_normalization()
            and not use_mi_miller_madow()
            and not use_mi_chao_shen()
        )
    except Exception:
        _analytic_ok = False
    if _analytic_ok:
        if classes_y is None:
            classes_y, freqs_y, _ = merge_vars(
                factors_data=factors_data, vars_indices=y,
                var_is_nominal=None, factors_nbins=factors_nbins, dtype=dtype,
            )
        assert freqs_y is not None  # the None branch above always sets classes_y/freqs_y together
        _n_rows = int(factors_data.shape[0])
        _by = int(freqs_y.shape[0])
        # FUSED single-var fast path (wasted-per-call-work audit, 2026-07-05): the analytic branch
        # needs only the MI scalar + the occupied-x-bin count -- it DISCARDS the length-n classes_x
        # that merge_vars(x) builds. For the single-variable relevance x (every MRMR/FE caller passes
        # x=(var,) / x=[0]), ``_relevance_mi_1var_fused`` builds the joint histogram + MI + bx in ONE
        # O(n) pass, skipping the separate merge_vars(x) accumulate (+ remap) pass entirely.
        # Bit-identical to the legacy merge_vars + compute_relevance_score(False, ...) below (proven
        # in test_mi_direct_fused_1var_relevance). Mirrors the joint_freqs_2var DCD win.
        if len(x) == 1:
            _ix = int(x[0])
            _nb_x = int(factors_nbins[_ix])
            _ax_mi, _ax_bx = _relevance_mi_1var_fused(factors_data, _ix, _nb_x, classes_y, freqs_y)
            if analytic_null_applicable(_n_rows, _ax_bx, _by):
                _ax_nm, _ax_p = analytic_mi_null(_ax_mi, _n_rows, _ax_bx, _by)
                if return_null_mean:
                    return _ax_mi, 1.0 - _ax_p, _ax_nm, _ax_p
                return _ax_mi, 1.0 - _ax_p
            # Sparse cells -> analytic null unreliable; FALL THROUGH to the permutation path below
            # (which rebuilds merge_vars(x); rare high-cardinality case, not the dense hot path).
        else:
            _ax_classes, _ax_freqs, _ = merge_vars(
                factors_data=factors_data, vars_indices=x,
                var_is_nominal=None, factors_nbins=factors_nbins, dtype=dtype,
            )
            # Occupancy safe-condition: the chi-square / Miller-Madow approximation is only trustworthy
            # when the contingency cells are not sparse (a high-cardinality x can have sparse cells even at
            # large n). When sparse, FALL THROUGH to the sparsity-correct permutation path below.
            if analytic_null_applicable(int(_ax_classes.shape[0]), int(_ax_freqs.shape[0]), _by):
                _ax_mi = compute_relevance_score(False, _ax_classes, _ax_freqs, classes_y, freqs_y, dtype=dtype)
                _ax_nm, _ax_p = analytic_mi_null(
                    _ax_mi, int(_ax_classes.shape[0]), int(_ax_freqs.shape[0]), _by,
                )
                if return_null_mean:
                    return _ax_mi, 1.0 - _ax_p, _ax_nm, _ax_p
                return _ax_mi, 1.0 - _ax_p

    # Transparent route to the GPU permutation path when CUDA is available AND
    # the caller is asking for a high enough permutation count to amortise the
    # H2D copy of (classes_x, classes_y). Profile on 1M x 30 features (commit
    # 033941b's profile_mrmr_layer3_1m bench) showed CPU ``shuffle_arr``
    # consuming 51% of MRMR.fit wall at fe_npermutations=10 and 84% at 50.
    # Routing to ``mi_direct_gpu`` further delegates to ``mi_direct_gpu_batched``
    # at npermutations>=32 (commit 033941b) which batches permutations 64 at a
    # time. Same ``(original_mi, confidence)`` contract.
    #
    # Gating notes:
    #   * ``parallelism in ("outer", "none")`` -- the inner/besag-clifford
    #     paths have their own numba-prange or sequential contracts that
    #     don't compose with GPU dispatch.
    #   * ``classes_y_safe`` is intentionally NOT a gate condition. The
    #     CPU ``classes_y_safe`` arg is a numpy buffer the caller pre-
    #     allocated for the prange permutation kernel; ``mi_direct_gpu``
    #     uses an independent CuPy buffer pool (``_GPU_POOL``) and does
    #     not consume the CPU pre-warm. Passing ``classes_y_safe=None``
    #     to ``mi_direct_gpu`` lets it allocate / reuse its own GPU
    #     buffers.
    # ``return_null_mean=True`` is forced onto the CPU permutation kernels (the only ones that accumulate the per-permutation MI sum). The GPU permutation path does not
    # surface the empirical null, and the screen's relevance-baseline budget (``npermutations<32``) never reaches the GPU branch anyway, so routing-to-CPU here costs nothing
    # in practice; it only guards the unusual case of a caller asking for the null mean with a large budget on a CUDA host.
    if prefer_gpu and npermutations >= 32 and parallelism in ("outer", "none") and not return_null_mean and not _MI_DIRECT_GPU_FAILED:
        # SCREEN_CONFIRM_B-4 fix: this internal fastpath gate never consulted
        # gpu_globally_disabled()/MLFRAME_DISABLE_GPU -- only is_cuda_available() (honors
        # CUDA_VISIBLE_DEVICES but not the mlframe-specific opt-out). A caller that decided NOT to use
        # GPU (e.g. confirm_candidate's _confirm_use_gpu=False fallback, which calls plain mi_direct()
        # with no prefer_gpu=False) could have mi_direct silently re-route to GPU anyway once
        # npermutations>=32, overriding the caller's explicit opt-out.
        try:
            from pyutilz.core.pythonlib import is_cuda_available
            from ._gpu_policy import gpu_globally_disabled
            _gpu_ok = (not gpu_globally_disabled()) and is_cuda_available()
        except Exception:
            _gpu_ok = False
        if _gpu_ok:
            # Proactive VRAM headroom check: before this,
            # ``mi_direct``'s GPU fastpath had no upfront capacity guard -- unlike the CMI path
            # (``_cmi_cuda._should_use_cuda``), which already probes ``memGetInfo`` + calls this SAME
            # ``fe_gpu_has_vram_cushion`` helper before launching. The dominant device buffer here is the
            # batched-permutation y-matrix (``max(npermutations, 64)`` int32 rows of length ``n``, per the
            # ``mi_direct_gpu_batched`` delegation above); a near-full card (another process sharing the
            # GPU, or VRAM eaten by a prior stage) would otherwise only be caught reactively by the
            # circuit breaker AFTER an actual launch fault. Permissive on probe failure/no-cupy (see the
            # helper's own docstring), so this can only DECLINE an already-risky launch, never block a
            # healthy one.
            try:
                _n = int(np.asarray(x).shape[0])
                _bytes_needed = _n * max(int(npermutations), 64) * 4 + _n * 8
                from mlframe.feature_selection.filters._fe_gpu_vram import fe_gpu_has_vram_cushion
                _gpu_ok = fe_gpu_has_vram_cushion(_bytes_needed)
            except Exception:
                _gpu_ok = True  # probe failure must not block an otherwise-eligible launch
        if _gpu_ok:
            try:
                from mlframe.feature_selection.filters.gpu import mi_direct_gpu
                return mi_direct_gpu(
                    factors_data=factors_data,
                    x=x,
                    y=y,
                    factors_nbins=factors_nbins,
                    min_occupancy=min_occupancy,
                    dtype=dtype,
                    npermutations=npermutations,
                    max_failed=max_failed,
                    min_nonzero_confidence=min_nonzero_confidence,
                    classes_y=classes_y,
                    freqs_y=freqs_y,
                    use_gpu=True,
                )
            except Exception as _exc:
                # Promoted DEBUG -> WARNING + trips the circuit breaker (2026-07-09 fix): a launch fault poisons
                # the CUDA context (see the breaker's module-level docstring), so silently retrying at DEBUG on
                # every future call was both invisible in a normal INFO-level log AND wasted a repeated failed
                # launch. First fault routes every subsequent mi_direct call straight to CPU for the rest of the
                # process; only the FIRST fault is logged at WARNING to avoid spamming on every later call.
                _MI_DIRECT_GPU_FAILED = True
                logger.warning(
                    "mi_direct: GPU fastpath failed (%s: %s); falling back to CPU for the rest of the process "
                    "(circuit breaker tripped; call reset_mi_direct_gpu_circuit_breaker() to re-arm).",
                    type(_exc).__name__, _exc,
                )

    # 2026-05-28: read SU toggle once per call; threaded into every njit branch
    # below so cardinality-bias correction stays consistent across the permutation
    # test (original score AND permuted scores both use the same scorer).
    _use_su = use_su_normalization()

    classes_x, freqs_x, _ = merge_vars(
        factors_data=factors_data, vars_indices=x,
        var_is_nominal=None, factors_nbins=factors_nbins, dtype=dtype,
    )
    if classes_y is None:
        classes_y, freqs_y, _ = merge_vars(
            factors_data=factors_data, vars_indices=y,
            var_is_nominal=None, factors_nbins=factors_nbins, dtype=dtype,
        )

    original_mi = compute_relevance_score(
        _use_su, classes_x, freqs_x, classes_y, freqs_y, dtype=dtype,
        use_mm=(use_mi_miller_madow() and not _use_su),
        use_cs=(use_mi_chao_shen() and not _use_su),
    )

    if return_null_mean:
        # Dedicated empirical-null sub-path, isolated from the legacy branch tree so it can never perturb existing ``return_null_mean=False`` behaviour. The relevance debiasing
        # needs a STABLE null mean AND a usable permutation p-value, so we run a larger null budget (``_NULL_MEAN_MIN_PERMS``, default 32) than the screen's tiny exceedance budget
        # (``npermutations``, default 2), then average the per-permutation MIs the kernel already computes and count the exceedances. The exceedance/rejection contract is preserved
        # on a RATE basis: a candidate is rejected (``original_mi -> 0``) only when the null-failure RATE meets the caller's ``1 - min_nonzero_confidence`` floor -- equivalent to the
        # legacy budget-absolute ``nfailed >= max_failed`` test at ``n_checked == npermutations``, but correct when the null budget is larger. The screen calls with
        # ``min_nonzero_confidence=0.0`` so the rate floor is 1.0 (reject only if EVERY shuffle ties/beats observed -- pure noise), keeping the Wave-9.1 unanimous-rejection semantics
        # intact. ``p_value = nfailed / n_checked`` is surfaced so the screen can SIGNIFICANCE-GATE the null-mean subtraction (subtract only when the feature is NOT significant).
        null_mean = 0.0
        confidence = 0.0
        p_value = 1.0  # default for the original_mi == 0 / no-perm path: an uninformative feature is maximally non-significant.
        if original_mi > 0 and npermutations > 0:
            _null_nperms = max(int(npermutations), _NULL_MEAN_MIN_PERMS)
            _cy = classes_y_safe if classes_y_safe is not None else classes_y
            nfailed, n_checked, sum_perm_mi = parallel_mi_prange_with_null(
                classes_x=classes_x,
                freqs_x=freqs_x,
                classes_y=_cy,
                freqs_y=freqs_y,
                npermutations=_null_nperms,
                original_mi=original_mi,
                base_seed=np.uint64(base_seed),
                dtype=dtype,
                use_su=_use_su,
                use_mm=(use_mi_miller_madow() and not _use_su),  # N-F1: null uses the SAME estimator as original_mi
                use_cs=(use_mi_chao_shen() and not _use_su),
            )
            if n_checked > 0:
                null_mean = sum_perm_mi / float(n_checked)
                # The rejection gate uses the RAW exceedance rate (legacy unanimous-rejection semantics on the screen's tiny budget), while the surfaced p_value / confidence use
                # the add-one Monte-Carlo estimator -- the gate decision is unchanged, only the reported confidence is calibrated. ``_null_nperms`` is the full budget for P2.
                _raw_rate = nfailed / float(n_checked)
                p_value = _perm_pvalue(nfailed, n_checked, full_budget=_null_nperms)
                confidence = 1.0 - p_value
                _rate_floor = float(1.0 - float(min_nonzero_confidence))
                if _raw_rate >= _rate_floor:
                    original_mi = 0.0
        return original_mi, confidence, null_mean, p_value

    confidence = 0.0
    i = -1  # caller-side guard: if no inner branch runs, n_total stays 0.
    nfailed = 0

    if original_mi > 0 and npermutations > 0:
        if not max_failed:
            max_failed = int(npermutations * (1 - min_nonzero_confidence))
            if max_failed <= 1:
                max_failed = 1

        if parallelism == "bc" and npermutations > NMAX_NONPARALLEL_ITERS:
            # Adaptive Besag-Clifford early-stopping permutation test. Sequential, single-LCG, typically 5-10x fewer permutations than fixed-budget paths.
            nfailed, n_checked = parallel_mi_besag_clifford(
                classes_x=classes_x,
                freqs_x=freqs_x,
                classes_y=classes_y_safe if classes_y_safe is not None else classes_y,
                freqs_y=freqs_y,
                npermutations=npermutations,
                original_mi=original_mi,
                base_seed=np.uint64(base_seed),
                dtype=dtype,
                use_su=_use_su,
            )
            i = n_checked - 1
            # 2026-05-30 Wave 9.1 fix (loop iter 10): use RATE-based check
            # for the BC early-stop path. BC may exit at small ``n_checked``
            # (as low as ``min_perms=30``) when its Wilson CI on the failure
            # rate proves the null hypothesis. In that case ``nfailed`` is
            # also small and the budget-absolute ``nfailed >= max_failed``
            # comparison (where ``max_failed = npermutations * (1 - mnc)``)
            # silently FAILS to reject candidates whose actual p-value sits
            # above the caller's ``1 - min_nonzero_confidence`` threshold.
            # Confirmed live (seed=8, n=600): BC accepted ``mi=0.0049`` at
            # ``conf=0.853`` (nfailed_rate=0.147 >> 1-0.99=0.01) while
            # parallelism='outer' correctly rejected on the same data.
            # The rate-based form is equivalent to the budget-absolute form
            # when ``n_checked == npermutations`` (the outer/inner/workers
            # paths below), so this fix is BC-specific.
            _rate_floor = float(1.0 - float(min_nonzero_confidence))
            if nfailed >= max_failed or (n_checked > 0 and (nfailed / float(n_checked)) >= _rate_floor):
                original_mi = 0.0
        elif parallelism == "inner" and npermutations > NMAX_NONPARALLEL_ITERS:
            # Inner parallelism via numba prange. Single function call returns (nfailed, npermutations) -- matches outer-pool aggregation contract.
            nfailed, n_checked = parallel_mi_prange(
                classes_x=classes_x,
                freqs_x=freqs_x,
                classes_y=classes_y_safe if classes_y_safe is not None else classes_y,
                freqs_y=freqs_y,
                npermutations=npermutations,
                original_mi=original_mi,
                base_seed=np.uint64(base_seed),
                dtype=dtype,
                use_su=_use_su,
            )
            i = n_checked - 1
            if nfailed >= max_failed:
                original_mi = 0.0
        # RETIRED (2026-07-19): the "outer" joblib.Parallel(backend="threading") pool branch used to live here,
        # gated on ``n_workers>1 and npermutations>NMAX_NONPARALLEL_ITERS`` (later tightened to
        # ``npermutations>=_OUTER_PARALLEL_MIN_PERMUTATIONS=64`` after an isolated A/B showed it losing below
        # that budget: 0.36-0.40x at npermutations=10, n=99401). A follow-up head-to-head bench
        # (``_benchmarks/bench_permutation_njit_prange_vs_joblib.py``) found the ALREADY-EXISTING
        # ``parallel_mi_prange`` njit(parallel=True) kernel (below, in the ``else`` branch) STRICTLY DOMINATES
        # this joblib pool at every scale tested, including where the pool used to win over serial: n=100000/
        # npermutations=500 -> prange 4.57x over serial vs joblib 1.6-1.7x; n=20000/npermutations=500 -> prange
        # 7.26x vs joblib 3.3-4.1x. Both kernels share the same per-iteration LCG seeding scheme, so their
        # ``(nfailed, n_checked)`` output is bit-identical (verified in the bench and in
        # ``test_mi_direct_outer_parallelism_always_routes_to_prange``). The branch below is kept verbatim
        # (never executes -- the ``elif`` condition that used to guard it has been removed, so control always
        # falls through to the unified ``else`` below) purely as a historical reference for the joblib-based
        # dispatch shape; do not re-enable it without a fresh A/B on current hardware.
        #
        #     if workers_pool is None:
        #         workers_pool = Parallel(n_jobs=n_workers, **parallel_kwargs)
        #     _worker_loads = distribute_permutations(npermutations=npermutations, n_workers=n_workers)
        #     _classes_y_for_workers = classes_y_safe if classes_y_safe is not None else classes_y
        #     _cumulative_offset = 0
        #     _delayed_calls = []
        #     for _widx, worker_npermutations in enumerate(_worker_loads):
        #         _delayed_calls.append(
        #             delayed(parallel_mi)(
        #                 classes_x=classes_x, freqs_x=freqs_x, classes_y=_classes_y_for_workers, freqs_y=freqs_y,
        #                 dtype=dtype, npermutations=worker_npermutations, original_mi=original_mi,
        #                 max_failed=max_failed, base_seed=np.uint64(base_seed), use_su=_use_su,
        #                 perm_offset=_cumulative_offset,
        #             )
        #         )
        #         _cumulative_offset += int(worker_npermutations)
        #     res = workers_pool(_delayed_calls)
        #     n_checked = 0
        #     for worker_nfailed, worker_i in res:
        #         nfailed += worker_nfailed
        #         n_checked += worker_i
        #     i = n_checked - 1
        #     if nfailed >= max_failed:
        #         original_mi = 0.0
        else:
            # 2026-05-30 iter573: route to ``parallel_mi_prange`` (the
            # njit @njit(parallel=True) kernel) for the n_workers<=1 /
            # parallelism="outer" fallback. The kernel was previously
            # gated by ``parallelism == "inner" AND npermutations >
            # NMAX_NONPARALLEL_ITERS=2`` -- so a default-parameter call
            # (n_workers=1, parallelism="outer", npermutations=10) fell
            # to the hand-rolled pure-Python LCG below it instead, which
            # ran 5 million Python loop iterations per call at n=500k
            # (3158 ms / call on the c0021_1508f146 @500k profile bench;
            # 65.3 s tottime / 156 calls = 420 ms / call on real-data
            # MRMR.fit). ``parallel_mi_prange`` implements the SAME
            # per-iter LCG (Knuth multiplicative hash + PCG step) so
            # the (nfailed, n_checked) output is bit-equivalent to what
            # the legacy Python loop produced -- the Wave 9.1 iter-18
            # fix (PRESERVED in this routing) aligned the seeding
            # schemes for exactly this reason. Below the kernel call
            # the early-stop ``if nfailed >= max_failed: original_mi = 0``
            # branch is preserved because ``parallel_mi_prange`` runs
            # the full budget (no early exit -- prange iterations are
            # independent), matching the inner-parallel contract at
            # line 412-427.
            nfailed, n_checked = parallel_mi_prange(
                classes_x=classes_x,
                freqs_x=freqs_x,
                classes_y=classes_y_safe if classes_y_safe is not None else classes_y,
                freqs_y=freqs_y,
                npermutations=npermutations,
                original_mi=original_mi,
                base_seed=np.uint64(base_seed),
                dtype=dtype,
                use_su=_use_su,
            )
            i = n_checked - 1
            if nfailed >= max_failed:
                original_mi = 0.0

        # Caller-side npermutations==0 guard. P2: the BC and outer-worker paths can early-break with ``n_checked = i + 1 < npermutations``; pass the full budget so the
        # reported confidence is full-budget-consistent (does not depend on where the break fired), and apply the add-one Monte-Carlo p-value estimator (P1).
        if i >= 0:
            confidence = 1.0 - _perm_pvalue(nfailed, i + 1, full_budget=npermutations)

    return original_mi, confidence
