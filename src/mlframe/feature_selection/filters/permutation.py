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

import numpy as np
from joblib import Parallel, delayed
from numba import njit, prange

from ._internals import NMAX_NONPARALLEL_ITERS
from .info_theory import (
    compute_mi_from_classes, merge_vars, mi_or_su_from_classes,
    # 2026-05-28: njit-callable dispatcher used inside permutation kernels;
    # branches on a bool param so the SU mode propagates without re-reading
    # the Python-level thread-local from inside @njit.
    compute_relevance_score, use_su_normalization,
)

logger = logging.getLogger(__name__)

# Number of y-permutations used by ``mi_direct(return_null_mean=True)`` to estimate BOTH the empirical relevance null mean AND the permutation p-value that gates the
# significance-aware debiasing in ``evaluate_candidate``. The MRMR screen's exceedance budget (``baseline_npermutations``, default 2) is far too few shuffles for either
# purpose -- 2 samples give a null-mean estimate with ~70% relative noise and a p-value that resolves only to {0, 0.5, 1.0}. 32 serves two needs: (a) it brings the null-mean
# standard error down ~4x, and (b) it makes the p-value resolve to 1/32 ~ 0.031, FINE ENOUGH for a textbook alpha=0.05 significance cut to cleanly separate weak-but-real
# signal (sits above its null, p ~ 0) from spurious noise (sits within its null, p large) -- at 16 perms the p-resolution (1/16 ~ 0.0625) is coarser than alpha and the gate
# would have to demand ZERO exceedances, which is too strict for a genuinely-weak leg whose null occasionally ties it. 32 of these per candidate is still microseconds on the
# screening hot path (each is one ``compute_relevance_score`` call). Tunable via the ``MLFRAME_MRMR_NULL_PERMS`` env var for users who want a tighter/looser null estimate.
import os as _os
_NULL_MEAN_MIN_PERMS = max(2, int(_os.environ.get("MLFRAME_MRMR_NULL_PERMS", "32")))


@njit(cache=True)
def distribute_permutations(npermutations: int, n_workers: int) -> list:
    """Split ``npermutations`` across ``n_workers``; the remainder lands on the last worker."""
    avg_perms_per_worker = npermutations // n_workers
    diff = npermutations - avg_perms_per_worker * n_workers
    workload = [avg_perms_per_worker] * n_workers
    if diff > 0:
        workload[-1] = workload[-1] + diff
    return workload


@njit(cache=True)
def shuffle_arr(arr: np.ndarray) -> None:
    """In-place Fisher-Yates shuffle via numba's ``np.random.shuffle`` (the simple permutation source; ``shuffle_arr_lcg`` is the faster inline-LCG variant for hot loops)."""
    np.random.shuffle(arr)


@njit(cache=True)
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
    """
    n = len(arr)
    for j in range(n - 1, 0, -1):
        state = state * np.uint64(6364136223846793005) + np.uint64(1442695040888963407)
        k = int(state >> np.uint64(33)) % (j + 1)
        tmp = arr[j]
        arr[j] = arr[k]
        arr[k] = tmp
    return state


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
) -> tuple:
    """Null-mean-accumulating twin of :func:`parallel_mi_prange`.

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
            use_su, classes_x, freqs_x, local, freqs_y, dtype=dtype,
        )
        mi_perm_arr[i] = mi_perm
        if mi_perm >= original_mi:
            nfailed_arr[i] = 1

    return int(nfailed_arr.sum()), npermutations, float(mi_perm_arr.sum())


@njit(cache=True)
def parallel_mi(
    classes_x: np.ndarray,
    freqs_x: np.ndarray,
    classes_y: np.ndarray,
    freqs_y: np.ndarray,
    npermutations: int,
    original_mi: float,
    max_failed: int,
    dtype: type = np.int32,
    base_seed: np.uint64 = np.uint64(0),
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


@njit(cache=True)
def parallel_mi_with_null(
    classes_x: np.ndarray,
    freqs_x: np.ndarray,
    classes_y: np.ndarray,
    freqs_y: np.ndarray,
    npermutations: int,
    original_mi: float,
    max_failed: int,
    dtype: type = np.int32,
    base_seed: np.uint64 = np.uint64(0),
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
    min_occupancy: int = None,
    dtype: type = np.int32,
    npermutations: int = 10,
    max_failed: int = None,
    min_nonzero_confidence: float = 0.95,
    classes_y: np.ndarray = None,
    classes_y_safe: np.ndarray = None,
    freqs_y: np.ndarray = None,
    n_workers: int = 1,
    workers_pool: object = None,
    parallel_kwargs: dict = None,
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
    * ``"outer"`` (default): joblib-process pool runs full ``parallel_mi`` workers.
    * ``"inner"``: numba ``prange`` over permutations inside a single thread pool, per-iteration LCG seed. Same ``(base_seed, npermutations)`` plus any
      ``n_workers`` value yields identical ``(nfailed, nchecked)``.
    * ``"bc"``: Besag-Clifford sequential permutation test with adaptive early stopping. Sequential but typically 5-10x fewer permutations than fixed budget.
    * ``"none"``: sequential, no parallelism (used by golden tests).

    Outer parallelism is preferred when ``len(candidates) >> n_workers`` (the orchestrator already amortises pool spawn cost). Inner is preferred when only a
    single candidate is being evaluated with a large permutation budget."""
    if parallel_kwargs is None:
        parallel_kwargs = {}

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
    if (
        prefer_gpu
        and npermutations >= 32
        and parallelism in ("outer", "none")
        and not return_null_mean
    ):
        try:
            from pyutilz.core.pythonlib import is_cuda_available
            _gpu_ok = is_cuda_available()
        except Exception:
            _gpu_ok = False
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
                logger.debug(
                    "mi_direct: GPU fastpath failed (%s: %s); falling back to CPU",
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
            )
            if n_checked > 0:
                null_mean = sum_perm_mi / float(n_checked)
                p_value = nfailed / float(n_checked)
                confidence = 1.0 - p_value
                _rate_floor = float(1.0 - float(min_nonzero_confidence))
                if p_value >= _rate_floor:
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
            if (nfailed >= max_failed
                    or (n_checked > 0 and (nfailed / float(n_checked)) >= _rate_floor)):
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
        elif n_workers and n_workers > 1 and npermutations > NMAX_NONPARALLEL_ITERS:
            if workers_pool is None:
                workers_pool = Parallel(n_jobs=n_workers, **parallel_kwargs)

            # Per-worker base_seed derived from outer base_seed via Knuth multiplicative hash + worker index so worker streams stay independent yet reproducible (same outer base_seed -> identical aggregate).
            _worker_loads = distribute_permutations(npermutations=npermutations, n_workers=n_workers)
            # 2026-05-30 Wave 9.1 fix (loop iter 16): fall back to
            # ``classes_y`` when ``classes_y_safe`` is None. The bc branch
            # (line 355) and inner branch (line 387) already do this; the
            # outer branch was the asymmetric one. Pre-fix
            # ``mi_direct(parallelism='outer', n_workers>1,
            # classes_y_safe=None, npermutations>NMAX_NONPARALLEL_ITERS)``
            # crashed inside ``parallel_mi`` with
            # ``TypingError: No implementation of function asarray(none)``.
            # Each worker ``parallel_mi`` already ``.copy()``s the array
            # internally so workers don't race on the shared ``classes_y``.
            _classes_y_for_workers = (
                classes_y_safe if classes_y_safe is not None else classes_y
            )
            # 2026-05-30 Wave 9.1 fix (loop iter 18): use the SAME
            # ``base_seed`` for every worker and pass each worker its
            # cumulative ``perm_offset`` so the perm-index seeding in
            # ``parallel_mi`` matches the single-worker run bit-for-bit
            # regardless of ``n_workers``. Pre-fix each worker got a
            # DIFFERENT base_seed derivation
            # (``base_seed * 2654435761 + widx + 1``) so the random
            # stream content depended on n_workers - confidence varied
            # by 70% across n_workers in {1, 2, 4, 8} for the same
            # base_seed, breaking the "same seed -> identical output"
            # contract for any consumer that switched ``n_jobs``.
            _cumulative_offset = 0
            _delayed_calls = []
            for _widx, worker_npermutations in enumerate(_worker_loads):
                _delayed_calls.append(
                    delayed(parallel_mi)(
                        classes_x=classes_x,
                        freqs_x=freqs_x,
                        classes_y=_classes_y_for_workers,
                        freqs_y=freqs_y,
                        dtype=dtype,
                        npermutations=worker_npermutations,
                        original_mi=original_mi,
                        max_failed=max_failed,
                        base_seed=np.uint64(base_seed),
                        use_su=_use_su,
                        perm_offset=_cumulative_offset,
                    )
                )
                _cumulative_offset += int(worker_npermutations)
            res = workers_pool(_delayed_calls)

            n_checked = 0
            for worker_nfailed, worker_i in res:
                nfailed += worker_nfailed
                n_checked += worker_i
            i = n_checked - 1  # caller-side i+1 = total checks across workers.

            if nfailed >= max_failed:
                original_mi = 0.0
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

        # Caller-side npermutations==0 guard.
        if i >= 0:
            confidence = 1 - nfailed / (i + 1)

    return original_mi, confidence
