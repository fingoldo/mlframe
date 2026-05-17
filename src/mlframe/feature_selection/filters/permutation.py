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

import numpy as np
from joblib import Parallel, delayed
from numba import njit, prange

from ._internals import NMAX_NONPARALLEL_ITERS
from .info_theory import compute_mi_from_classes, merge_vars


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
    np.random.shuffle(arr)


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
    dtype=np.int32,
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

        mi_perm = compute_mi_from_classes(
            classes_x=classes_x, freqs_x=freqs_x,
            classes_y=local, freqs_y=freqs_y, dtype=dtype,
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


@njit(parallel=True, nogil=True, cache=True)
def parallel_mi_prange(
    classes_x: np.ndarray,
    freqs_x: np.ndarray,
    classes_y: np.ndarray,
    freqs_y: np.ndarray,
    npermutations: int,
    original_mi: float,
    base_seed: np.uint64,
    dtype=np.int32,
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

        mi_perm = compute_mi_from_classes(
            classes_x=classes_x, freqs_x=freqs_x,
            classes_y=local, freqs_y=freqs_y, dtype=dtype,
        )
        if mi_perm >= original_mi:
            nfailed_arr[i] = 1

    return int(nfailed_arr.sum()), npermutations


@njit(cache=True)
def parallel_mi(
    classes_x: np.ndarray,
    freqs_x: np.ndarray,
    classes_y: np.ndarray,
    freqs_y: np.ndarray,
    npermutations: int,
    original_mi: float,
    max_failed: int,
    dtype=np.int32,
) -> tuple[int, int]:
    """Worker for the joblib pool used by ``mi_direct``. Returns ``(n_failed, n_checked)`` so the caller can aggregate across pool members. ``npermutations=0`` returns ``(0, 0)`` cleanly."""
    if npermutations == 0:
        return 0, 0

    nfailed = 0
    classes_y_safe = np.asarray(classes_y).copy()
    _i = 0
    for _i in range(npermutations):
        np.random.shuffle(classes_y_safe)
        mi = compute_mi_from_classes(
            classes_x=classes_x, freqs_x=freqs_x,
            classes_y=classes_y_safe, freqs_y=freqs_y, dtype=dtype,
        )
        if mi >= original_mi:
            nfailed += 1
            if nfailed >= max_failed:
                break
    return nfailed, _i + 1


def mi_direct(
    factors_data,
    x: tuple,
    y: tuple,
    factors_nbins: np.ndarray,
    min_occupancy: int = None,
    dtype=np.int32,
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
) -> tuple:
    """CPU mutual-information + permutation-test wrapper.

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
    classes_x, freqs_x, _ = merge_vars(
        factors_data=factors_data, vars_indices=x,
        var_is_nominal=None, factors_nbins=factors_nbins, dtype=dtype,
    )
    if classes_y is None:
        classes_y, freqs_y, _ = merge_vars(
            factors_data=factors_data, vars_indices=y,
            var_is_nominal=None, factors_nbins=factors_nbins, dtype=dtype,
        )

    original_mi = compute_mi_from_classes(
        classes_x=classes_x, freqs_x=freqs_x,
        classes_y=classes_y, freqs_y=freqs_y, dtype=dtype,
    )

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
            )
            i = n_checked - 1
            if nfailed >= max_failed:
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
            )
            i = n_checked - 1
            if nfailed >= max_failed:
                original_mi = 0.0
        elif n_workers and n_workers > 1 and npermutations > NMAX_NONPARALLEL_ITERS:
            if workers_pool is None:
                workers_pool = Parallel(n_jobs=n_workers, **parallel_kwargs)

            res = workers_pool(
                delayed(parallel_mi)(
                    classes_x=classes_x,
                    freqs_x=freqs_x,
                    classes_y=classes_y_safe,
                    freqs_y=freqs_y,
                    dtype=dtype,
                    npermutations=worker_npermutations,
                    original_mi=original_mi,
                    max_failed=max_failed,
                )
                for worker_npermutations in distribute_permutations(
                    npermutations=npermutations, n_workers=n_workers,
                )
            )

            n_checked = 0
            for worker_nfailed, worker_i in res:
                nfailed += worker_nfailed
                n_checked += worker_i
            i = n_checked - 1  # caller-side i+1 = total checks across workers.

            if nfailed >= max_failed:
                original_mi = 0.0
        else:
            if classes_y_safe is None:
                classes_y_safe = classes_y.copy()
            i = -1
            for _i in range(npermutations):
                i = _i
                shuffle_arr(classes_y_safe)
                mi = compute_mi_from_classes(
                    classes_x=classes_x, freqs_x=freqs_x,
                    classes_y=classes_y_safe, freqs_y=freqs_y, dtype=dtype,
                )
                if mi >= original_mi:
                    nfailed += 1
                    if nfailed >= max_failed:
                        original_mi = 0.0
                        break

        # Caller-side npermutations==0 guard.
        if i >= 0:
            confidence = 1 - nfailed / (i + 1)

    return original_mi, confidence
