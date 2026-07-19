"""A/B bench: existing njit(parallel=True) ``parallel_mi_prange`` kernel vs joblib(backend="threading")
outer-pool dispatch of the njit(nogil=True) ``parallel_mi`` worker, at the realistic ``mi_direct``
call site (``permutation.py``'s ``Parallel(backend="threading")`` around the "outer" branch).

Context: a 7-site joblib.Parallel audit (2026-07-19, gpu-utilization worktree) found this call site
(permutation.py's outer-parallelism ``mi_direct`` path) never wins joblib over serial at the realistic
``fe_npermutations=2-3`` regime (0.36-0.40x) and only starts winning above ~50-100 permutations
(1.9-3.4x at n_jobs=4). The question this bench answers: at that SAME >=64-permutation regime where
joblib already wins over serial, does the ALREADY-EXISTING ``parallel_mi_prange`` njit(parallel=True)
kernel (used today only for the ``n_workers<=1`` / ``parallelism="inner"`` fallback path, see
``permutation.py:972``) beat the joblib outer-pool dispatch outright -- i.e. is the joblib branch at
``permutation.py:883-940`` obsolete and replaceable by always routing through ``parallel_mi_prange``
regardless of ``n_workers``?

Bit-identity: ``parallel_mi_prange`` and ``parallel_mi`` implement the SAME per-iteration LCG seeding
scheme (Knuth multiplicative hash keyed by absolute permutation index, not worker-partition-relative --
see ``permutation.py`` Wave 9.1 iter-18 fix docstring), so ``(nfailed, n_checked)`` must match bit-for-bit
between the two dispatch paths for the same ``base_seed``/``npermutations``. Verified below.

Run: python -m mlframe.feature_selection.filters._benchmarks.bench_permutation_njit_prange_vs_joblib
"""

import time

import numpy as np

from joblib import Parallel, delayed

from mlframe.feature_selection.filters.permutation import (
    parallel_mi,
    parallel_mi_prange,
    distribute_permutations,
)


def make_classes(n: int, k_x: int = 5, k_y: int = 2, seed: int = 0):
    rng = np.random.default_rng(seed)
    classes_x = rng.integers(0, k_x, size=n).astype(np.int32)
    classes_y = rng.integers(0, k_y, size=n).astype(np.int32)
    freqs_x = np.bincount(classes_x, minlength=k_x).astype(np.float64)
    freqs_y = np.bincount(classes_y, minlength=k_y).astype(np.float64)
    return classes_x, freqs_x, classes_y, freqs_y


def outer_joblib(classes_x, freqs_x, classes_y, freqs_y, npermutations, original_mi, base_seed, n_jobs):
    """Mirror the real ``mi_direct`` outer-parallel branch (permutation.py:890-940): distribute
    permutations across a joblib(backend="threading") pool of ``parallel_mi`` (njit nogil) workers."""
    pool = Parallel(n_jobs=n_jobs, backend="threading")
    worker_loads = distribute_permutations(npermutations=npermutations, n_workers=n_jobs)
    cumulative_offset = 0
    calls = []
    for worker_npermutations in worker_loads:
        calls.append(
            delayed(parallel_mi)(
                classes_x=classes_x,
                freqs_x=freqs_x,
                classes_y=classes_y,
                freqs_y=freqs_y,
                npermutations=int(worker_npermutations),
                original_mi=original_mi,
                max_failed=int(npermutations),  # disable early-exit so both paths run the FULL budget
                base_seed=np.uint64(base_seed),
                perm_offset=cumulative_offset,
            )
        )
        cumulative_offset += int(worker_npermutations)
    res = pool(calls)
    nfailed = sum(r[0] for r in res)
    n_checked = sum(r[1] for r in res)
    return nfailed, n_checked


def inner_prange(classes_x, freqs_x, classes_y, freqs_y, npermutations, original_mi, base_seed):
    return parallel_mi_prange(
        classes_x=classes_x,
        freqs_x=freqs_x,
        classes_y=classes_y,
        freqs_y=freqs_y,
        npermutations=npermutations,
        original_mi=original_mi,
        base_seed=np.uint64(base_seed),
    )


def best_of(fn, n_reps=5):
    times = []
    for _ in range(n_reps):
        t0 = time.perf_counter()
        result = fn()
        times.append(time.perf_counter() - t0)
    return min(times), result


def main():
    base_seed = 12345
    original_mi = 0.01  # small, so most permutations don't early-fail (irrelevant here: max_failed forced high)

    print(f"{'n':>8} {'nperm':>6} {'njobs':>6} {'serial(s)':>12} {'joblib(s)':>12} {'prange(s)':>12} " f"{'joblib_x':>10} {'prange_x':>10} {'identical':>10}")

    for n in (2_000, 20_000, 100_000):
        classes_x, freqs_x, classes_y, freqs_y = make_classes(n)
        # warm njit caches once (compile time excluded from timing)
        parallel_mi(classes_x, freqs_x, classes_y, freqs_y, 2, original_mi, 100, np.uint64(base_seed), 0)
        parallel_mi_prange(classes_x, freqs_x, classes_y, freqs_y, 2, original_mi, np.uint64(base_seed))

        for npermutations in (2, 3, 100, 500):
            # serial baseline: single parallel_mi call, full budget, no early exit
            def _serial():
                return parallel_mi(
                    classes_x, freqs_x, classes_y, freqs_y, npermutations, original_mi,
                    int(npermutations), np.uint64(base_seed), 0,
                )
            t_serial, serial_res = best_of(_serial)

            def _prange():
                return inner_prange(classes_x, freqs_x, classes_y, freqs_y, npermutations, original_mi, base_seed)
            t_prange, prange_res = best_of(_prange)

            for n_jobs in (2, 4):
                def _joblib():
                    return outer_joblib(classes_x, freqs_x, classes_y, freqs_y, npermutations, original_mi, base_seed, n_jobs)
                t_joblib, joblib_res = best_of(_joblib, n_reps=3)

                identical = (joblib_res == serial_res == prange_res)
                print(f"{n:>8} {npermutations:>6} {n_jobs:>6} {t_serial:>12.5f} {t_joblib:>12.5f} {t_prange:>12.5f} "
                      f"{t_serial / t_joblib:>10.2f} {t_serial / t_prange:>10.2f} {str(identical):>10}")


if __name__ == "__main__":
    main()
