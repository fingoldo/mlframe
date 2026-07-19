"""Measurement-only bench for joblib site 1: ``permutation.py``'s outer-parallel
``Parallel(n_jobs=n_workers, **parallel_kwargs)(delayed(parallel_mi)(...))`` pool
(``permutation.py:881-928``), gated by ``npermutations > NMAX_NONPARALLEL_ITERS`` (=2).

Production MRMR defaults ``fe_npermutations=3`` / ``baseline_npermutations=2`` (see
``mrmr/_mrmr_class.py:427-428``, ``_screen_predictors.py:104-105``) -- both AT or BARELY
above the ``NMAX_NONPARALLEL_ITERS=2`` gate, so this pool almost never dispatches in a
real fit. This bench measures (a) the realistic case (npermutations=3) to confirm the
guard suppresses the pool entirely, and (b) a swept range of larger npermutations values
to find where (if ever) n_jobs=2/4 pays off over n_jobs=1 for THIS specific worker
function (``parallel_mi``, a hand-rolled pure-Python Fisher-Yates + MI loop -- no numba,
no GIL release).

Run: PYTHONPATH=src python src/mlframe/feature_selection/filters/_benchmarks/bench_joblib_njobs_site1_outer_permutation_mi.py
"""
from __future__ import annotations

import time

import numpy as np
from joblib import Parallel, delayed

from mlframe.feature_selection.filters._internals import NMAX_NONPARALLEL_ITERS
from mlframe.feature_selection.filters.permutation import distribute_permutations, mi_direct, parallel_mi


def _make_classes(n, seed):
    rng = np.random.default_rng(seed)
    classes_x = rng.integers(0, 8, n).astype(np.int32)
    classes_y = rng.integers(0, 2, n).astype(np.int32)
    freqs_x = np.bincount(classes_x, minlength=8).astype(np.float64)
    freqs_y = np.bincount(classes_y, minlength=2).astype(np.float64)
    return classes_x, freqs_x, classes_y, freqs_y


def _best_of(fn, reps=3):
    best = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def _run_outer_pool(n_workers, npermutations, classes_x, freqs_x, classes_y, freqs_y):
    pool = Parallel(n_jobs=n_workers, backend="threading")
    loads = distribute_permutations(npermutations=npermutations, n_workers=n_workers)
    offset = 0
    calls = []
    for w in loads:
        calls.append(
            delayed(parallel_mi)(
                classes_x=classes_x, freqs_x=freqs_x, classes_y=classes_y, freqs_y=freqs_y,
                dtype=np.int32, npermutations=w, original_mi=0.05, max_failed=10**9,
                base_seed=np.uint64(42), use_su=False, perm_offset=offset,
            )
        )
        offset += int(w)
    return pool(calls)


def _run_serial(npermutations, classes_x, freqs_x, classes_y, freqs_y):
    return parallel_mi(
        classes_x=classes_x, freqs_x=freqs_x, classes_y=classes_y, freqs_y=freqs_y,
        dtype=np.int32, npermutations=npermutations, original_mi=0.05, max_failed=10**9,
        base_seed=np.uint64(42), use_su=False, perm_offset=0,
    )


def main():
    n = 99401  # wellbore-100k row count
    classes_x, freqs_x, classes_y, freqs_y = _make_classes(n, seed=0)

    print("=== Realistic MRMR scale: fe_npermutations=3 (guard NMAX_NONPARALLEL_ITERS=%d) ===" % NMAX_NONPARALLEL_ITERS)
    print(f"npermutations=3 > NMAX_NONPARALLEL_ITERS={NMAX_NONPARALLEL_ITERS}: {3 > NMAX_NONPARALLEL_ITERS} -> outer pool dispatches" if 3 > NMAX_NONPARALLEL_ITERS else "npermutations=3 does NOT clear the guard -> pool never built, purely serial mi_direct path")
    # warm up
    _run_serial(3, classes_x, freqs_x, classes_y, freqs_y)
    t_serial3 = _best_of(lambda: _run_serial(3, classes_x, freqs_x, classes_y, freqs_y))
    print(f"serial parallel_mi(npermutations=3): {t_serial3*1e3:.3f} ms")

    print("\n=== Sweep: does the outer joblib pool ever pay off for parallel_mi at realistic n=99401? ===")
    for npermutations in (10, 100, 1000, 10000):
        # warm
        _run_serial(npermutations, classes_x, freqs_x, classes_y, freqs_y)
        t1 = _best_of(lambda: _run_serial(npermutations, classes_x, freqs_x, classes_y, freqs_y))
        _run_outer_pool(2, npermutations, classes_x, freqs_x, classes_y, freqs_y)
        t2 = _best_of(lambda: _run_outer_pool(2, npermutations, classes_x, freqs_x, classes_y, freqs_y))
        _run_outer_pool(4, npermutations, classes_x, freqs_x, classes_y, freqs_y)
        t4 = _best_of(lambda: _run_outer_pool(4, npermutations, classes_x, freqs_x, classes_y, freqs_y))
        print(f"npermutations={npermutations:6d}  n_jobs=1: {t1*1e3:9.3f} ms  n_jobs=2: {t2*1e3:9.3f} ms  n_jobs=4: {t4*1e3:9.3f} ms  "
              f"speedup@2={t1/t2:.2f}x  speedup@4={t1/t4:.2f}x")


if __name__ == "__main__":
    main()
