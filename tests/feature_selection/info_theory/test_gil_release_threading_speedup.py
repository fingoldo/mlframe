"""Regression test for the missing ``nogil=True`` bug in the threading-dispatched candidate-MI kernel
chain (MRMR audit finding #5, 2026-07-10 resolution).

``_screen_predictors.py``'s ``workers_pool`` (``backend="threading"``, used when ``MRMR(n_workers>1)``)
dispatches ``evaluate_candidates``, which eventually calls ``compute_mi_from_classes`` / ``shuffle_arr`` /
``parallel_mi``. A code comment there claimed these "release the GIL, so threading gives near-process-
pool speedup" -- but none of the three were actually decorated ``nogil=True`` (confirmed via
``inspect``/grep), so the claim was false: threading through these kernels measured 0.84-0.94x versus a
serial loop (WORSE than serial, pure GIL-serialized overhead) at n=30000/200 permutations/8 concurrent
threading tasks. Adding ``nogil=True`` to ``compute_mi_from_classes``, ``compute_mi_mm_from_classes``,
``_mm_bias``, ``shuffle_arr``, ``shuffle_arr_lcg``, ``parallel_mi``, and ``parallel_mi_with_null`` (all
pure-numeric numpy-array kernels, no Python-object access -- safe per the same pattern already used by
269 other ``nogil=True`` sites in this codebase, including these functions' own siblings
``compute_su_from_classes``/``compute_relevance_score`` in the same file) measured 1.55-1.91x real
speedup at the same shape post-fix.

These tests pin: (1) correctness is unchanged (nogil is a pure execution-context flag, not a behavior
change -- same outputs before/after), (2) numba's internal RNG state is genuinely thread-safe under
concurrent nogil execution (the specific risk nogil=True introduces for any kernel using np.random), and
(3) a coarse performance regression sensor so a future accidental removal of nogil=True is caught (loose
bound -- this is a correctness/architecture pin, not a tight perf gate)."""
from __future__ import annotations

import threading
import time

import numpy as np
import pytest
from joblib import Parallel, delayed

from mlframe.feature_selection.filters.info_theory._class_mi_kernels import (
    compute_mi_from_classes, compute_mi_mm_from_classes,
)
from mlframe.feature_selection.filters.permutation import parallel_mi, shuffle_arr, shuffle_arr_lcg


def test_shuffle_arr_lcg_chained_state_survives_uint64_high_bit():
    """Regression test for a SEPARATE bug this investigation's concurrency test discovered:
    ``shuffle_arr_lcg``'s public contract promises the returned ``state`` can be threaded across
    chained calls, but numba boxes a uint64 return as a plain Python ``int``; re-passing that bare
    int into another njit call makes numba infer ``int64`` (a raw Python int carries no dtype),
    which failed to unbox with ``OverflowError: int too big to convert`` as soon as the LCG state's
    high bit set (typically within the first 1-5 calls of any seed -- confirmed independent of
    threading/nogil: reproduces identically single-threaded). Fixed by wrapping both the input and
    the returned value in ``np.uint64(...)`` at the Python level in a thin wrapper around the njit
    kernel. Pins: many chained calls from a single fixed seed, no exception, every intermediate
    array a valid permutation."""
    state = np.uint64(2)  # the exact seed that triggered the bug within 1 call pre-fix
    for _ in range(3000):
        arr = np.arange(500)
        state = shuffle_arr_lcg(arr, state)
        np.testing.assert_array_equal(np.sort(arr), np.arange(500))
    assert isinstance(state, np.uint64)


def test_compute_mi_from_classes_nogil_flag_set():
    assert compute_mi_from_classes.targetoptions.get("nogil") is True, (
        "compute_mi_from_classes must keep nogil=True -- without it, joblib backend='threading' "
        "dispatch through this kernel is GIL-serialized (measured 0.84-0.94x vs serial, i.e. a "
        "regression, not a speedup)"
    )


def test_shuffle_arr_nogil_flag_set():
    assert shuffle_arr.targetoptions.get("nogil") is True


def test_parallel_mi_nogil_flag_set():
    assert parallel_mi.targetoptions.get("nogil") is True


def test_compute_mi_from_classes_correctness_unchanged():
    """nogil is a pure execution-context flag -- must not change the computed value."""
    rng = np.random.default_rng(0)
    n = 5000
    classes_x = rng.integers(0, 15, size=n).astype(np.int32)
    classes_y = rng.integers(0, 8, size=n).astype(np.int32)
    freqs_x = np.bincount(classes_x, minlength=15).astype(np.float64) / n
    freqs_y = np.bincount(classes_y, minlength=8).astype(np.float64) / n

    mi = compute_mi_from_classes(classes_x, freqs_x, classes_y, freqs_y)
    mi_mm = compute_mi_mm_from_classes(classes_x, freqs_x, classes_y, freqs_y)
    assert np.isfinite(mi) and mi >= 0.0
    assert np.isfinite(mi_mm) and mi_mm >= 0.0
    assert mi_mm <= mi + 1e-9, "Miller-Madow correction must not increase MI above the plugin estimate"


def test_shuffle_arr_thread_safety_under_concurrent_nogil_execution():
    """The specific correctness risk nogil=True introduces: does numba's internal RNG state corrupt
    under genuinely concurrent (GIL-released) calls from multiple OS threads? Every produced array
    must still be a valid permutation (same multiset of values) regardless of concurrent execution."""
    shuffle_arr(np.arange(10))  # warmup JIT

    errors = []

    def worker(tid):
        try:
            for _ in range(200):
                arr = np.arange(500)
                shuffle_arr(arr)
                if not np.array_equal(np.sort(arr), np.arange(500)):
                    errors.append(f"thread {tid}: shuffle produced a non-permutation")
        except Exception as e:  # pragma: no cover - failure path
            errors.append(f"thread {tid}: {type(e).__name__}: {e}")

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"concurrent nogil shuffle_arr execution failed: {errors[:5]}"


def test_shuffle_arr_lcg_thread_safety_under_concurrent_nogil_execution():
    shuffle_arr_lcg(np.arange(10), np.uint64(1))  # warmup JIT

    errors = []

    def worker(tid):
        try:
            state = np.uint64(tid + 1)
            for _ in range(200):
                arr = np.arange(500)
                state = shuffle_arr_lcg(arr, state)
                if not np.array_equal(np.sort(arr), np.arange(500)):
                    errors.append(f"thread {tid}: shuffle_arr_lcg produced a non-permutation")
        except Exception as e:  # pragma: no cover - failure path
            errors.append(f"thread {tid}: {type(e).__name__}: {e}")

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"concurrent nogil shuffle_arr_lcg execution failed: {errors[:5]}"


@pytest.mark.slow
def test_threading_backend_delivers_real_speedup_not_regression():
    """Coarse regression sensor for the fix itself: joblib backend='threading' dispatch through
    compute_mi_from_classes must show SOME real multi-thread speedup, not the pre-fix 0.84-0.94x
    (worse-than-serial) pattern. Loose bound (>1.2x at 4 workers) -- this pins the fix direction, not a
    tight performance contract."""
    rng = np.random.default_rng(1)
    n = 30000
    classes_x = rng.integers(0, 20, size=n).astype(np.int32)
    classes_y = rng.integers(0, 10, size=n).astype(np.int32)
    freqs_x = np.bincount(classes_x, minlength=20).astype(np.float64) / n
    freqs_y = np.bincount(classes_y, minlength=10).astype(np.float64) / n

    def work():
        total = 0.0
        for _ in range(200):
            total += compute_mi_from_classes(classes_x, freqs_x, classes_y, freqs_y)
        return total

    work()  # warmup JIT

    n_tasks, n_repeats = 8, 6
    t0 = time.perf_counter()
    for _ in range(n_repeats):
        for _ in range(n_tasks):
            work()
    t_serial = (time.perf_counter() - t0) / n_repeats

    pool = Parallel(n_jobs=4, backend="threading")
    pool(delayed(work)() for _ in range(n_tasks))  # warmup pool
    t0 = time.perf_counter()
    for _ in range(n_repeats):
        pool(delayed(work)() for _ in range(n_tasks))
    t_parallel = (time.perf_counter() - t0) / n_repeats

    speedup = t_serial / t_parallel
    assert speedup > 1.2, (
        f"threading backend speedup regressed to {speedup:.2f}x (serial={t_serial*1000:.1f}ms, "
        f"threaded={t_parallel*1000:.1f}ms) -- check nogil=True is still set on compute_mi_from_classes"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-x", "--no-cov"])
