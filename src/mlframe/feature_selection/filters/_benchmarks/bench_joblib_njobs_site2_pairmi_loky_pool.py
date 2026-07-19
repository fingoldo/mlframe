"""Measurement-only bench for joblib site 2: ``_step_pairmi.py``'s loky CPU pool
(``_run_loky_pair_mi_pool``, ``_step_pairmi.py:326-353``) scoring prospective-pair MI
chunks via ``compute_pairs_mis`` (``feature_engineering.py:160``).

Serial-bypass guard (``_step_pairmi.py:234``): ``n_jobs<=1 or n_pairs<max(2,n_jobs) or
_all_pairs_precomputed``. In production (2026-07-18 plan finding), GPU batch
pair-MI precompute (``dispatch_batch_pair_mi_chunked``) covers ~100% of pairs on a
wellbore-100k fit, so ``_all_pairs_precomputed=True`` and this loky pool is SKIPPED
entirely on that path. It only fires when GPU precompute is unavailable/incomplete
(CPU-only run, or a VRAM-starved partial precompute). This bench measures the
CPU-only-fallback case directly: real ``compute_pairs_mis`` over chunks of pairs,
``fe_npermutations=3`` (MRMR production default, ``mrmr/_mrmr_class.py:770``).

Run: PYTHONPATH=src CUDA_VISIBLE_DEVICES="" python src/mlframe/feature_selection/filters/_benchmarks/bench_joblib_njobs_site2_pairmi_loky_pool.py
"""
from __future__ import annotations

import time
from itertools import combinations

import numpy as np
from joblib import Parallel, delayed
from joblib._parallel_backends import LokyBackend

from mlframe.feature_selection.filters._joblib_safe import disable_cuda_in_worker
from mlframe.feature_selection.filters.feature_engineering import compute_pairs_mis


def _make_data(n, n_cols, seed):
    rng = np.random.default_rng(seed)
    data = rng.integers(0, 8, size=(n, n_cols)).astype(np.int16)
    y = rng.integers(0, 2, n).astype(np.int32)
    nbins = np.full(n_cols, 8, dtype=np.int64)
    freqs_y = np.bincount(y, minlength=2).astype(np.float64)
    return data, y, nbins, freqs_y


def _best_of(fn, reps=3):
    best = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def _run_serial(all_pairs, data, y, nbins, freqs_y, fe_npermutations):
    return compute_pairs_mis(
        all_pairs=list(all_pairs), data=data, target_indices=None, nbins=nbins,
        classes_y=y, classes_y_safe=None, freqs_y=freqs_y,
        fe_min_nonzero_confidence=0.99, fe_npermutations=fe_npermutations,
        cached_confident_MIs={}, cached_MIs={}, fe_min_pair_mi=-1.0, fe_min_pair_mi_prevalence=0.0,
    )


def _lazy_chunks(iterable, chunk_size):
    it = list(iterable)
    for i in range(0, len(it), chunk_size):
        yield it[i : i + chunk_size]


def _run_pool(n_jobs, all_pairs, data, y, nbins, freqs_y, fe_npermutations):
    backend = LokyBackend(inner_max_num_threads=1, initializer=disable_cuda_in_worker)
    all_pairs = list(all_pairs)
    chunk_size = max(1, len(all_pairs) // (n_jobs * 2))
    return list(
        Parallel(n_jobs=n_jobs, backend=backend, timeout=300)(
            delayed(compute_pairs_mis)(
                all_pairs=chunk, data=data, target_indices=None, nbins=nbins,
                classes_y=y, classes_y_safe=None, freqs_y=freqs_y,
                fe_min_nonzero_confidence=0.99, fe_npermutations=fe_npermutations,
                cached_confident_MIs={}, cached_MIs={}, fe_min_pair_mi=-1.0, fe_min_pair_mi_prevalence=0.0,
            )
            for chunk in _lazy_chunks(all_pairs, chunk_size)
        )
    )


def main():
    n = 99401
    fe_npermutations = 3  # production MRMR default

    for n_cols, label in ((20, "small round (C(20,2)=190 pairs)"), (100, "large round (C(100,2)=4950 pairs)")):
        data, y, nbins, freqs_y = _make_data(n, n_cols, seed=0)
        all_pairs = list(combinations(range(n_cols), 2))
        print(f"\n=== {label}, n_pairs={len(all_pairs)}, fe_npermutations={fe_npermutations} ===")
        # warm up (JIT / imports)
        _run_serial(all_pairs[:5], data, y, nbins, freqs_y, fe_npermutations)
        t1 = _best_of(lambda: _run_serial(all_pairs, data, y, nbins, freqs_y, fe_npermutations), reps=3)
        print(f"n_jobs=1 (serial): {t1*1e3:.1f} ms")
        for n_jobs in (2, 4):
            _run_pool(n_jobs, all_pairs[:8], data, y, nbins, freqs_y, fe_npermutations)  # warm pool spawn
            t = _best_of(lambda: _run_pool(n_jobs, all_pairs, data, y, nbins, freqs_y, fe_npermutations), reps=3)
            print(f"n_jobs={n_jobs} (loky pool): {t*1e3:.1f} ms  speedup={t1/t:.2f}x")


if __name__ == "__main__":
    main()
