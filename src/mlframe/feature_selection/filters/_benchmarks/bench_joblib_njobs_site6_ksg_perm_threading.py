"""Measurement-only bench for joblib site 6: ``estimators.py:163``'s
``Parallel(n_jobs=n_jobs, max_nbytes=int(1e7), backend="threading")`` pool running
``_one_perm`` (a full ``ksg_mi_with_target`` sklearn KNN-MI call on a shuffled target)
inside ``ksg_mi_with_significance``.

Each ``_one_perm`` call is itself sklearn's KSG estimator over ALL requested features at
once (not a cheap per-permutation scalar) -- docstring cites ~1-2s serial for n=10000,
p=100, n_permutations=50, i.e. ~20-40ms/call. This is a much heavier per-task unit than
sites 1/2/3/5, so threading should plausibly pay off here; measuring to confirm.

Run: PYTHONPATH=src python src/mlframe/feature_selection/filters/_benchmarks/bench_joblib_njobs_site6_ksg_perm_threading.py
"""
from __future__ import annotations

import time

import numpy as np

from mlframe.feature_selection.filters.estimators import ksg_mi_with_significance


def _make_data(n, p, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p))
    y = (X[:, 0] + X[:, 1] > 0).astype(np.int64)
    return X, y


def _run(n_jobs, n_permutations, X, y):
    t0 = time.perf_counter()
    ksg_mi_with_significance(
        X=X, y=y, feature_indices=list(range(X.shape[1])),
        n_permutations=n_permutations, n_jobs=n_jobs,
    )
    return time.perf_counter() - t0


def main():
    for n, p, n_permutations, label in (
        (2000, 10, 10, "small (n=2000, p=10, n_permutations=10)"),
        (10000, 50, 50, "docstring-representative (n=10000, p=50, n_permutations=50)"),
    ):
        X, y = _make_data(n, p)
        print(f"\n=== {label} ===")
        _run(1, n_permutations, X, y)  # warm sklearn/numba
        t1 = min(_run(1, n_permutations, X, y) for _ in range(2))
        for n_jobs in (2, 4):
            _run(n_jobs, n_permutations, X, y)  # warm pool
            t = min(_run(n_jobs, n_permutations, X, y) for _ in range(2))
            print(f"n_jobs=1: {t1:6.2f}s   n_jobs={n_jobs}: {t:6.2f}s   speedup={t1/t:.2f}x")


if __name__ == "__main__":
    main()
