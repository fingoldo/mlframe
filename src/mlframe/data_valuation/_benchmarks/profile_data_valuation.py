"""cProfile + wall-clock harness for gt_04's ``knn_shapley`` closed-form engine.

Run: python -m mlframe.data_valuation._benchmarks.profile_data_valuation

Measures ``knn_shapley`` wall at ``n_train`` in {2000, 20000, 100000} x ``n_val`` in {200, 1000}
(section 5's grid) and records whether the njit recursion kernel wins over a pure-numpy reference
loop -- kept per the repo's acceleration-ladder convention (keep both backends if the win is real).
"""

from __future__ import annotations

import cProfile
import io
import pstats
import time

import numpy as np


def _make_fixture(n_train: int, n_val: int, n_features: int = 8, seed: int = 0):
    """Synthetic 2-class blob-style fixture of the given (n_train, n_val, n_features) shape."""
    rng = np.random.default_rng(seed)
    X_train = rng.standard_normal((n_train, n_features))
    y_train = (X_train[:, 0] + X_train[:, 1] > 0).astype(np.int64)
    X_val = rng.standard_normal((n_val, n_features))
    y_val = (X_val[:, 0] + X_val[:, 1] > 0).astype(np.int64)
    return X_train, y_train, X_val, y_val


def _numpy_reference_recursion(match: np.ndarray, k: int) -> np.ndarray:
    """Pure-Python/numpy reference implementation of the recursion, for the njit-vs-numpy wall comparison."""
    n = match.shape[0]
    value = np.empty(n, dtype=np.float64)
    value[n - 1] = match[n - 1] / n
    for p in range(n - 2, -1, -1):
        i = p + 1
        value[p] = value[p + 1] + (match[p] - match[p + 1]) / k * min(k, i) / i
    return value


def bench_wall_grid():
    """Wall-clock ``knn_shapley`` across the plan's (n_train, n_val) grid; report seconds per point."""
    from mlframe.data_valuation import knn_shapley

    print("n_train, n_val, wall_s")
    for n_train in (2000, 20000, 100000):
        for n_val in (200, 1000):
            X_train, y_train, X_val, y_val = _make_fixture(n_train, n_val)
            knn_shapley(X_train, y_train, X_val[:5], y_val[:5], k=5)  # warm numba JIT
            t0 = time.perf_counter()
            knn_shapley(X_train, y_train, X_val, y_val, k=5)
            wall = time.perf_counter() - t0
            print(f"{n_train}, {n_val}, {wall:.4f}")


def bench_njit_vs_numpy_recursion():
    """Compare the njit recursion kernel against a pure-numpy reference loop at realistic n_train widths."""
    from mlframe.data_valuation._knn_shapley import _knn_shapley_recursion

    rng = np.random.default_rng(0)
    for n_train in (2000, 20000, 100000):
        match = (rng.random(n_train) < 0.5).astype(np.float64)
        k = 5
        _knn_shapley_recursion(match, k)  # warm

        t0 = time.perf_counter()
        for _ in range(50):
            _knn_shapley_recursion(match, k)
        wall_njit = (time.perf_counter() - t0) / 50

        t0 = time.perf_counter()
        for _ in range(50):
            _numpy_reference_recursion(match, k)
        wall_numpy = (time.perf_counter() - t0) / 50

        print(f"n_train={n_train}: njit={wall_njit * 1000:.4f}ms, numpy_loop={wall_numpy * 1000:.4f}ms, speedup={wall_numpy / wall_njit:.2f}x")


def bench_cprofile():
    """cProfile the largest grid point, print top-25 by cumtime."""
    from mlframe.data_valuation import knn_shapley

    X_train, y_train, X_val, y_val = _make_fixture(20000, 200)
    knn_shapley(X_train, y_train, X_val[:5], y_val[:5], k=5)  # warm
    pr = cProfile.Profile()
    pr.enable()
    knn_shapley(X_train, y_train, X_val, y_val, k=5)
    pr.disable()
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats("cumtime")
    ps.print_stats(25)
    print(s.getvalue())


if __name__ == "__main__":
    bench_njit_vs_numpy_recursion()
    bench_wall_grid()
    bench_cprofile()
