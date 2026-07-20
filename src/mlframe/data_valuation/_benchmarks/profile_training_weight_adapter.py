"""cProfile + wall-clock harness for ``training_sample_weight_from_valuation`` -- the scalability
concern is whether the ``max_valued_rows`` cap actually keeps wall-clock roughly FLAT as ``n_train``
grows past it (linear ``propagate_subsample_values`` cost, not the quadratic ``knn_shapley`` cost it
would pay uncapped).

Run: python -m mlframe.data_valuation._benchmarks.profile_training_weight_adapter
"""

from __future__ import annotations

import cProfile
import io
import pstats
import time

import numpy as np


def _make_fixture(n_train: int, n_val: int = 2000, n_features: int = 12, seed: int = 0):
    """Synthetic 2-class blob-style fixture of the given (n_train, n_val, n_features) shape."""
    rng = np.random.default_rng(seed)
    X_train = rng.standard_normal((n_train, n_features))
    y_train = (X_train[:, 0] + X_train[:, 1] > 0).astype(np.int64)
    X_val = rng.standard_normal((n_val, n_features))
    y_val = (X_val[:, 0] + X_val[:, 1] > 0).astype(np.int64)
    return X_train, y_train, X_val, y_val


def bench_wall_vs_n_train():
    """Wall-clock at n_train in {20k, 100k, 500k, 2M}, capped at max_valued_rows=20000 -- expected to
    grow much slower than n_train itself (linear propagation, not quadratic valuation) past the cap."""
    from mlframe.data_valuation import training_sample_weight_from_valuation

    print("n_train, n_val, max_valued_rows, wall_s")
    _X, _y, X_val, y_val = _make_fixture(1000)  # warm numba JIT cheaply first
    training_sample_weight_from_valuation(_X, _y, X_val[:5], y_val[:5], max_valued_rows=200, rng=np.random.default_rng(0))
    for n_train in (20_000, 100_000, 500_000, 2_000_000):
        X_train, y_train, X_val, y_val = _make_fixture(n_train)
        t0 = time.perf_counter()
        training_sample_weight_from_valuation(X_train, y_train, X_val, y_val, max_valued_rows=20_000, rng=np.random.default_rng(1))
        wall = time.perf_counter() - t0
        print(f"{n_train}, {len(y_val)}, 20000, {wall:.4f}")


def bench_cprofile():
    """cProfile the largest grid point (n_train=2,000,000), print top-25 by cumtime."""
    from mlframe.data_valuation import training_sample_weight_from_valuation

    X_train, y_train, X_val, y_val = _make_fixture(2_000_000)
    training_sample_weight_from_valuation(X_train[:1000], y_train[:1000], X_val[:5], y_val[:5], max_valued_rows=200, rng=np.random.default_rng(0))  # warm
    pr = cProfile.Profile()
    pr.enable()
    training_sample_weight_from_valuation(X_train, y_train, X_val, y_val, max_valued_rows=20_000, rng=np.random.default_rng(2))
    pr.disable()
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats("cumtime")
    ps.print_stats(25)
    print(s.getvalue())


if __name__ == "__main__":
    bench_wall_vs_n_train()
    bench_cprofile()
