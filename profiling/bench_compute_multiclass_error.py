"""A/B bench: legacy per-class Python loop vs batched numba kernel
for ``compute_probabilistic_multiclass_error`` on 1M-row inputs.

Wave 6 fuzz aggregate attributed 23 s of wall-time to this function
across 4 combos (60 ms / call * 392 calls). The hot path is
``method='multicrit'`` + ``verbose=False`` which calls
``fast_ice_only`` per class -- K Python->numba transitions per call.

The new ``_batch_per_class_ice_kernel`` fuses all K classes into one
numba.njit(parallel=True) call, processing each class in a prange
iteration. Inner per-class logic is bit-exact equivalent of
``fast_ice_only``.

Usage:
    python -m mlframe.profiling.bench_compute_multiclass_error
"""

from __future__ import annotations

import statistics
import time

import numpy as np

from mlframe.metrics.core import (
    compute_probabilistic_multiclass_error,
    fast_ice_only,
)


def _legacy_compute(y_true, probs, K):
    """Pre-Wave-7b form: per-class Python loop over fast_ice_only."""
    total_error = 0.0
    weights_sum = 0
    for class_id in range(K):
        if K == 2 and class_id == 0:
            continue
        y_pred = probs[class_id]
        correct = (y_true == class_id).astype(np.int8)
        ce = fast_ice_only(
            y_true=correct, y_pred=y_pred, nbins=10, use_weights=True,
        )
        total_error += ce
        weights_sum += 1
    return total_error / weights_sum if weights_sum > 0 else float("nan")


def main() -> None:
    rng = np.random.default_rng(42)
    N = 1_000_000
    print(f"# compute_probabilistic_multiclass_error on N={N:_}")
    print()

    for K in (3, 5, 10):
        y_true = rng.integers(0, K, N).astype(np.int64)
        # Simulate one-vs-rest probabilities (independent uniform per class).
        probs = [rng.uniform(0, 1, N).astype(np.float64) for _ in range(K)]

        # Equivalence check first
        a = _legacy_compute(y_true, probs, K)
        b = compute_probabilistic_multiclass_error(
            y_true=y_true, y_score=probs, method="multicrit", verbose=False,
        )
        ok = abs(a - b) < 1e-9
        print(f"## K={K}  equivalence: legacy={a:.6f} batched={b:.6f} {'OK' if ok else 'MISMATCH'}")

        # Warm numba (first call lowers the kernel)
        compute_probabilistic_multiclass_error(
            y_true=y_true, y_score=probs, method="multicrit", verbose=False,
        )
        _legacy_compute(y_true, probs, K)

        def bench(fn, label, n_repeat=5):
            times = []
            for _ in range(n_repeat):
                t0 = time.perf_counter()
                fn()
                times.append(time.perf_counter() - t0)
            m = statistics.mean(times)
            s = statistics.stdev(times) if len(times) > 1 else 0.0
            print(f"  {label:<50} {m*1000:>9.1f} ms +/- {s*1000:>6.1f} ms")

        bench(lambda: _legacy_compute(y_true, probs, K),
              "OLD: per-class Python loop (fast_ice_only)")
        bench(lambda: compute_probabilistic_multiclass_error(
                y_true=y_true, y_score=probs, method="multicrit", verbose=False,
              ),
              "NEW: batched numba kernel (prange over K)")
        print()


if __name__ == "__main__":
    main()
