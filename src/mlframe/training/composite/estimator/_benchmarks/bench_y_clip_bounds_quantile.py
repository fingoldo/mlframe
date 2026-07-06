"""Microbench: single vs two np.quantile calls in _y_train_clip_bounds.

One np.quantile(y, (0.001, 0.999)) does ONE O(n log n) sort instead of two -> ~2x, bit-identical.
Run: python bench_y_clip_bounds_quantile.py

Measured (n=100k float64, warm, 500 iters):
    two-call : ~137 us/call
    one-call : ~ 69 us/call   -> ~2.0x speedup, maxdiff 0.0
"""

import time

import numpy as np


def _two(y):
    return float(np.quantile(y, 0.001)), float(np.quantile(y, 0.999))


def _one(y):
    lo, hi = (float(v) for v in np.quantile(y, (0.001, 0.999)))
    return lo, hi


def _bench(fn, y, iters):
    fn(y)  # warm
    t0 = time.perf_counter()
    for _ in range(iters):
        fn(y)
    return (time.perf_counter() - t0) / iters * 1e6  # us/call


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    y = rng.standard_normal(100_000)
    iters = 500
    a = _two(y)
    b = _one(y)
    maxdiff = max(abs(a[0] - b[0]), abs(a[1] - b[1]))
    t_two = _bench(_two, y, iters)
    t_one = _bench(_one, y, iters)
    print(f"two-call: {t_two:.1f} us/call")
    print(f"one-call: {t_one:.1f} us/call")
    print(f"speedup : {t_two / t_one:.2f}x")
    print(f"maxdiff : {maxdiff}")
    assert maxdiff == 0.0, "single-quantile must be bit-identical to two calls"  # nosec B101 - internal invariant check in src/mlframe/training/composite/estimator/_benchmarks, not reachable with untrusted input
