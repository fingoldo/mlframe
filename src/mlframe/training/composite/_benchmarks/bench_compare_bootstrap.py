"""Bench for the CPX26 row-chunked paired-bootstrap CI in compare.py.

Demonstrates the RAM win: the OLD monolithic `rng.integers(0, n, size=(n_boot, n))`
+ `diff[idx]` gather materialises two n_boot*n int64/float64 temporaries
(~16 GB at n_boot=1000, n=1e6); the NEW path caps peak temp at (block, n).

Run:  python -m mlframe.training.composite._benchmarks.bench_compare_bootstrap
"""
from __future__ import annotations

import time
import tracemalloc

import numpy as np

from mlframe.training.composite.compare import _paired_bootstrap_ci


def _old_monolithic(diff, n_boot, alpha, rng):
    n = diff.shape[0]
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_means = diff[idx].mean(axis=1)
    lo = float(np.quantile(boot_means, alpha / 2.0))
    hi = float(np.quantile(boot_means, 1.0 - alpha / 2.0))
    obs = float(diff.mean())
    tail = float(np.mean(boot_means <= 0.0)) if obs >= 0 else float(np.mean(boot_means >= 0.0))
    return lo, hi, min(1.0, 2.0 * tail)


def _peak_bytes(fn, *args):
    tracemalloc.start()
    tracemalloc.reset_peak()
    fn(*args)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak


def _best_of(fn, args, reps=5):
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn(*args)
        ts.append(time.perf_counter() - t0)
    return min(ts), float(np.median(ts))


def main():
    alpha = 0.05
    print("=== CPX26 paired-bootstrap chunk bench ===")

    # RAM demonstration: peak-temp bytes (theoretical full shape too large to
    # actually allocate the OLD path on most boxes, so measure at a shape that
    # still fits, then report the analytic n_boot*n*8*2 for the prod shape).
    for n, n_boot in [(100_000, 1000), (1_000_000, 200)]:
        rng_d = np.random.default_rng(7)
        diff = rng_d.standard_normal(n) * 0.1 + 0.01
        old_peak = _peak_bytes(_old_monolithic, diff, n_boot, alpha, np.random.default_rng(0))
        new_peak = _peak_bytes(_paired_bootstrap_ci, diff, n_boot, alpha, np.random.default_rng(0))
        print(f"\nn={n:,} n_boot={n_boot}: peak-temp OLD={old_peak/1e6:.1f} MB  NEW={new_peak/1e6:.1f} MB  ratio={old_peak/max(new_peak,1):.1f}x")

    # Analytic prod-shape temp size (OLD path would OOM most boxes).
    n, n_boot, block = 1_000_000, 1000, 64
    old_temp = n_boot * n * 8 * 2  # idx int64 + gather float64
    new_temp = block * n * 8 * 2
    print(f"\nPROD shape n={n:,} n_boot={n_boot}: OLD peak-temp ~{old_temp/1e9:.1f} GB  NEW peak-temp ~{new_temp/1e9:.3f} GB  ({old_temp/new_temp:.0f}x reduction)")

    # Wall time at moderate shape.
    n, n_boot = 100_000, 1000
    rng_d = np.random.default_rng(7)
    diff = rng_d.standard_normal(n) * 0.1 + 0.01
    old_min, old_med = _best_of(lambda: _old_monolithic(diff, n_boot, alpha, np.random.default_rng(0)), ())
    new_min, new_med = _best_of(lambda: _paired_bootstrap_ci(diff, n_boot, alpha, np.random.default_rng(0)), ())
    print(f"\nWall n={n:,} n_boot={n_boot}: OLD min={old_min*1e3:.1f}ms med={old_med*1e3:.1f}ms  NEW min={new_min*1e3:.1f}ms med={new_med*1e3:.1f}ms")

    # Identity at bench scale.
    a = _old_monolithic(diff, n_boot, alpha, np.random.default_rng(123))
    b = _paired_bootstrap_ci(diff, n_boot, alpha, np.random.default_rng(123))
    print(f"\nCI identity OLD={a} NEW={b}  bit-identical={a == b}")


if __name__ == "__main__":
    main()
