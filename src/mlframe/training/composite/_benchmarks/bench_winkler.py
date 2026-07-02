"""cProfile harness for the Winkler interval score at a production-representative shape.

Run: ``CUDA_VISIBLE_DEVICES="" python -m mlframe.training.composite._benchmarks.bench_winkler``

Shape: 1M rows / 500 groups. Warms the njit kernels first (cache=True), then profiles the mean score, the
per-group score, and the summary. The mean unweighted score delegates to the already-optimized
``mlframe.metrics.quantile.winkler_score`` njit kernel; the weighted / per-group / per-row paths are fused
njit sweeps here. ``pd.factorize`` of the group ids is the only O(n) non-kernel cost. See the module docstring
for the verdict.
"""
from __future__ import annotations

import cProfile
import pstats
import time

import numpy as np

from mlframe.training.composite import _winkler as wk


def _make(n=1_000_000, n_groups=500, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.normal(size=n)
    lo = y - 1.6 - rng.random(n)
    hi = y + 1.6 + rng.random(n)
    lo[::10] += 3.0  # inject misses
    g = rng.integers(0, n_groups, n)
    return y, lo, hi, g


def main() -> None:
    y, lo, hi, g = _make()
    wk.winkler_interval_score(y[:1000], lo[:1000], hi[:1000], 0.1)  # warm
    wk.winkler_score_per_group(y[:1000], lo[:1000], hi[:1000], 0.1, g[:1000])

    for label, fn in (
        ("mean", lambda: wk.winkler_interval_score(y, lo, hi, 0.1)),
        ("weighted", lambda: wk.winkler_interval_score(y, lo, hi, 0.1, sample_weight=np.ones(y.shape[0]))),
        ("per_group", lambda: wk.winkler_score_per_group(y, lo, hi, 0.1, g)),
    ):
        t0 = time.perf_counter()
        fn()
        print(f"warm wall: {label} {(time.perf_counter() - t0) * 1e3:.1f} ms  (n={y.shape[0]})")

    pr = cProfile.Profile()
    pr.enable()
    for _ in range(5):
        wk.winkler_interval_score(y, lo, hi, 0.1)
        wk.winkler_score_per_group(y, lo, hi, 0.1, g)
    pr.disable()
    pstats.Stats(pr).sort_stats("cumulative").print_stats(12)


if __name__ == "__main__":
    main()
