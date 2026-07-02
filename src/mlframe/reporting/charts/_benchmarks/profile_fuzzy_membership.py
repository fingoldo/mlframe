"""cProfile harness for the fuzzy-partition membership chart (charts/fuzzy_membership.py).

Run: ``python -m mlframe.reporting.charts._benchmarks.profile_fuzzy_membership``

The curve evaluation is ``grid * n_partitions`` (tiny -- a few hundred triangular tent lookups); the only length-n work
is the quantile centre fit, which is already capped by the <=200k subsample gate. This harness confirms the wall is
dominated by ``np.quantile`` (the O(subsample log subsample) sort) and the bounded subsample gather, not by the grid
evaluation, so there is no actionable per-call speedup: the fit is one sort of at most 200k values and the transform is a
sub-millisecond njit tent pass.

Measured (n=5M, cap 200k, 5 sets, grid 200): 39.7 ms/call. tottime breaks down as the length-n ``np.isfinite`` mask +
bounded subsample gather over the 5M input (~0.16 s / 5 calls, the genuine floor -- it must scan the column once) and the
capped ``np.quantile`` partition sort over 200k values (~0.027 s / 5). The grid tent evaluation does not appear in the top
12. Verdict: no actionable speedup -- the cost is the unavoidable one-pass finite scan plus a bounded 200k sort.
"""

from __future__ import annotations

import cProfile
import pstats
import time
from io import StringIO

import numpy as np

from mlframe.reporting.charts.fuzzy_membership import fuzzy_membership_curves


def main(n: int = 5_000_000):
    x = np.random.default_rng(0).normal(size=n)
    fuzzy_membership_curves(x, n_partitions=5, grid=200)  # warmup (njit compile + subsample path)

    t0 = time.perf_counter()
    for _ in range(5):
        fuzzy_membership_curves(x, n_partitions=5, grid=200)
    wall = (time.perf_counter() - t0) / 5.0
    print(f"fuzzy_membership_curves @ n={n} (cap 200k, 5 sets, grid 200): {wall*1000:.1f} ms/call (mean of 5)")

    pr = cProfile.Profile()
    pr.enable()
    for _ in range(5):
        fuzzy_membership_curves(x, n_partitions=5, grid=200)
    pr.disable()
    s = StringIO()
    pstats.Stats(pr, stream=s).sort_stats("tottime").print_stats(12)
    print(s.getvalue())


if __name__ == "__main__":
    main()
