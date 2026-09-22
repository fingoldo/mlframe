"""cProfile harness for the suite-level zero-inflation detection in ``training.composite._hurdle_dispatch``.

Run: ``python -m mlframe.training.composite._benchmarks.bench_hurdle_dispatch``

Detection runs on every suite call with a regression target, so its cost is paid even when no hurdle is added.

Verdict (8 regression targets, train rows = 75%): the first version sampled 200k rows and ran ``np.unique`` per target --
217 ms at 1M rows, of which the no-replacement sample was 69 ms and the sort 85 ms. Because the atom must be the minimum and
the threshold is at least one half, the test reduces to "does the minimum hold that share": one comparison pass, no sort,
no sampling. 1.1 / 4.9 / 65 ms at 10k / 100k / 1M rows (3.3x at 1M, 7x at 100k); the remaining cost is the float64 copy
and ``isfinite`` filter per target.
"""
from __future__ import annotations

import cProfile
import pstats
import time
from io import StringIO

import numpy as np

from mlframe.training._configs_base import TargetTypes
from mlframe.training.composite._hurdle_dispatch import zero_inflated_regression_targets


def _targets(n: int, k: int, seed: int = 0) -> dict:
    """``k`` regression targets of ``n`` rows, alternating zero-inflated and continuous."""
    rng = np.random.default_rng(seed)
    out = {}
    for j in range(k):
        y = rng.lognormal(3.0, 1.0, n)
        if j % 2 == 0:
            y[rng.random(n) < 0.7] = 0.0
        out[f"t{j}"] = y
    return {TargetTypes.REGRESSION: out}


if __name__ == "__main__":
    for n in (10_000, 100_000, 1_000_000):
        tb = _targets(n, 8)
        idx = np.arange(int(n * 0.75))
        t0 = time.perf_counter()
        found = zero_inflated_regression_targets(tb, idx)
        print(f"n={n:>9} targets=8 -> {(time.perf_counter() - t0) * 1000:8.1f} ms, zero-inflated: {sum(len(v) for v in found.values())}")

    tb = _targets(1_000_000, 8)
    idx = np.arange(750_000)
    pr = cProfile.Profile()
    pr.enable()
    zero_inflated_regression_targets(tb, idx)
    pr.disable()
    s = StringIO()
    pstats.Stats(pr, stream=s).sort_stats("cumulative").print_stats(12)
    print(s.getvalue())
