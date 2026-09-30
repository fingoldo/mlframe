"""cProfile + wall harness for ``futility_verdict``: the per-iteration cost of the stop check vs. one RFECV iteration.

Run: python -m mlframe.feature_selection.wrappers.rfecv._benchmarks.profile_futility_stop
"""
from __future__ import annotations

import cProfile
import pstats
import time

import numpy as np

from mlframe.feature_selection.wrappers.rfecv._futility_stop import futility_verdict


def _trace(n_iters: int, k: int, p: int = 88, seed: int = 0) -> list:
    rng = np.random.default_rng(seed)
    eff = rng.normal(0, 0.02, k)
    return [(p - 3 * i, tuple(0.8 + eff + rng.normal(0, 0.002, k))) for i in range(n_iters)]


def main() -> None:
    for n_iters, k in ((10, 3), (30, 5), (100, 5)):
        tr = _trace(n_iters, k)
        futility_verdict(tr, remaining=20, full_n=88)  # warm scipy imports/caches
        t0 = time.perf_counter()
        reps = 200
        for _ in range(reps):
            futility_verdict(tr, remaining=20, full_n=88)
        per_call_us = (time.perf_counter() - t0) / reps * 1e6
        print(f"iters={n_iters:4d} k={k}: {per_call_us:8.1f} us/call (an RFECV iteration is >= 1e5 us even on toy data)")
    tr = _trace(100, 5)
    pr = cProfile.Profile()
    pr.enable()
    for _ in range(200):
        futility_verdict(tr, remaining=20, full_n=88)
    pr.disable()
    pstats.Stats(pr).sort_stats("cumtime").print_stats(12)


if __name__ == "__main__":
    main()
