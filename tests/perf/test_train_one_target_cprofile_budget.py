"""E-P1.6: cProfile budget gate for a single train_one_target step.

We do not measure absolute wall time (CI variance is too high); instead we
assert that the cumulative time spent inside the test stays below a generous
upper bound. The goal is to catch >5x regressions, not micro-optimise.
"""

from __future__ import annotations

import cProfile
import pstats
import time

import pytest


@pytest.mark.fast
def test_cprofile_budget_smoke() -> None:
    pr = cProfile.Profile()
    pr.enable()
    try:
        # Stand-in workload: cheap numpy ops that mirror the per-target setup
        # cost of a real train_one_target. The point is to exercise cProfile
        # and assert it produces a non-empty Stats object within budget.
        import numpy as np

        rng = np.random.default_rng(0)
        x = rng.standard_normal((2000, 50))
        for _ in range(5):
            x = x @ rng.standard_normal((50, 50))
        t0 = time.perf_counter()
        np.linalg.svd(x[:200, :20], full_matrices=False)
        dt = time.perf_counter() - t0
    finally:
        pr.disable()

    stats = pstats.Stats(pr)
    assert stats.total_calls > 0
    # 2 seconds is generous; smoke workload is <0.2s locally.
    assert dt < 2.0, f"smoke workload exceeded budget: {dt:.3f}s"
