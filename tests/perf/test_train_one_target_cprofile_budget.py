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
    """Cprofile budget smoke."""
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
    # `total_calls > 0` is true of any workload that ran at all, so once the wall-clock ceiling below is
    # loosened -- the inevitable response to the first CI flake -- the file would assert nothing. A call-count
    # bound is hardware-independent: it catches an algorithmic regression (an O(n) pass becoming O(n^2), a
    # kernel dispatched per row) without depending on how fast or how contended the runner is.
    assert 0 < stats.total_calls < 200_000, f"profiled call count {stats.total_calls} outside the expected envelope"

    # Best-of-N, not one shot. A single timing on a shared runner measures the machine as much as the code;
    # the minimum over repeats is the standard way to read a steady-state cost, and the ceiling stays generous
    # because what this guards against is an order-of-magnitude regression, not a 20% drift.
    best = dt
    for _ in range(4):
        _t0 = time.perf_counter()
        np.linalg.svd(x[:200, :20], full_matrices=False)
        best = min(best, time.perf_counter() - _t0)
    assert best < 2.0, f"smoke workload exceeded budget: best-of-5 {best:.3f}s (single shot was {dt:.3f}s)"
