"""Bench + identity check for the O(grid)-saddle IVAP envelope vs the prior per-grid-
point sklearn ``IsotonicRegression`` refit loop.

CPX-P0-3: ``_isotonic_envelopes`` used to fit TWO full sklearn isotonic regressions
per grid point (label-0 and label-1 augmented), i.e. O(grid) PAV refits each O(n log n)
-> O(n^2 log n) overall, despite the docstring claiming O(n log n). The replacement
computes both envelopes from the cumulative-sum-diagram corners via a single
``max_l min_r`` greatest-convex-minorant saddle (``_ivap_saddle_njit``), bit-exact to
the sklearn refit (validated to ~1e-16, single-ULP division order).

Run: python src/mlframe/training/composite/_benchmarks/bench_venn_abers_ivap_envelopes.py
"""
from __future__ import annotations

import subprocess
import time

import numpy as np


def _load_head_baseline():
    """The REAL prior ``_isotonic_envelopes`` (sklearn-refit loop) from git HEAD."""
    src = subprocess.run(
        ["git", "show", "HEAD:src/mlframe/training/composite/venn_abers.py"],
        capture_output=True, text=True,
    ).stdout
    ns: dict = {}
    exec(compile(src, "venn_abers_head", "exec"), ns)
    return ns["_isotonic_envelopes"]


def main():
    from mlframe.training.composite.venn_abers import _isotonic_envelopes as new_env

    old_env = _load_head_baseline()
    # Warm numba.
    new_env(np.array([0.1, 0.2, 0.3]), np.array([0.0, 1.0, 1.0]))

    print(f"{'n':>6} {'old(ms)':>10} {'new(ms)':>9} {'speedup':>9} {'max|d_p0|':>11} {'max|d_p1|':>11}")
    for n in (200, 1000, 3000):
        rng = np.random.default_rng(3)
        s = np.sort(rng.uniform(0, 1, n))
        y = (rng.uniform(0, 1, n) < s).astype(float)

        # Identity.
        _, lo_o, hi_o = old_env(s, y)
        _, lo_n, hi_n = new_env(s, y)
        d0 = np.abs(lo_o - lo_n).max()
        d1 = np.abs(hi_o - hi_n).max()

        t0 = time.perf_counter(); old_env(s, y); t_old = time.perf_counter() - t0
        best = float("inf")
        for _ in range(5):
            t0 = time.perf_counter(); new_env(s, y); best = min(best, time.perf_counter() - t0)
        t_new = best

        print(f"{n:>6} {t_old*1e3:>10.1f} {t_new*1e3:>9.3f} {t_old/t_new:>8.1f}x {d0:>11.2e} {d1:>11.2e}")
        assert d0 < 1e-9 and d1 < 1e-9, "IVAP envelope must match the sklearn refit within ~1e-9"


if __name__ == "__main__":
    main()
