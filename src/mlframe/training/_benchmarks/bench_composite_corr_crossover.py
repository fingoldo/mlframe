"""Multi-size warm best-of-N crossover of the composite abs-corr numpy reference vs the numba kernel; checks the fallback gate (20k rows, 64 cols)."""

from __future__ import annotations

import os
import time

import numpy as np

from mlframe.training.composite.discovery import _corr_numba as cn
from mlframe.training.composite.discovery.screening import _safe_abs_corr_all, _safe_abs_corr_all_numpy


def _best(fn, reps=7):
    """Best perf_counter over reps."""
    b = 1e9
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        b = min(b, time.perf_counter() - t)
    return b


def main():
    """Print numpy vs numba timings and the max abs difference per size."""
    rng = np.random.default_rng(0)
    cn._abs_corr_all_kernel(rng.normal(size=(100, 4)), rng.normal(size=100), 1.0, 1e-9)
    os.environ["MLFRAME_COMPOSITE_CORR_BACKEND"] = "numba"
    for n in (500, 2_000, 5_000, 20_000, 100_000):
        for f in (2, 4, 8, 16, 32, 64, 128):
            X = rng.normal(size=(n, f))
            y = rng.normal(size=n)
            ref = _safe_abs_corr_all_numpy(y, X)
            kern = _safe_abs_corr_all(y, X)
            tn = tb = 1e9
            for _ in range(3):
                tn = min(tn, _best(lambda: _safe_abs_corr_all_numpy(y, X)))
                tb = min(tb, _best(lambda: _safe_abs_corr_all(y, X)))
            print(f"n={n} F={f} numpy={tn*1e3:.2f}ms numba={tb*1e3:.2f}ms ratio={tn/tb:.2f} maxdiff={np.abs(ref-kern).max():.1e}")


if __name__ == "__main__":
    main()
